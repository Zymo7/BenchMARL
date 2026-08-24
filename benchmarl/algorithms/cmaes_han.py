from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Type

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import Categorical
from torchrl.data import Composite, Unbounded
from torchrl.modules import IndependentNormal, ProbabilisticActor, TanhNormal
from torchrl.modules.distributions import MaskedCategorical
from torchrl.objectives import LossModule

from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


class CmaesHan(Algorithm):
    """CMA-ES trained Hebbian Attractor Network (HAN).

    Single-phase training: CMA-ES directly optimizes all ABCD Hebbian
    parameters. The plastic weights ``W`` of each :class:`HanLayer` evolve
    online during inference via the HAN rule (decoupled ``update_weights``
    with a sliding window of pre/post activations and layer-wise
    max-abs normalization). No gradient-based (PPO) phase.

    The policy is a deterministic actor wrapped in a ``ProbabilisticActor``
    for compatibility with the experiment framework.
    """

    def __init__(
        self,
        scale_mapping: str,
        use_tanh_normal: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scale_mapping = scale_mapping
        self.use_tanh_normal = use_tanh_normal

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        # CMA-ES does not use gradient-based loss, but we need to provide
        # a dummy loss for the framework. We use a simple PPO-style loss
        # on zeros, mirroring CmaesHebbian.
        from torchrl.objectives import ClipPPOLoss, ValueEstimators
        from benchmarl.models import MlpConfig

        n_agents = len(self.group_map[group])
        critic_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        critic_output_spec = Composite(
            {
                group: Composite(
                    {"state_value": Unbounded(shape=(n_agents, 1))},
                    shape=(n_agents,),
                )
            }
        )
        critic_model_config = self.critic_model_config
        value_module = critic_model_config.get_model(
            input_spec=critic_input_spec,
            output_spec=critic_output_spec,
            n_agents=n_agents,
            centralised=False,
            input_has_agent_dim=True,
            agent_group=group,
            share_params=True,
            device=self.device,
            action_spec=self.action_spec,
        )

        loss_module = ClipPPOLoss(
            actor=policy_for_loss,
            critic=value_module,
            clip_epsilon=0.2,
            entropy_coeff=0.0,
            critic_coeff=1.0,
            loss_critic_type="l2",
            normalize_advantage=False,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=0.9
        )
        return loss_module, False

    def _get_parameters(self, group: str, loss) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        n_agents = len(self.group_map[group])
        if continuous:
            logits_shape = list(self.action_spec[group, "action"].shape)
            logits_shape[-1] *= 2
        else:
            logits_shape = [
                *self.action_spec[group, "action"].shape,
                self.action_spec[group, "action"].space.n,
            ]

        actor_input_spec = Composite(
            {group: self.observation_spec[group].clone().to(self.device)}
        )
        actor_output_spec = Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=logits_shape)},
                    shape=(n_agents,),
                )
            }
        )
        actor_module = model_config.get_model(
            input_spec=actor_input_spec,
            output_spec=actor_output_spec,
            agent_group=group,
            input_has_agent_dim=True,
            n_agents=n_agents,
            centralised=False,
            share_params=self.experiment_config.share_policy_params,
            device=self.device,
            action_spec=self.action_spec,
        )

        if continuous:
            extractor_module = TensorDictModule(
                NormalParamExtractor(scale_mapping=self.scale_mapping),
                in_keys=[(group, "logits")],
                out_keys=[(group, "loc"), (group, "scale")],
            )
            policy = ProbabilisticActor(
                module=TensorDictSequential(actor_module, extractor_module),
                spec=self.action_spec[group, "action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=(
                    IndependentNormal if not self.use_tanh_normal else TanhNormal
                ),
                distribution_kwargs=(
                    {
                        "low": self.action_spec[(group, "action")].space.low,
                        "high": self.action_spec[(group, "action")].space.high,
                    }
                    if self.use_tanh_normal
                    else {}
                ),
                return_log_prob=True,
                log_prob_key=(group, "log_prob"),
            )
        else:
            if self.action_mask_spec is None:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys=[(group, "logits")],
                    out_keys=[(group, "action")],
                    distribution_class=Categorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )
            else:
                policy = ProbabilisticActor(
                    module=actor_module,
                    spec=self.action_spec[group, "action"],
                    in_keys={
                        "logits": (group, "logits"),
                        "mask": (group, "action_mask"),
                    },
                    out_keys=[(group, "action")],
                    distribution_class=MaskedCategorical,
                    return_log_prob=True,
                    log_prob_key=(group, "log_prob"),
                )

        return policy

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        return policy_for_loss

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape

        nested_done_key = ("next", group, "done")
        nested_terminated_key = ("next", group, "terminated")
        nested_reward_key = ("next", group, "reward")

        if nested_done_key not in keys:
            batch.set(
                nested_done_key,
                batch.get(("next", "done")).unsqueeze(-1).expand((*group_shape, 1)),
            )
        if nested_terminated_key not in keys:
            batch.set(
                nested_terminated_key,
                batch.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand((*group_shape, 1)),
            )
        if nested_reward_key not in keys:
            batch.set(
                nested_reward_key,
                batch.get(("next", "reward")).unsqueeze(-1).expand((*group_shape, 1)),
            )

        loss = self.get_loss_and_updater(group)[0]
        with torch.no_grad():
            loss.value_estimator(batch, params=loss.critic_network_params)

        return batch

    def get_han_model(self, group: str = None):
        """Navigate the policy to find the HanModel instance."""
        if group is None:
            group = list(self.group_map.keys())[0]

        policy = self._policies_for_loss.get(group)
        if policy is None:
            policy = self.get_policy_for_loss(group)

        from benchmarl.models.han import HanModel
        for sub_module in policy.module.modules():
            if isinstance(sub_module, HanModel):
                return sub_module

        return None

    def get_hgn_model(self, group: str = None):
        """Navigate the policy to find the HgnModel instance (sister of
        :meth:`get_han_model`)."""
        if group is None:
            group = list(self.group_map.keys())[0]

        policy = self._policies_for_loss.get(group)
        if policy is None:
            policy = self.get_policy_for_loss(group)

        from benchmarl.models.hgn import HgnModel
        for sub_module in policy.module.modules():
            if isinstance(sub_module, HgnModel):
                return sub_module

        return None


@dataclass
class CmaesHanConfig(AlgorithmConfig):
    """Config for :class:`~benchmarl.algorithms.CmaesHan`."""

    scale_mapping: str = "biased_softplus_1.0"
    use_tanh_normal: bool = True

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        import pathlib
        from benchmarl.utils import _read_yaml_config

        if path is None:
            yaml_path = (
                pathlib.Path(__file__).parent.parent
                / "conf"
                / "algorithm"
                / "cmaes_han.yaml"
            )
            config = _read_yaml_config(str(yaml_path.resolve()))
        else:
            config = _read_yaml_config(path)
        return cls(**config)

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return CmaesHan

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return True

    @staticmethod
    def on_policy() -> bool:
        return True

    @staticmethod
    def has_independent_critic() -> bool:
        return True
