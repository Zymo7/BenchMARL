from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import List, Optional, Type

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig


class HebbianLayer(nn.Module):
    """Single Hebbian layer with ABCD weight update rule.

    delta_W[i,j] = A[i,j] * pre[i] * post[j] + B[i,j] * pre[i] + C[i,j] * post[j] + D[i,j]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        w_max: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        # Per-element upper bound on |W| applied after every Hebbian update.
        # Prevents W from drifting unboundedly across an episode (which is what
        # made the CMA-ES policy collapse to a near-constant action). Set to a
        # non-positive value to disable.
        self.w_max = w_max

        W = torch.randn(in_features, out_features) / (in_features ** 0.5)
        self.register_buffer("W_init", W.clone())
        self.register_buffer("W", W)

        self.register_buffer("A", torch.zeros(in_features, out_features))
        self.register_buffer("B", torch.zeros(in_features, out_features))
        self.register_buffer("C", torch.zeros(in_features, out_features))
        self.register_buffer("D", torch.zeros(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W.data
        output = x @ W

        with torch.no_grad():
            pre = x.detach().mean(dim=tuple(range(x.ndim - 1)))
            post = output.detach().mean(dim=tuple(range(output.ndim - 1)))

            outer = pre.unsqueeze(1) * post.unsqueeze(0)
            delta_W = (
                self.A * outer
                + self.B * pre.unsqueeze(1)
                + self.C * post.unsqueeze(0)
                + self.D
            )

            if delta_W.abs().sum() > 0:
                new_W = W + self.lr_hebb * delta_W
                if self.w_max > 0:
                    new_W = new_W.clamp_(min=-self.w_max, max=self.w_max)
                self.W.data = new_W.clone()

        return output

    def reset_weights(self):
        with torch.no_grad():
            W = self.W_init
            if self.w_max > 0:
                W = W.clamp(min=-self.w_max, max=self.w_max)
            self.W.copy_(W)

    def get_abcd_vector(self) -> torch.Tensor:
        return torch.cat([
            self.A.flatten(),
            self.B.flatten(),
            self.C.flatten(),
            self.D.flatten(),
        ])

    def set_abcd_from_vector(self, vector: torch.Tensor):
        n = self.in_features * self.out_features
        with torch.no_grad():
            self.A.copy_(vector[:n].reshape(self.in_features, self.out_features))
            self.B.copy_(vector[n:2*n].reshape(self.in_features, self.out_features))
            self.C.copy_(vector[2*n:3*n].reshape(self.in_features, self.out_features))
            self.D.copy_(vector[3*n:4*n].reshape(self.in_features, self.out_features))

    @property
    def num_abcd_params(self) -> int:
        return 4 * self.in_features * self.out_features


class FullHebbianModel(Model):
    """Multi-layer network where ALL layers are Hebbian layers.

    Architecture: input -> hidden(hidden_size) -> hidden(hidden_size) -> output
    All weight matrices use the ABCD Hebbian update rule.
    ABCD parameters are optimized by CMA-ES.
    """

    def __init__(
        self,
        hidden_size: int = 9,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        w_max: float = 1.0,
        activation_class: Type[nn.Module] = None,
        activation_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        self.w_max = w_max
        self.num_feature_dims = kwargs.pop("num_feature_dims", 1)

        super().__init__(
            input_spec=kwargs.pop("input_spec"),
            output_spec=kwargs.pop("output_spec"),
            agent_group=kwargs.pop("agent_group"),
            input_has_agent_dim=kwargs.pop("input_has_agent_dim"),
            n_agents=kwargs.pop("n_agents"),
            centralised=kwargs.pop("centralised"),
            share_params=kwargs.pop("share_params"),
            device=kwargs.pop("device"),
            action_spec=kwargs.pop("action_spec"),
            model_index=kwargs.pop("model_index"),
            is_critic=kwargs.pop("is_critic"),
        )

        self.input_features = sum(
            [
                torch.prod(torch.tensor(spec.shape[-self.num_feature_dims :])).item()
                for spec in self.input_spec.values(True, True)
            ]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        # Three Hebbian layers: input->hidden, hidden->hidden, hidden->output
        self.layers = nn.ModuleList([
            HebbianLayer(self.input_features, self.hidden_size, self.lr_hebb, self.weight_init, self.w_max),
            HebbianLayer(self.hidden_size, self.hidden_size, self.lr_hebb, self.weight_init, self.w_max),
            HebbianLayer(self.hidden_size, self.output_features, self.lr_hebb, self.weight_init, self.w_max),
        ])

        if activation_class is not None:
            self.activation = activation_class(**(activation_kwargs or {}))
        else:
            self.activation = nn.Tanh()

        if self.input_has_agent_dim and not self.share_params:
            self.per_agent_layers = nn.ModuleList([
                nn.ModuleList([
                    HebbianLayer(self.input_features, self.hidden_size, self.lr_hebb, self.weight_init, self.w_max),
                    HebbianLayer(self.hidden_size, self.hidden_size, self.lr_hebb, self.weight_init, self.w_max),
                    HebbianLayer(self.hidden_size, self.output_features, self.lr_hebb, self.weight_init, self.w_max),
                ])
                for _ in range(self.n_agents)
            ])

    def _perform_checks(self):
        super()._perform_checks()

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = torch.cat(
            [
                torch.flatten(tensordict.get(in_key), start_dim=-self.num_feature_dims)
                for in_key in self.in_keys
            ],
            dim=-1,
        )

        if self.input_has_agent_dim and not self.share_params:
            res = torch.stack(
                [self._forward_through_layers(x, layers) for layers in self.per_agent_layers],
                dim=-2,
            )
        else:
            res = self._forward_through_layers(x, self.layers)
            if self.input_has_agent_dim and not self.output_has_agent_dim:
                res = res[..., 0, :]

        tensordict.set(self.out_key, res)
        return tensordict

    def _forward_through_layers(self, x: torch.Tensor, layers) -> torch.Tensor:
        # Layer 1: input -> hidden (with activation)
        x = layers[0](x)
        if self.activation is not None:
            x = self.activation(x)
        # Layer 2: hidden -> hidden (with activation)
        x = layers[1](x)
        if self.activation is not None:
            x = self.activation(x)
        # Layer 3: hidden -> output (no activation)
        x = layers[2](x)
        return x

    def get_all_hebbian_layers(self) -> List[HebbianLayer]:
        """Return all HebbianLayer instances in the network."""
        return list(self.layers)

    def get_abcd_vector(self) -> torch.Tensor:
        """Flatten ABCD parameters from ALL layers into a single vector."""
        return torch.cat([layer.get_abcd_vector() for layer in self.layers])

    def set_abcd_from_vector(self, vector: torch.Tensor):
        """Set ABCD parameters for ALL layers from a flat vector."""
        offset = 0
        for layer in self.layers:
            n = layer.num_abcd_params
            layer.set_abcd_from_vector(vector[offset:offset + n])
            offset += n

    def reset_all_weights(self):
        """Reset plastic weights in all layers to initial values."""
        for layer in self.layers:
            layer.reset_weights()

    @property
    def total_abcd_params(self) -> int:
        return sum(layer.num_abcd_params for layer in self.layers)


@dataclass
class FullHebbianConfig(ModelConfig):
    """Config for :class:`~benchmarl.models.FullHebbianModel`."""

    hidden_size: int = 9
    lr_hebb: float = 0.01
    weight_init: float = 1.0
    w_max: float = 1.0
    activation_class: Type[nn.Module] = None
    activation_kwargs: Optional[dict] = None
    num_feature_dims: int = 1

    @staticmethod
    def associated_class():
        return FullHebbianModel
