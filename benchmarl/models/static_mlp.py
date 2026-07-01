#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Static MLP baseline for the HAN comparison.

This module provides a thin wrapper around :class:`benchmarl.models.Mlp`
and :class:`benchmarl.models.MlpConfig`. The architecture is identical
to ``Mlp``: a stack of fully-connected layers with a chosen activation.
**There is no Hebbian plasticity, no ABCD parameters, and no online
weight mutation** — every weight is updated only by the gradient
descent of the chosen algorithm (e.g. PPO) or by an external
optimization algorithm such as CMA-ES.

The model is registered under its own name (``static_mlp``) so the
comparison plots, training meta, and experiment folders can be labelled
explicitly as ``HAN vs static-MLP``. The contrast is "plastic Hebbian
weights" vs "fixed weights updated only by gradient/optimizer".

We also add the same flat-vector API as :class:`HanModel` —
``get_weights_vector``, ``set_weights_from_vector``, ``reset_weights``,
and ``total_weights`` — so the same
:class:`benchmarl.algorithms.CmaesStaticMlpOptimizer` can train this
network with CMA-ES on a flat weight vector, in lock-step with the
CMA-ES / HAN optimizer's training dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import List, Optional, Sequence, Type

import torch
from torch import nn
from torchrl.data import Composite

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.mlp import Mlp, MlpConfig
from benchmarl.utils import DEVICE_TYPING


class StaticMlpModel(Mlp):
    """Multi-layer perceptron without plasticity.

    Identical forward pass to :class:`benchmarl.models.Mlp`. The only
    addition is a CMA-ES-friendly flat-vector API used by
    :class:`CmaesStaticMlpOptimizer`:

        get_weights_vector()        : float32 1-D tensor of all params
        set_weights_from_vector(x)  : load_state_dict-shaped load of x
        reset_weights()             : restore the initial random init
        total_weights               : property returning int dimension

    These mirror :class:`HanModel`'s ``get_abcd_vector`` /
    ``set_abcd_from_vector`` / ``reset_all_weights`` API but on plain
    MLP weights instead of ABCD Hebbian parameters.

    Note: we snapshot the *initial* parameters inside ``__init__`` so
    that :meth:`reset_weights` can restore the model to the exact
    random state that was used to build the network. We do this rather
    than relying on a lazy first-call snapshot because BenchMARL may
    invoke :meth:`set_weights_from_vector` (e.g. for CMA-ES) before
    the first :meth:`reset_weights` call, which would otherwise capture
    the perturbed state.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Snapshot the *current* (just-initialized) parameter tensors.
        # Mlp has already built ``self.mlp`` by this point and the
        # parameters are real, not lazy, so cloning them gives a clean
        # copy of the random init.
        self._weight_init_state = [
            p.detach().clone() for p in self.parameters()
        ]

    def reset_weights(self) -> None:
        """Reset all MLP parameters to the initial random snapshot.

        We copy the cloned init tensors back into the live parameters.
        This works for the underlying ``MultiAgentMLP`` because
        ``self.parameters()`` returns references to the same tensors
        that the forward pass consumes.
        """
        with torch.no_grad():
            for p, init in zip(self.parameters(), self._weight_init_state):
                p.copy_(init)

    def get_weights_vector(self) -> torch.Tensor:
        """Flatten all trainable parameters into a single 1-D float32 tensor.

        We read via ``self.parameters()`` rather than ``state_dict()``:
        ``MultiAgentMLP`` stores its params in a TensorDict with a
        meta-device ``_empty_net``, and reading through ``state_dict()``
        does not always materialise the latest write. Iterating
        ``self.parameters()`` always returns the live tensors.
        """
        return torch.cat([
            p.detach().flatten().float() for p in self.parameters()
        ])

    def set_weights_from_vector(self, vector: torch.Tensor) -> None:
        """Load a flat vector back into the network parameters.

        The vector must have shape ``(total_weights,)``; we split it
        into per-parameter chunks matching the iteration order of
        ``self.parameters()``.
        """
        params = list(self.parameters())
        sizes = [p.numel() for p in params]
        flat = vector.detach().flatten().float()
        if flat.numel() != sum(sizes):
            raise ValueError(
                f"vector size {flat.numel()} does not match "
                f"model param sum {sum(sizes)}"
            )
        offset = 0
        with torch.no_grad():
            for p, n in zip(params, sizes):
                chunk = flat[offset:offset + n].view_as(p)
                p.copy_(chunk)
                offset += n

    @property
    def total_weights(self) -> int:
        """Total number of trainable parameters (int)."""
        return sum(p.numel() for p in self.parameters())


@dataclass
class StaticMlpConfig(ModelConfig):
    """Dataclass config for :class:`StaticMlpModel`.

    Field semantics are identical to :class:`benchmarl.models.MlpConfig`;
    we re-declare them so the config type is distinct (for hydra / yaml
    registration) and so the documentation can refer to "static"
    semantics (no plasticity) explicitly.

    The ``bias`` flag (default ``True``) controls whether the linear
    layers include a bias term. The HAN-vs-static-MLP comparison uses
    ``bias=False`` to match HAN's parameter count exactly: HAN's W
    matrix has no bias, its ABCD parameters apply directly to a
    ``(in, out)`` weight matrix, so a bias-free ``nn.Linear``-stacked
    MLP gives the same parameter count.
    """

    num_cells: Sequence[int] = MISSING
    layer_class: Type[nn.Module] = MISSING

    activation_class: Type[nn.Module] = MISSING
    activation_kwargs: Optional[dict] = None

    norm_class: Type[nn.Module] = None
    norm_kwargs: Optional[dict] = None

    num_feature_dims: int = 1
    layer_kwargs: Optional[dict] = None
    bias: bool = True

    @staticmethod
    def associated_class():
        return StaticMlpModel

    def get_model(
        self,
        input_spec: Composite,
        output_spec: Composite,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: Composite,
        model_index: int = 0,
    ):
        """Construct the model, applying ``bias=False`` if requested.

        We replicate :meth:`ModelConfig.get_model` here so we can drop
        the ``bias`` field (which torchrl's MLP does not accept as a
        top-level kwarg) and inject ``{"bias": False}`` into
        ``layer_kwargs`` instead — torchrl's MLP pops that key from
        ``layer_kwargs`` to drive ``nn.Linear(bias=...)`` for every
        layer.
        """
        from dataclasses import asdict
        cfg_dict = asdict(self)
        cfg_dict.pop("bias", None)  # torchrl's MLP does not accept this.
        if not self.bias:
            lk = dict(cfg_dict.get("layer_kwargs") or {})
            lk["bias"] = False
            cfg_dict["layer_kwargs"] = lk
        return self.associated_class()(
            **cfg_dict,
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
            model_index=model_index,
            is_critic=self.is_critic,
        )
