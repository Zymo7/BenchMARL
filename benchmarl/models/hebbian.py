from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Optional, Type

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig


class HebbianLayer(nn.Module):
    """Hebbian learning layer with ABCD weight update rule.

    The weight update rule for each connection (i, j):
        delta_W[i,j] = A[i,j] * pre[i] * post[j] + B[i,j] * pre[i] + C[i,j] * post[j] + D[i,j]

    W is the plastic weight matrix that updates online during forward passes.
    A, B, C, D are the Hebbian parameters that control how W updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr_hebb: float = 0.01,
        weight_init: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init

        # Plastic weights W: updated by Hebbian rule, not by gradients
        # Initialize with Kaiming-like scaling so output magnitude is reasonable
        W = torch.randn(in_features, out_features) / (in_features ** 0.5)
        self.register_buffer("W_init", W.clone())
        self.register_buffer("W", W)

        # ABCD Hebbian parameters: optimized by CMA-ES in Phase 2
        self.register_buffer("A", torch.zeros(in_features, out_features))
        self.register_buffer("B", torch.zeros(in_features, out_features))
        self.register_buffer("C", torch.zeros(in_features, out_features))
        self.register_buffer("D", torch.zeros(in_features, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output with current weights (W is a buffer, not a parameter)
        W = self.W.data
        output = x @ W

        # Update W using Hebbian ABCD rule
        # Only update if ABCD has non-zero values (skip during Phase 1 when ABCD=0)
        with torch.no_grad():
            pre = x.detach().mean(dim=tuple(range(x.ndim - 1)))  # (in_features,)
            post = output.detach().mean(dim=tuple(range(output.ndim - 1)))  # (out_features,)

            outer = pre.unsqueeze(1) * post.unsqueeze(0)  # (in_features, out_features)
            delta_W = self.A * outer + self.B * pre.unsqueeze(1) + self.C * post.unsqueeze(0) + self.D

            # Only update W if there is a non-zero change (avoids inplace-modify during Phase 1)
            if delta_W.abs().sum() > 0:
                self.W.data = (W + self.lr_hebb * delta_W).clone()

        return output

    def reset_weights(self):
        """Reset plastic weights to initial values."""
        with torch.no_grad():
            self.W.copy_(self.W_init)

    def get_abcd_vector(self) -> torch.Tensor:
        """Flatten ABCD parameters into a single vector."""
        return torch.cat([self.A.flatten(), self.B.flatten(), self.C.flatten(), self.D.flatten()])

    def set_abcd_from_vector(self, vector: torch.Tensor):
        """Set ABCD parameters from a flat vector."""
        n = self.in_features * self.out_features
        with torch.no_grad():
            self.A.copy_(vector[:n].reshape(self.in_features, self.out_features))
            self.B.copy_(vector[n:2*n].reshape(self.in_features, self.out_features))
            self.C.copy_(vector[2*n:3*n].reshape(self.in_features, self.out_features))
            self.D.copy_(vector[3*n:4*n].reshape(self.in_features, self.out_features))

    @property
    def num_abcd_params(self) -> int:
        return 4 * self.in_features * self.out_features


class Hebbian(Model):
    """Hebbian learning model for BenchMARL.

    A single linear layer with Hebbian ABCD weight update rules.
    """

    def __init__(
        self,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        activation_class: Type[nn.Module] = None,
        activation_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
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

        self.hebbian_layer = HebbianLayer(
            in_features=self.input_features,
            out_features=self.output_features,
            lr_hebb=self.lr_hebb,
            weight_init=self.weight_init,
        )

        # Activation after Hebbian layer (optional)
        if activation_class is not None:
            self.activation = activation_class(**(activation_kwargs or {}))
        else:
            self.activation = None

        # Handle multi-agent dimension with share_params
        if self.input_has_agent_dim and not self.share_params:
            # Per-agent Hebbian layers (rarely used, but supported)
            self.hebbian_layers = nn.ModuleList([
                HebbianLayer(self.input_features, self.output_features, self.lr_hebb, self.weight_init)
                for _ in range(self.n_agents)
            ])

    def _perform_checks(self):
        super()._perform_checks()
        # Same shape checks as MLP

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        input = torch.cat(
            [
                torch.flatten(tensordict.get(in_key), start_dim=-self.num_feature_dims)
                for in_key in self.in_keys
            ],
            dim=-1,
        )

        if self.input_has_agent_dim:
            if not self.share_params:
                res = torch.stack(
                    [layer(input) for layer in self.hebbian_layers],
                    dim=-2,
                )
            else:
                res = self.hebbian_layer.forward(input)
                if not self.output_has_agent_dim:
                    res = res[..., 0, :]
        else:
            res = self.hebbian_layer.forward(input)

        if self.activation is not None:
            res = self.activation(res)

        tensordict.set(self.out_key, res)
        return tensordict


@dataclass
class HebbianConfig(ModelConfig):
    """Config for :class:`~benchmarl.models.Hebbian`."""

    lr_hebb: float = 0.01
    weight_init: float = 1.0
    activation_class: Type[nn.Module] = None
    activation_kwargs: Optional[dict] = None
    num_feature_dims: int = 1

    @staticmethod
    def associated_class():
        return Hebbian
