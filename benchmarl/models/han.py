from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Type

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig


class HanLayer(nn.Module):
    """Single Hebbian Attractor Network (HAN) layer.

    Three strict design rules, mirroring the spec:

    1. ``forward(x)`` performs **only** inference (matrix multiply) and pushes
       the current step's pre/post activations into a fixed-length sliding
       window. It NEVER mutates ``W``.

    2. ``update_weights()`` consumes the time-averaged pre/post from the
       window, computes the generalized ABCD Hebbian update, applies it to
       ``W``, then performs hard layer-wise max-abs normalization so the
       absolute maximum element of ``W`` is exactly 1.0 at the end of every
       update. The window is cleared after consumption so the next ``M``
       steps form a fresh window.

    3. The frequency of ``update_weights()`` is controlled externally: the
       owning :class:`HanModel` is responsible for calling it. The layer
       itself does not know about ``f_NN`` or ``f_hebb``.

    ABCD update (uses the time-averaged activations, NOT the current step):

        delta_W[i, j] = a[i, j] * x_bar_pre[j] * x_bar_post[i]
                      + b[i, j] * x_bar_pre[j]
                      + c[i, j] * x_bar_post[i]
                      + d[i, j]

    The default orientation matches the existing HNN: ``W`` has shape
    ``(in_features, out_features)`` and ``output = x @ W``; ``x_bar_pre``
    has shape ``(in_features,)`` and ``x_bar_post`` has shape
    ``(out_features,)``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        window_size: int = 10,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        self.window_size = int(window_size)

        # Plastic weights W: updated by HAN rule, not by gradients.
        W = torch.randn(in_features, out_features) / (in_features ** 0.5)
        self.register_buffer("W_init", W.clone())
        self.register_buffer("W", W.clone())

        # ABCD Hebbian parameters: optimized by CMA-ES.
        self.register_buffer("A", torch.zeros(in_features, out_features))
        self.register_buffer("B", torch.zeros(in_features, out_features))
        self.register_buffer("C", torch.zeros(in_features, out_features))
        self.register_buffer("D", torch.zeros(in_features, out_features))

        # Per-layer sliding window buffers for pre/post activations.
        # Stored on the Python side (not as buffers) because deque does not
        # need to be a state-dict entry; the contents are only ever read
        # once and then cleared by update_weights().
        self._pre_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self._post_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------ #
    # Inference path — no weight mutation.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W.data
        output = x @ W

        with torch.no_grad():
            # Reduce over all non-feature dims to get a single vector per
            # step. This matches the convention used by FullHebbianModel:
            # a single env step (with a possible agent dim) yields one
            # representative vector for the layer's pre and post.
            pre_vec = x.detach().mean(dim=tuple(range(x.ndim - 1))).to(W.device)
            post_vec = output.detach().mean(dim=tuple(range(output.ndim - 1))).to(W.device)

            # The spec requires recording activations "after tanh". We
            # apply tanh here for storage only — the inference path itself
            # is untouched (the activation between layers is applied by the
            # model, not the layer).
            self._pre_window.append(torch.tanh(pre_vec).clone())
            self._post_window.append(torch.tanh(post_vec).clone())

        return output

    # ------------------------------------------------------------------ #
    # Hebbian weight update — called externally, never from forward.
    # ------------------------------------------------------------------ #
    def update_weights(self) -> None:
        """Run one HAN weight update using the time-averaged activations.

        No-op if the window is empty (e.g. called before any forward pass).
        """
        if len(self._pre_window) == 0 or len(self._post_window) == 0:
            return

        with torch.no_grad():
            pre_stack = torch.stack(list(self._pre_window), dim=0)   # (T, in_features)
            post_stack = torch.stack(list(self._post_window), dim=0)  # (T, out_features)

            # Time-average over the M-step window. Strictly NEVER use the
            # current step's instantaneous values.
            x_bar_pre = pre_stack.mean(dim=0)   # (in_features,)
            x_bar_post = post_stack.mean(dim=0)  # (out_features,)

            outer_bar = x_bar_pre.unsqueeze(1) * x_bar_post.unsqueeze(0)  # (in, out)
            delta_W = (
                self.A * outer_bar
                + self.B * x_bar_pre.unsqueeze(1)
                + self.C * x_bar_post.unsqueeze(0)
                + self.D
            )

            new_W = self.W.data + self.lr_hebb * delta_W

            # Hard layer-wise max-abs normalization: per spec, the
            # absolute maximum element of every layer's weight matrix is
            # exactly 1.0 at the end of every update.
            max_abs = new_W.abs().max()
            if max_abs.item() > 0.0:
                new_W = new_W / max_abs

            self.W.data = new_W.clone()

            # Consume the window so the next M steps form a fresh window.
            self._pre_window.clear()
            self._post_window.clear()

    # ------------------------------------------------------------------ #
    # Standard helpers (compatible with the existing HNN API).
    # ------------------------------------------------------------------ #
    def reset_weights(self):
        with torch.no_grad():
            self.W.data = self.W_init.clone()

    def reset_window(self):
        """Clear the sliding-window buffers without touching W."""
        self._pre_window.clear()
        self._post_window.clear()

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
            self.B.copy_(vector[n:2 * n].reshape(self.in_features, self.out_features))
            self.C.copy_(vector[2 * n:3 * n].reshape(self.in_features, self.out_features))
            self.D.copy_(vector[3 * n:4 * n].reshape(self.in_features, self.out_features))

    @property
    def num_abcd_params(self) -> int:
        return 4 * self.in_features * self.out_features


class HanModel(Model):
    """Single-hidden-layer Hebbian Attractor Network.

    Architecture:
        input -> hidden(hidden_size) -> output
    with the three strict HAN rules in :class:`HanLayer` and an explicit
    ``ticks`` counter driving ``update_weights()`` at frequency ``f_hebb``
    while inference happens at frequency ``f_NN``.
    """

    def __init__(
        self,
        hidden_size: int = 18,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        window_size: int = 10,
        f_nn: int = 1,
        f_hebb: int = 1,
        activation_class: Type[nn.Module] = None,
        activation_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        self.window_size = int(window_size)
        self.f_nn = int(f_nn)
        self.f_hebb = int(f_hebb)
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
                torch.prod(torch.tensor(spec.shape[-self.num_feature_dims:])).item()
                for spec in self.input_spec.values(True, True)
            ]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        # Two HAN layers: input -> hidden(18) -> output. (Single hidden layer.)
        self.layers = nn.ModuleList([
            HanLayer(self.input_features, self.hidden_size, self.lr_hebb,
                     self.weight_init, self.window_size),
            HanLayer(self.hidden_size, self.output_features, self.lr_hebb,
                     self.weight_init, self.window_size),
        ])

        if activation_class is not None:
            self.activation = activation_class(**(activation_kwargs or {}))
        else:
            self.activation = nn.Tanh()

        # Multi-agent case: per-agent layer stacks. Each per-agent layer
        # has its own sliding window, which is correct: the activation
        # history is per-agent.
        if self.input_has_agent_dim and not self.share_params:
            self.per_agent_layers = nn.ModuleList([
                nn.ModuleList([
                    HanLayer(self.input_features, self.hidden_size, self.lr_hebb,
                             self.weight_init, self.window_size),
                    HanLayer(self.hidden_size, self.output_features, self.lr_hebb,
                             self.weight_init, self.window_size),
                ])
                for _ in range(self.n_agents)
            ])

        # Environment step counter. Incremented once per _forward call
        # (i.e. once per env step), NOT once per per-agent layer-stack
        # invocation.
        self.ticks: int = 0

        # Cache the trigger interval. Trigger every (f_nn // f_hebb)
        # forward passes; guard against invalid configurations.
        if self.f_hebb <= 0 or self.f_nn <= 0 or self.f_hebb > self.f_nn:
            self._update_interval = None  # disabled
        else:
            self._update_interval = self.f_nn // self.f_hebb

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

        # Advance the tick counter once per env step and conditionally
        # trigger the weight update. The trigger lives here, NOT inside
        # HanLayer.forward, so the frequency control is centralized at
        # the model level.
        self.ticks += 1
        self._maybe_update_weights()

        return tensordict

    def _forward_through_layers(self, x: torch.Tensor, layers) -> torch.Tensor:
        # Layer 1: input -> hidden (with activation)
        x = layers[0](x)
        if self.activation is not None:
            x = self.activation(x)
        # Layer 2: hidden -> output (no activation on inference)
        x = layers[1](x)
        return x

    def _maybe_update_weights(self) -> None:
        """Trigger ``update_weights()`` on every layer when the interval hits.

        W is strictly static between updates: forward() never mutates W.
        """
        if self._update_interval is None:
            return
        if self.ticks % self._update_interval != 0:
            return
        for layer in self.get_all_han_layers():
            layer.update_weights()

    # ------------------------------------------------------------------ #
    # Public API used by the CMA-ES optimizer.
    # ------------------------------------------------------------------ #
    def get_all_han_layers(self) -> List[HanLayer]:
        """Return all HanLayer instances in the network.

        For the per-agent case we return the per-agent stacks; the shared
        case returns ``self.layers``.
        """
        if self.input_has_agent_dim and not self.share_params:
            out: List[HanLayer] = []
            for stack in self.per_agent_layers:
                out.extend(list(stack))
            return out
        return list(self.layers)

    def get_all_hebbian_layers(self) -> List[HanLayer]:
        """Alias of :meth:`get_all_han_layers` for API symmetry with
        :class:`FullHebbianModel`."""
        return self.get_all_han_layers()

    def get_abcd_vector(self) -> torch.Tensor:
        """Flatten ABCD parameters from ALL layers into a single vector.

        For the per-agent case we deduplicate (the parameters of the
        per-agent layer stacks are independent nn.Module instances, so
        each has its own ABCD). For CMA-ES the optimizer only needs a
        single concatenated vector; if the per-agent case is active the
        vector is much longer, but that's the cost of independent plastic
        weights per agent.
        """
        return torch.cat([layer.get_abcd_vector() for layer in self.get_all_han_layers()])

    def set_abcd_from_vector(self, vector: torch.Tensor):
        offset = 0
        for layer in self.get_all_han_layers():
            n = layer.num_abcd_params
            layer.set_abcd_from_vector(vector[offset:offset + n])
            offset += n

    def reset_all_weights(self):
        """Reset plastic weights in all layers to initial values, and clear
        all sliding windows (so the next episode starts with a clean
        history)."""
        for layer in self.get_all_han_layers():
            layer.reset_weights()
            layer.reset_window()
        # Reset the tick counter so a new episode begins at ticks=0.
        self.ticks = 0

    @property
    def total_abcd_params(self) -> int:
        return sum(layer.num_abcd_params for layer in self.get_all_han_layers())


@dataclass
class HanConfig(ModelConfig):
    """Config for :class:`~benchmarl.models.HanModel`."""

    hidden_size: int = 18
    lr_hebb: float = 0.01
    weight_init: float = 1.0
    window_size: int = 10
    f_nn: int = 1
    f_hebb: int = 1
    activation_class: Type[nn.Module] = None
    activation_kwargs: Optional[dict] = None
    num_feature_dims: int = 1

    @staticmethod
    def associated_class():
        return HanModel
