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
       ``W``, then normalizes ``W``. The window is cleared after consumption
       so the next ``M`` steps form a fresh window.

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

    ------------------------------------------------------------------------
    Rework notes (see ``HAN_diagnosis_and_rework_plan.md``)
    ------------------------------------------------------------------------
    Four mechanisms were changed because measurements showed the original
    implementation was **not** data-driven. On a trained navigation model
    the constant ``D`` term contributed **80.6%** of |dW| while the
    Hebbian ``A`` term contributed **1.3%**, and the converged ``W`` was
    numerically equal to ``normalize(-D)`` (cosine 0.98) — i.e. a frozen
    matrix independent of the environment.

    * ``stats_mode`` — the outer product is now formed **per sample** and
      then averaged ("mean of products") instead of averaging the
      activations first and then taking the outer product ("mean of
      means"). The latter violates ``E[xy^T] != E[x]E[y]^T``: averaging
      over a batch of parallel environments with *different* obstacle
      layouts cancels the input-output correlation before it is ever
      measured. Measured effect on a real navigation rollout: the Hebbian
      term grew **8.6x** when switched to per-sample.

    * ``normalize_stats`` — the recorded activations are RMS-normalised
      before the outer product. Raw ``tanh`` activations have RMS ~0.13,
      so the outer product had magnitude ~0.02 and was swamped by a
      typical ``|D| ~ 0.4``. RMS-normalising makes the Hebbian term
      scale-free, so it is comparable to ``D`` regardless of the input
      magnitude (and it also stops the max-abs rescale from acting as a
      crude input-gain control).

    * ``lr_oja`` — adds the classic Oja stabilisation term
      ``-eta * W * E[y^2]``. Without it there is no term that keeps the
      Hebbian growth bounded along the data subspace, which is why the
      original code needed a crude max-abs rescale. With Oja, ``W`` grows
      along the *currently informative* input directions and decays along
      the orthogonal ones.

    * ``normalization`` — "spectral" divides ``W`` by its largest singular
      value, bounding the layer's overall gain while preserving the
      relative gains between output units. The legacy "max_abs" mode let
      the single largest element of the layer scale every other element;
      per-column normalisation would instead force all output units to
      have *equal* gain, discarding the relative gains the network is
      trying to learn.

    Together these make the fixed point of ``W`` a function of the
    environment's data rather than of the constant ``D``: with
    ``d_scale=0`` the converged ``W`` had cosine similarity 0.05 between
    two different random initialisations (vs 0.88 under the old rule),
    i.e. the memory of the initialisation is essentially gone.
    """

    #: Supported values for ``stats_mode``.
    STATS_MODES = ("per_sample", "population")
    #: Supported values for ``normalization``.
    NORMALIZATIONS = ("spectral", "column_maxabs", "column_l2", "max_abs", "none")

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        window_size: int = 10,
        lr_oja: float = 0.0,
        d_scale: float = 0.0,
        stats_mode: str = "per_sample",
        normalization: str = "spectral",
        normalize_stats: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        self.window_size = int(window_size)
        self.lr_oja = float(lr_oja)
        self.d_scale = float(d_scale)
        self.normalize_stats = bool(normalize_stats)
        if stats_mode not in self.STATS_MODES:
            raise ValueError(
                f"stats_mode must be one of {self.STATS_MODES}, got {stats_mode!r}"
            )
        if normalization not in self.NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {self.NORMALIZATIONS}, "
                f"got {normalization!r}"
            )
        self.stats_mode = stats_mode
        self.normalization = normalization

        # Plastic weights W: updated by HAN rule, not by gradients.
        #
        # ``weight_init`` is the target max-abs value of the random
        # initialization. Previously this parameter was accepted and stored
        # but never used, so ``W`` always started from
        # ``randn/sqrt(in)`` regardless of the requested value.
        W = torch.randn(in_features, out_features) / (in_features ** 0.5)
        if weight_init is not None and weight_init > 0:
            max_abs = W.abs().max()
            if max_abs.item() > 0.0:
                W = W * (float(weight_init) / max_abs.item())
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
        #
        # In "per_sample" mode each entry holds a (n_samples, features)
        # matrix; in "population" mode each entry holds a single
        # (1, features) row holding the population mean (legacy behaviour).
        self._pre_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        self._post_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
        # Raw (un-normalised) E[y^2] per output unit, used by the Oja
        # term. Kept separate from ``_post_window`` because the Oja term
        # must see the true second moment of the activations.
        self._post_sq_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------ #
    # Inference path — no weight mutation.
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W.data
        output = x @ W

        with torch.no_grad():
            # The spec requires recording activations "after tanh". We
            # apply tanh here for storage only — the inference path itself
            # is untouched (the activation between layers is applied by the
            # model, not the layer).
            pre_act = torch.tanh(x.detach()).to(W.device)
            post_act = torch.tanh(output.detach()).to(W.device)

            if self.stats_mode == "per_sample":
                # Keep every sample as its own row so the outer product is
                # formed per sample and only then averaged. Averaging the
                # activations first (legacy behaviour) cancels the
                # input-output correlation across the batch.
                pre_rows = pre_act.reshape(-1, pre_act.shape[-1])
                post_rows = post_act.reshape(-1, post_act.shape[-1])
            else:
                # Legacy "population" mode: collapse every non-feature dim
                # into one representative vector per step.
                pre_rows = pre_act.mean(
                    dim=tuple(range(pre_act.ndim - 1)), keepdim=True
                ).reshape(1, -1)
                post_rows = post_act.mean(
                    dim=tuple(range(post_act.ndim - 1)), keepdim=True
                ).reshape(1, -1)

            # The Oja term needs the RAW second moment E[y^2]: it must
            # measure how strongly each output unit is actually firing.
            # After RMS-normalisation E[y^2] is 1 by construction and the
            # term degenerates into plain weight decay.
            self._post_sq_window.append(
                post_rows.pow(2).mean(dim=-1, keepdim=True).clone()
            )

            if self.normalize_stats:
                # RMS-normalise each recorded row so the outer product is
                # scale-free. Raw tanh activations have RMS ~0.13, which
                # made the Hebbian term ~20x smaller than a typical |D|
                # element and let the constant D term dominate.
                pre_rows = self._rms_normalize(pre_rows)
                post_rows = self._rms_normalize(post_rows)

            self._pre_window.append(pre_rows.clone())
            self._post_window.append(post_rows.clone())

        return output

    @staticmethod
    def _rms_normalize(rows: torch.Tensor) -> torch.Tensor:
        """Scale each row of ``rows`` to unit RMS.

        Rows whose RMS is (numerically) zero are left untouched, so a
        silent input dimension does not blow up to unit scale.
        """
        rms = rows.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return rows / rms.clamp_min(1e-8)

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
            pre_stack = torch.stack(list(self._pre_window), dim=0)   # (T, N, in)
            post_stack = torch.stack(list(self._post_window), dim=0)  # (T, N, out)

            # Time-average over the M-step window AND over the sample
            # dimension. Strictly NEVER use the current step's
            # instantaneous values.
            x_bar_pre = pre_stack.mean(dim=(0, 1))   # (in_features,)
            x_bar_post = post_stack.mean(dim=(0, 1))  # (out_features,)

            if self.stats_mode == "per_sample":
                # "Mean of products": form the outer product per sample,
                # then average over samples and time.
                #
                #   E[x y^T] = mean_s( x_s y_s^T )
                #
                # The legacy path computed E[x] E[y]^T instead, which is
                # not the same quantity and systematically cancels the
                # Hebbian correlation when the batch contains different
                # environments.
                outer_bar = (
                    pre_stack.unsqueeze(-1) * post_stack.unsqueeze(-2)
                ).mean(dim=(0, 1))                        # (in, out)
            else:
                outer_bar = (
                    x_bar_pre.unsqueeze(1) * x_bar_post.unsqueeze(0)
                )                                          # (in, out)

            delta_W = (
                self.A * outer_bar
                + self.B * x_bar_pre.unsqueeze(1)
                + self.C * x_bar_post.unsqueeze(0)
                + self.d_scale * self.D
            )

            new_W = self.W.data + self.lr_hebb * delta_W

            if self.lr_oja > 0.0:
                # Oja stabilisation: penalise the component of W along the
                # currently active output direction. This is the classic
                # boundedness term of the Hebbian/Oja family and is what
                # lets the fixed point of W be set by the *data* rather
                # than by the constant D term.
                #
                # ``E[y^2]`` is taken from the RAW activations: after the
                # RMS normalisation the second moment is 1 by construction
                # and the term would degenerate into plain weight decay.
                if len(self._post_sq_window) > 0:
                    post_sq = torch.stack(
                        list(self._post_sq_window), dim=0
                    ).mean(dim=(0, 1))                       # (out,)
                else:
                    post_sq = torch.ones(
                        self.out_features, device=new_W.device
                    )
                oja = new_W * post_sq.unsqueeze(0)
                new_W = new_W - self.lr_hebb * self.lr_oja * oja

            new_W = self._normalize(new_W)

            self.W.data = new_W.clone()

            # Consume the window so the next M steps form a fresh window.
            self._pre_window.clear()
            self._post_window.clear()
            self._post_sq_window.clear()

    def _normalize(self, W: torch.Tensor) -> torch.Tensor:
        """Normalize ``W`` according to ``self.normalization``.

        * ``spectral`` (default): divide by the largest singular value, so
          the layer's overall gain is bounded by 1 **while the relative
          gains between output units are preserved**. This is the mode
          that fixes the legacy ``max_abs`` pathology: under ``max_abs``
          a single large element set the scale of every other element in
          the layer, and per-column normalisation (below) would instead
          force all output units to have *equal* gain, discarding the
          relative gains the network is trying to learn.
        * ``column_maxabs``: each output neuron's incoming weight vector
          is rescaled so its own largest element is 1.0.
        * ``column_l2``: each output neuron's incoming weight vector has
          unit L2 norm.
        * ``max_abs``: the legacy behaviour — the absolute maximum element
          of the layer is exactly 1.0.
        * ``none``: no normalisation.
        """
        if self.normalization == "none":
            return W
        if self.normalization == "max_abs":
            max_abs = W.abs().max()
            if max_abs.item() > 0.0:
                return W / max_abs
            return W
        if self.normalization == "column_maxabs":
            col_max = W.abs().amax(dim=0, keepdim=True).clamp_min(1e-8)  # (1, out)
            return W / col_max
        if self.normalization == "column_l2":
            norms = W.norm(dim=0, keepdim=True).clamp_min(1e-8)   # (1, out)
            return W / norms
        # spectral
        sv = torch.linalg.matrix_norm(W, ord=2)
        return W / sv.clamp_min(1e-8)

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
        self._post_sq_window.clear()

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
        lr_oja: float = 0.0,
        d_scale: float = 0.0,
        stats_mode: str = "per_sample",
        normalization: str = "spectral",
        normalize_stats: bool = True,
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
        self.lr_oja = float(lr_oja)
        self.d_scale = float(d_scale)
        self.stats_mode = stats_mode
        self.normalization = normalization
        self.normalize_stats = bool(normalize_stats)
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

        layer_kwargs = dict(
            lr_hebb=self.lr_hebb,
            weight_init=self.weight_init,
            window_size=self.window_size,
            lr_oja=self.lr_oja,
            d_scale=self.d_scale,
            stats_mode=self.stats_mode,
            normalization=self.normalization,
            normalize_stats=self.normalize_stats,
        )

        # Two HAN layers: input -> hidden(18) -> output. (Single hidden layer.)
        self.layers = nn.ModuleList([
            HanLayer(self.input_features, self.hidden_size, **layer_kwargs),
            HanLayer(self.hidden_size, self.output_features, **layer_kwargs),
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
                    HanLayer(self.input_features, self.hidden_size, **layer_kwargs),
                    HanLayer(self.hidden_size, self.output_features, **layer_kwargs),
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

        # When True, the Hebbian update is skipped entirely — W stays
        # exactly as set. Used by the static-network control: with
        # ``lr_hebb == 0`` the update ``W + 0*delta_W`` is inert, but the
        # normalization would still rescale W every interval, which would
        # corrupt a CMA-ES candidate that CMA-ES wrote directly into W.
        self.freeze_weights: bool = False

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
            # Each agent has its own layer stack, so agent i's stack must
            # see ONLY agent i's observation slice ``x[..., i, :]``
            # (shape ``(..., features)``), not the full agent tensor.
            #
            # The previous code passed the whole ``x`` to every stack and
            # then stacked the results, which produced a tensor with a
            # spurious extra agent axis:
            #
            #   x (B, N, F) -> per stack (B, N, out) -> stack -> (B, N, N, out)
            #
            # The expected output shape is ``(B, N, out)``.
            res = torch.stack(
                [
                    self._forward_through_layers(x[..., i, :], layers)
                    for i, layers in enumerate(self.per_agent_layers)
                ],
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
        if self.freeze_weights:
            return
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

    # ------------------------------------------------------------------ #
    # Direct access to the plastic weight matrices W.
    #
    # Used by the CMA-ES "static mode" control: when ``lr_hebb == 0`` the
    # Hebbian rule is inert (ABCD never influences anything), so optimizing
    # ABCD would be optimizing a no-op. The static control instead lets
    # CMA-ES optimize W directly, producing a *trained* static feedforward
    # network as the correct baseline for "what does plasticity buy?".
    # ------------------------------------------------------------------ #
    def get_w_vector(self) -> torch.Tensor:
        """Flatten the plastic weight matrices ``W`` of all layers.

        Mirrors :meth:`get_abcd_vector` but for W (140 dims for the
        10->10->4 flocking HAN, vs 560 for ABCD).
        """
        return torch.cat([
            layer.W.data.flatten() for layer in self.get_all_han_layers()
        ])

    def set_w_from_vector(self, vector: torch.Tensor):
        """Write a flat vector into each layer's ``W`` **and** ``W_init``.

        ``W_init`` is synced deliberately: :meth:`reset_all_weights`
        restores ``W`` from ``W_init``, and ``W_init`` holds the random
        initialization. Re-syncing it makes that reset a no-op for the
        candidate instead of randomizing the network between episodes.
        """
        offset = 0
        with torch.no_grad():
            for layer in self.get_all_han_layers():
                n = layer.W.numel()
                chunk = vector[offset:offset + n].reshape(layer.W.shape)
                layer.W.data.copy_(chunk)
                layer.W_init.copy_(chunk)
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
    # Oja stabilisation strength (0 disables it). See HanLayer docstring.
    lr_oja: float = 0.0
    # Scale applied to the constant D term. 0 disables the drift term so
    # the update is driven purely by the data-driven A/B/C terms.
    d_scale: float = 0.0
    # "per_sample" (data-driven) or "population" (legacy).
    stats_mode: str = "per_sample"
    # RMS-normalise the recorded activations before the outer product.
    normalize_stats: bool = True
    # "spectral" (default), "column_maxabs", "column_l2",
    # "max_abs" (legacy), "none".
    normalization: str = "spectral"
    activation_class: Type[nn.Module] = None
    activation_kwargs: Optional[dict] = None
    num_feature_dims: int = 1

    @staticmethod
    def associated_class():
        return HanModel
