#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Hebbian Graph Network (HGN) — multi-agent actor with plastic inter-agent edges.

Architecture
------------
Agents are nodes in a graph; the directed edges between them carry a *shared*
plastic Hebbian weight matrix ``W_edge`` that evolves online via the same
ABCD + sliding-window + hard-normalization rule used by :class:`HanLayer`. Per
env step:

    1. Each agent's observation is embedded into a D_h-dim hidden state.
    2. L rounds of message passing aggregate (sum) the per-edge messages
       ``m_{j→i} = tanh(W_edge · x_j)`` from every neighbour j into ``h_i``.
    3. A node-update :class:`HanLayer` maps ``[h_i ; agg_i] → h_i'``.
    4. An output :class:`HanLayer` maps ``h_i'`` → action logits.

The hidden state ``h_i`` is a pure black-box D_h-vector with no pre-defined
semantic split. Three plastic components share a single CMA-ES search budget:
``W_edge``, the node-update matrix, and the output matrix.

See ``benchmarl/models/han.py`` for the HanLayer contract this model reuses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

import torch
from tensordict import TensorDictBase
from torch import nn

from benchmarl.models.common import Model, ModelConfig
from benchmarl.models.han import HanLayer


TOPOLOGY_TYPES = {"full", "from_pos"}


class HgnModel(Model):
    """Hebbian Graph Network actor.

    Three plastic HanLayer instances, all governed by the same HAN rule
    (decoupled forward / update_weights, sliding-window time-averaging, hard
    ``max|W|=1`` normalization):

    * ``edge_layer`` (``D_h × D_h``): the shared inter-agent message matrix.
    * ``node_layer`` (``2·D_h × D_h``): per-step node update from concatenated
      own-state and aggregated incoming messages.
    * ``output_layer`` (``D_h × D_action``): per-agent action-logit head.

    Strict rules inherited from :class:`HanLayer`:
        * ``forward()`` NEVER mutates ``W``.
        * ``update_weights()`` consumes the sliding window and applies the
          hard max-abs normalization.
        * A ``ticks`` counter triggers ``update_weights()`` every
          ``f_nn // f_hebb`` env steps (or disabled if the configuration is
          invalid).
    """

    def __init__(
        self,
        d_h: int = 18,
        n_message_steps: int = 2,
        topology: str = "full",
        edge_radius: Optional[float] = None,
        lr_hebb: float = 0.01,
        weight_init: float = 1.0,
        window_size: int = 10,
        f_nn: int = 1,
        f_hebb: int = 1,
        activation_class: Type[nn.Module] = None,
        activation_kwargs: Optional[dict] = None,
        num_feature_dims: int = 1,
        **kwargs,
    ):
        self.d_h = int(d_h)
        self.n_message_steps = int(n_message_steps)
        self.topology = topology
        self.edge_radius = edge_radius
        self.lr_hebb = lr_hebb
        self.weight_init = weight_init
        self.window_size = int(window_size)
        self.f_nn = int(f_nn)
        self.f_hebb = int(f_hebb)
        self.num_feature_dims = int(num_feature_dims)

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

        if self.topology not in TOPOLOGY_TYPES:
            raise ValueError(
                f"Got topology={self.topology!r}, expected one of {TOPOLOGY_TYPES}"
            )
        if self.topology == "from_pos" and self.edge_radius is None:
            raise ValueError("topology='from_pos' requires edge_radius to be set")

        # Total per-agent feature dim after flattening observation leaves.
        self.input_features = sum(
            [
                torch.prod(torch.tensor(spec.shape[-self.num_feature_dims:])).item()
                for spec in self.input_spec.values(True, True)
            ]
        )
        self.output_features = self.output_leaf_spec.shape[-1]

        if not self.input_has_agent_dim:
            raise ValueError(
                "HgnModel requires input_has_agent_dim=True (per-agent features)"
            )

        # ----- static observation embedding (NOT Hebbian) -----
        self.embed = nn.Linear(self.input_features, self.d_h)
        if activation_class is not None:
            self.embed_act = activation_class(**(activation_kwargs or {}))
        else:
            self.embed_act = nn.Tanh()

        # ----- three plastic components (all HanLayer) -----
        # Single shared edge matrix: one set of W, one sliding window.
        self.edge_layer = HanLayer(
            self.d_h, self.d_h,
            lr_hebb=self.lr_hebb,
            weight_init=self.weight_init,
            window_size=self.window_size,
        )
        # Node update: 2·D_h → D_h
        self.node_layer = HanLayer(
            2 * self.d_h, self.d_h,
            lr_hebb=self.lr_hebb,
            weight_init=self.weight_init,
            window_size=self.window_size,
        )
        # Output head: D_h → D_action (single shared).
        self.output_layer = HanLayer(
            self.d_h, self.output_features,
            lr_hebb=self.lr_hebb,
            weight_init=self.weight_init,
            window_size=self.window_size,
        )

        # ----- ticks counter (mirrors HanModel) -----
        self.ticks: int = 0
        if self.f_hebb <= 0 or self.f_nn <= 0 or self.f_hebb > self.f_nn:
            self._update_interval = None
        else:
            self._update_interval = self.f_nn // self.f_hebb

        # ----- edge_index cache -----
        # For "full" topology the edge set is static; cache it once.
        # For "from_pos" the edge set depends on positions and is rebuilt
        # every step.
        self._cached_edge_index: Optional[torch.Tensor] = None
        if self.topology == "full":
            self._cached_edge_index = self._build_full_edge_index(self.n_agents)

    # ------------------------------------------------------------------ #
    # Topology helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_full_edge_index(n_agents: int) -> torch.Tensor:
        """All (i, j) pairs with i != j.

        Convention matches ``benchmarl/models/gnn.py::_get_edge_index(
        topology="full", self_loops=False)``: edge_index[0] = destinations,
        edge_index[1] = sources. We use this convention for ``index_add_``
        (aggregate incoming messages into destination nodes).
        """
        rows: List[int] = []
        cols: List[int] = []
        for i in range(n_agents):
            for j in range(n_agents):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        return torch.tensor([rows, cols], dtype=torch.long)

    def _build_from_pos_edge_index(
        self, pos_flat: torch.Tensor, batch_vec: torch.Tensor
    ) -> torch.Tensor:
        """Build edge_index dynamically from positions.

        Args:
            pos_flat: (P_total, 2) flattened positions across all parallel
                envs. P_total = num_envs * n_agents.
            batch_vec: (P_total,) long tensor indicating which env each node
                belongs to (same convention as
                ``benchmarl/models/gnn.py``).

        Returns:
            edge_index: (2, E) long tensor where edge_index[0] = dst,
                edge_index[1] = src, restricted to within-env pairs whose
                L2 distance is <= edge_radius.
        """
        # Try torch_geometric first (already used by gnn.py).
        try:
            import torch_geometric
        except ImportError as exc:  # pragma: no cover - torch_geometric is a
            # hard dep of this repo.
            raise ImportError(
                "topology='from_pos' requires torch_geometric; install it "
                "or use topology='full'."
            ) from exc

        # torch_geometric.nn.pool.radius_graph expects (x, batch, r, loop).
        edge_index = torch_geometric.nn.pool.radius_graph(
            pos_flat,
            batch=batch_vec,
            r=self.edge_radius,
            loop=False,
        )
        return edge_index

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # 1. Flatten per-agent observation leaves into (..., n_agents, D_obs).
        x = torch.cat(
            [
                torch.flatten(tensordict.get(in_key), start_dim=-self.num_feature_dims)
                for in_key in self.in_keys
            ],
            dim=-1,
        )
        # Tensordict shape: (B..., n_agents, D_obs) — already has agent dim.
        # We collapse all leading batch dims into P for the message-passing step.
        if x.ndim < 2 or x.shape[-2] != self.n_agents:
            raise RuntimeError(
                f"HgnModel expects input shape (..., n_agents={self.n_agents}, "
                f"D_obs), got {tuple(x.shape)}"
            )
        N = self.n_agents
        D_obs = self.input_features
        D_h = self.d_h
        D_act = self.output_features

        # Save the leading batch shape so we can restore it at the end.
        leading = x.shape[:-2]
        x_flat = x.reshape(-1, N, D_obs)            # (P, N, D_obs)
        P = x_flat.shape[0]

        # 2. Static observation embedding: (P, N, D_h)
        h = self.embed_act(self.embed(x_flat))

        # 3. L rounds of message passing.
        for _ in range(self.n_message_steps):
            if self.topology == "full":
                # edge_index is (2, E), same for every parallel env.
                ei = self._cached_edge_index
                if ei is None:
                    ei = self._build_full_edge_index(N).to(h.device)
                src = h[:, ei[1]]                    # (P, E, D_h)
                msg = self.edge_layer(src)            # HanLayer.forward — records pre/post
                agg = torch.zeros((P, N, D_h), device=h.device, dtype=h.dtype)
                agg.index_add_(1, ei[0].to(h.device), msg)
            else:  # from_pos
                # Recover per-node positions. The first observation leaf is
                # expected to start with ``pos(2)`` per our scenario convention.
                # Fallback: use the first 2 dims of the flattened obs.
                pos_flat = x_flat[..., :2].reshape(P * N, 2)
                batch_vec = (
                    torch.arange(P, device=h.device).repeat_interleave(N)
                )
                ei = self._build_from_pos_edge_index(pos_flat, batch_vec)
                # Map flattened node ids back to per-env ids.
                env_of_node = batch_vec
                dst_env = env_of_node[ei[0]]
                # We need per-env index_add_. For simplicity we batch over P
                # by gathering per-env sources. A loop over P is acceptable
                # here because CMA-ES evaluation is on single-env rollouts.
                # For multi-env we fall back to building one (E,) per env.
                if P == 1:
                    src = h[0, ei[1]]                # (E, D_h)
                    msg = self.edge_layer(src.unsqueeze(0))  # (1, E, D_h)
                    agg = torch.zeros((1, N, D_h), device=h.device, dtype=h.dtype)
                    agg.index_add_(1, ei[0].to(h.device), msg[0].unsqueeze(0))
                else:
                    # General (rarely used in CMA-ES evaluation) path.
                    agg = torch.zeros((P, N, D_h), device=h.device, dtype=h.dtype)
                    for p in range(P):
                        mask = (ei[0] // N) == p
                        ei_p = ei[:, mask] % N
                        src_p = h[p, ei_p[1]]
                        msg_p = self.edge_layer(src_p.unsqueeze(0))[0]
                        agg[p].index_add_(0, ei_p[0], msg_p)

            cat = torch.cat([h, agg], dim=-1)        # (P, N, 2·D_h)
            h = torch.tanh(self.node_layer(cat))      # (P, N, D_h)

        # 4. Output head.
        out = self.output_layer(h)                   # (P, N, D_action)
        out = out.reshape(*leading, N, D_act)

        tensordict.set(self.out_key, out)

        # 5. Trigger weight update if interval reached.
        self.ticks += 1
        self._maybe_update_weights()
        return tensordict

    # ------------------------------------------------------------------ #
    # Weight update — same semantics as HanModel._maybe_update_weights
    # ------------------------------------------------------------------ #
    def _maybe_update_weights(self) -> None:
        if self._update_interval is None:
            return
        if self.ticks % self._update_interval != 0:
            return
        for layer in self.get_all_han_layers():
            layer.update_weights()

    # ------------------------------------------------------------------ #
    # Public API used by the CMA-ES optimizer (mirrors HanModel)
    # ------------------------------------------------------------------ #
    def get_all_han_layers(self) -> List[HanLayer]:
        return [self.edge_layer, self.node_layer, self.output_layer]

    def get_abcd_vector(self) -> torch.Tensor:
        return torch.cat(
            [layer.get_abcd_vector() for layer in self.get_all_han_layers()]
        )

    def set_abcd_from_vector(self, vector: torch.Tensor):
        offset = 0
        for layer in self.get_all_han_layers():
            n = layer.num_abcd_params
            layer.set_abcd_from_vector(vector[offset:offset + n])
            offset += n

    def reset_all_weights(self):
        """Reset plastic weights in all three layers to their initial values
        and clear all sliding windows. The ``ticks`` counter is also reset
        so a new episode starts at ticks=0."""
        for layer in self.get_all_han_layers():
            layer.reset_weights()
            layer.reset_window()
        self.ticks = 0

    @property
    def total_abcd_params(self) -> int:
        return sum(layer.num_abcd_params for layer in self.get_all_han_layers())


@dataclass
class HgnConfig(ModelConfig):
    """Config for :class:`~benchmarl.models.hgn.HgnModel`."""

    d_h: int = 18
    n_message_steps: int = 2
    topology: str = "full"
    edge_radius: Optional[float] = None
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
        return HgnModel
