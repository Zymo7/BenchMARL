"""Smoke tests for the Hebbian Graph Network (HGN) model.

Run with:  conda run -n benchmarl python tests/test_hgn.py
"""
from __future__ import annotations

import sys

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from benchmarl.models.hgn import HgnConfig, HgnModel


def _build_specs(n_agents: int, d_obs: int, d_action: int):
    input_spec = Composite(
        {"agents": Composite({"observation": Unbounded(shape=(n_agents, d_obs))},
                             shape=(n_agents,))}
    )
    output_spec = Composite(
        {"agents": Composite({"logits": Unbounded(shape=(n_agents, d_action))},
                             shape=(n_agents,))}
    )
    action_spec = Composite(
        {"agents": Composite({"action": Unbounded(shape=(n_agents, d_action))},
                             shape=(n_agents,))}
    )
    return input_spec, output_spec, action_spec


def _make_td(n_envs: int, n_agents: int, d_obs: int, d_action: int):
    return TensorDict(
        {
            "agents": TensorDict(
                {"observation": torch.randn(n_envs, n_agents, d_obs)},
                batch_size=[n_envs, n_agents],
            ),
        },
        batch_size=[n_envs],
    )


def test_forward_shape():
    """Smoke: HgnModel._forward produces (n_envs, n_agents, d_action) output."""
    print("\n[test_forward_shape]")
    n_agents, d_obs, d_action, d_h = 4, 10, 2, 8
    cfg = HgnConfig(d_h=d_h, n_message_steps=2)
    input_spec, output_spec, action_spec = _build_specs(n_agents, d_obs, d_action)

    model: HgnModel = cfg.get_model(
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group="agents",
        input_has_agent_dim=True,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=action_spec,
    )

    td = _make_td(n_envs=2, n_agents=n_agents, d_obs=d_obs, d_action=d_action)
    out = model(td)
    logits = out["agents", "logits"]
    print(f"  logits.shape={tuple(logits.shape)}")
    assert logits.shape == (2, n_agents, d_action), logits.shape
    assert torch.isfinite(logits).all(), "logits contain non-finite values"
    print("  PASS")


def test_reset_all_weights():
    """Smoke: reset_all_weights() resets W, clears deques, zeroes ticks."""
    print("\n[test_reset_all_weights]")
    n_agents, d_obs, d_action, d_h = 4, 10, 2, 8
    cfg = HgnConfig(d_h=d_h, n_message_steps=2, window_size=4, f_nn=2, f_hebb=1)
    input_spec, output_spec, action_spec = _build_specs(n_agents, d_obs, d_action)

    model: HgnModel = cfg.get_model(
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group="agents",
        input_has_agent_dim=True,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=action_spec,
    )

    # Snapshot initial W.
    edge_w0 = model.edge_layer.W.clone()
    node_w0 = model.node_layer.W.clone()
    out_w0 = model.output_layer.W.clone()

    td = _make_td(n_envs=1, n_agents=n_agents, d_obs=d_obs, d_action=d_action)
    for _ in range(10):
        model(td)

    # Sanity: after a few ticks something should have moved (most likely).
    # We don't assert movement — random init may keep W near zero.
    print(f"  ticks after 10 steps: {model.ticks}")
    print(f"  deques len after 10 steps: pre={len(model.edge_layer._pre_window)}, "
          f"post={len(model.edge_layer._post_window)}")

    # Reset and check.
    model.reset_all_weights()
    assert model.ticks == 0
    assert torch.equal(model.edge_layer.W, edge_w0), "edge_layer.W not reset"
    assert torch.equal(model.node_layer.W, node_w0), "node_layer.W not reset"
    assert torch.equal(model.output_layer.W, out_w0), "output_layer.W not reset"
    assert len(model.edge_layer._pre_window) == 0
    assert len(model.edge_layer._post_window) == 0
    print("  reset_all_weights restores W, clears deques, zeroes ticks: PASS")


def test_abcd_vector_roundtrip():
    """Smoke: get_abcd_vector / set_abcd_from_vector round-trip, plus update
    fires when window fills."""
    print("\n[test_abcd_vector_roundtrip]")
    n_agents, d_obs, d_action, d_h = 4, 10, 2, 8
    cfg = HgnConfig(d_h=d_h, n_message_steps=2, window_size=5, f_nn=1, f_hebb=1)
    input_spec, output_spec, action_spec = _build_specs(n_agents, d_obs, d_action)

    model: HgnModel = cfg.get_model(
        input_spec=input_spec,
        output_spec=output_spec,
        agent_group="agents",
        input_has_agent_dim=True,
        n_agents=n_agents,
        centralised=False,
        share_params=True,
        device="cpu",
        action_spec=action_spec,
    )

    n_total = model.total_abcd_params
    expected = 4 * (d_h * d_h + 2 * d_h * d_h + d_h * d_action)
    print(f"  total_abcd_params = {n_total}, expected = {expected}")
    assert n_total == expected, f"abcd count mismatch: {n_total} vs {expected}"

    # Round-trip a random vector.
    v = torch.randn(n_total)
    model.set_abcd_from_vector(v.clone())
    v2 = model.get_abcd_vector()
    assert torch.allclose(v, v2), "ABCD vector round-trip failed"
    print("  ABCD round-trip: PASS")

    # Drive the model past the window to ensure update_weights fires.
    td = _make_td(n_envs=1, n_agents=n_agents, d_obs=d_obs, d_action=d_action)
    pre_max = model.edge_layer.W.abs().max().item()
    for _ in range(15):
        model(td)
    post_max = model.edge_layer.W.abs().max().item()
    print(f"  edge_layer W max|W| before/after 15 steps: {pre_max:.4f} / {post_max:.4f}")
    # After update_weights(), the spec mandates max|W| == 1.0
    # for non-zero matrices.
    assert abs(post_max - 1.0) < 1e-5, (
        f"HAN rule broken: max|W|={post_max}, expected 1.0"
    )
    print("  HAN hard-normalization rule fires correctly: PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("HGN smoke tests")
    print("=" * 60)
    try:
        test_forward_shape()
        test_reset_all_weights()
        test_abcd_vector_roundtrip()
        print("\nAll HGN smoke tests passed.")
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
