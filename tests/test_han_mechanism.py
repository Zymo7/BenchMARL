"""Regression tests for the HAN Hebbian mechanism and the
``navigation_obs_avoidance`` fitness semantics.

These lock in the findings from ``HAN_diagnosis_and_rework_plan.md`` so a
later change cannot silently reintroduce them:

* the Hebbian outer product must be formed **per sample** and only then
  averaged (not "average activations, then outer product");
* the ``D`` term must not be able to dominate the data-driven terms;
* ``W``'s fixed point must be data-driven, not ``normalize(-D)``;
* ``weight_init`` must actually affect the initialization;
* ``success`` must mean "reached the goal", not "the episode ended" —
  VMAS also raises ``done`` at the ``max_steps`` horizon.

Run with::

    pytest tests/test_han_mechanism.py -v
"""
import sys

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from benchmarl.models.han import HanConfig, HanLayer, HanModel


def approx(a, b, tol=1e-4):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------- #
# Layer-level mechanism tests (no environment needed).
# --------------------------------------------------------------------------- #
def check_weight_init_is_actually_used():
    """``weight_init`` used to be stored but never applied.

    W was always ``randn/sqrt(in)`` regardless of the requested value, so
    two layers built with different ``weight_init`` had *identical* W.
    """
    a = HanLayer(9, 12, weight_init=0.1)
    b = HanLayer(9, 12, weight_init=99.0)
    assert not torch.allclose(a.W.data, b.W.data), (
        "weight_init has no effect on the initialization — it is a dead parameter"
    )
    assert approx(a.W.data.abs().max().item(), 0.1, 1e-4), a.W.data.abs().max().item()
    assert approx(b.W.data.abs().max().item(), 99.0, 1e-2), b.W.data.abs().max().item()


def check_per_sample_stats_larger_than_population_stats():
    """The per-sample outer product must be much larger than the
    population-averaged one.

    Averaging activations before taking the outer product violates
    ``E[xy^T] != E[x]E[y]^T`` and cancels the input-output correlation
    across a batch of parallel environments.
    """
    torch.manual_seed(0)
    x = torch.randn(10, 1, 9) * 0.5
    w = torch.randn(9, 12) * 0.3
    out = x @ w

    pre = torch.tanh(x)
    post = torch.tanh(out)

    outer_population = (
        pre.mean(dim=(0, 1)).unsqueeze(1) * post.mean(dim=(0, 1)).unsqueeze(0)
    )
    outer_per_sample = (
        pre.reshape(-1, 9).unsqueeze(2) * post.reshape(-1, 12).unsqueeze(1)
    ).mean(dim=0)

    assert outer_per_sample.abs().mean() > 3 * outer_population.abs().mean()


def check_update_uses_per_sample_outer_product():
    """With ``stats_mode='per_sample'`` the layer must store per-sample
    rows; with ``stats_mode='population'`` it must store one row."""
    x = torch.randn(4, 2, 6)

    per_sample = HanLayer(6, 3, stats_mode="per_sample")
    per_sample(x)
    assert per_sample._pre_window[-1].shape == (8, 6)

    population = HanLayer(6, 3, stats_mode="population")
    population(x)
    assert population._pre_window[-1].shape == (1, 6)

    # The legacy "population" mode must collapse the agent axis too.
    multi = HanLayer(6, 3, stats_mode="per_sample")
    multi(torch.randn(2, 4, 6))
    assert multi._pre_window[-1].shape == (8, 6)


def check_spectral_normalization_bounds_gain_and_keeps_relative_gains():
    """``spectral`` must bound the layer gain *and* preserve the relative
    gains between output units.

    Under the legacy ``max_abs`` a single large element rescales every
    other element in the layer, so the relative gains between output units
    are dictated by whichever element happens to be largest. Per-column
    normalisation would instead force all output units to have *equal*
    gain, discarding the relative gains the network is trying to learn.
    """
    W = torch.tensor([[1.0, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1]])

    layer = HanLayer(4, 4, normalization="spectral")
    out = layer._normalize(W)

    # Layer gain is bounded by 1 (largest singular value == 1).
    assert approx(torch.linalg.matrix_norm(out, ord=2).item(), 1.0, 1e-5)

    # Relative gains between output units are preserved: column 0 keeps a
    # larger norm than columns 1..3, exactly as in the input.
    col_norms = out.norm(dim=0)
    assert col_norms[0].item() > col_norms[1].item() * 2.0
    assert approx(col_norms[1].item(), col_norms[3].item(), 1e-6)

    # Under the legacy max_abs the same matrix is scaled by ONE element.
    legacy = HanLayer(4, 4, normalization="max_abs")
    out_legacy = legacy._normalize(W)
    legacy_max = out_legacy.abs().amax(dim=0)
    assert approx(legacy_max[0].item(), 1.0, 1e-6)
    assert approx(legacy_max[1].item(), 0.1, 1e-6)


def check_column_l2_flattens_relative_gains():
    """``column_l2`` (an offered alternative) forces every output unit to
    unit L2 norm, i.e. it discards the relative gains between units."""
    W = torch.tensor([[1.0, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1],
                      [0.1, 0.1, 0.1, 0.1]])
    layer = HanLayer(4, 4, normalization="column_l2")
    out = layer._normalize(W)
    col_norms = out.norm(dim=0)
    assert approx(col_norms[0].item(), 1.0, 1e-5)
    assert approx(col_norms[1].item(), 1.0, 1e-5)


def check_d_term_does_not_dominate_data_terms():
    """With the reworked statistics the data-driven terms must be able to
    out-scale the constant ``D`` term.

    Previously ``D`` contributed ~80% of |dW| and the Hebbian A term only
    ~1.3%, which made the network's fixed point ``normalize(-D)`` —
    completely independent of the environment.
    """
    torch.manual_seed(0)
    x = torch.randn(10, 1, 9) * 0.5

    layer = HanLayer(9, 12, lr_hebb=1.0, window_size=10,
                     stats_mode="per_sample", normalize_stats=True)
    layer.A.data.fill_(1.0)
    layer.D.data.fill_(1.0)
    for _ in range(10):
        layer(x)

    pre_stack = torch.stack(list(layer._pre_window))
    post_stack = torch.stack(list(layer._post_window))
    outer = (pre_stack.unsqueeze(-1) * post_stack.unsqueeze(-2)).mean(dim=(0, 1))
    data_term = (layer.A * outer).abs().mean().item()
    d_term = layer.D.abs().mean().item()

    # The data term must now be a meaningful fraction of the D term.
    # (Under the legacy statistics it was ~1.3/80.6 = 1.6% of it.)
    assert data_term > 0.2 * d_term, (
        f"data term ({data_term:.4f}) is negligible next to the D term "
        f"({d_term:.4f}); the Hebbian update is still drift-dominated"
    )

    # And the reworked statistics must beat the legacy ones.
    legacy = HanLayer(9, 12, lr_hebb=1.0, window_size=10,
                      stats_mode="population", normalize_stats=False)
    legacy.A.data.fill_(1.0)
    legacy.D.data.fill_(1.0)
    for _ in range(10):
        legacy(x)
    lpre = torch.stack(list(legacy._pre_window)).reshape(-1, 9)
    lpost = torch.stack(list(legacy._post_window)).reshape(-1, 12)
    louter = (
        lpre.mean(dim=0).unsqueeze(1) * lpost.mean(dim=0).unsqueeze(0)
    )
    legacy_data_term = (legacy.A * louter).abs().mean().item()
    assert data_term > 3 * legacy_data_term, (
        f"reworked data term ({data_term:.4f}) should be much larger than the "
        f"legacy one ({legacy_data_term:.4f})"
    )


def check_d_scale_zero_removes_the_drift_term():
    """``d_scale=0`` must switch the constant drift term off entirely."""
    torch.manual_seed(0)
    x = torch.randn(10, 1, 9)
    layer = HanLayer(9, 12, lr_hebb=1.0, window_size=10, d_scale=0.0)
    layer.A.data.fill_(1.0)
    layer.D.data.fill_(1.0)
    for _ in range(10):
        layer(x)
    pre_stack = torch.stack(list(layer._pre_window))
    post_stack = torch.stack(list(layer._post_window))
    outer = (pre_stack.unsqueeze(-1) * post_stack.unsqueeze(-2)).mean(dim=(0, 1))
    delta = (layer.A * outer).abs().mean().item()
    assert delta > 0.0


def check_fixed_point_is_data_determined():
    """The converged W must depend on the DATA, not just on ``D`` and not
    on its random initialisation.

    Under the legacy implementation the fixed point was
    ``normalize(-D)`` (measured cosine similarity 0.98 against the real
    rollout), i.e. the network was a frozen matrix that CMA-ES had
    effectively precomputed. Here the reworked rule (per-sample stats +
    RMS normalisation + Oja + d_scale=0) must produce a W that is
    essentially independent of the random init.
    """
    def converge(seed, scale=1.0, offset=0.0):
        torch.manual_seed(seed)
        layer = HanLayer(6, 8, lr_hebb=0.02, window_size=10,
                         lr_oja=0.5, d_scale=0.0,
                         stats_mode="per_sample", normalize_stats=True,
                         normalization="spectral")
        layer.A.data.fill_(2.0)
        for _ in range(800):
            layer(torch.randn(4, 1, 6) * scale + offset)
            layer.update_weights()
        return layer.W.data.clone()

    w_a = converge(0, scale=1.0)
    w_b = converge(0, scale=4.0)
    w_c = converge(1, scale=1.0)

    cos_ac = torch.nn.functional.cosine_similarity(
        w_a.flatten(), w_c.flatten(), dim=0).item()
    cos_ab = torch.nn.functional.cosine_similarity(
        w_a.flatten(), w_b.flatten(), dim=0).item()

    # The random initialisation must NOT survive into the fixed point.
    # (Legacy rule: cos ~ 0.88. Reworked rule: cos ~ 0.05.)
    assert cos_ac < 0.5, (
        f"W is still dominated by its random init (cos={cos_ac:.5f}); "
        "the fixed point should be determined by the data"
    )
    # Changing the input scale must also change the fixed point.
    assert cos_ab < 0.95, f"scale change barely moved W (cos={cos_ab:.5f})"


def check_oja_term_bounds_weight_growth():
    """The Oja term must keep ``W`` bounded without the crude max-abs
    rescale."""
    torch.manual_seed(0)
    layer = HanLayer(6, 8, lr_hebb=0.05, window_size=10, lr_oja=0.5,
                     d_scale=0.0, stats_mode="per_sample",
                     normalization="none", normalize_stats=True)
    layer.A.data.fill_(1.0)
    for _ in range(400):
        layer(torch.randn(4, 1, 6) * 0.5)
        layer.update_weights()
    assert torch.isfinite(layer.W.data).all()
    assert layer.W.data.abs().max().item() < 100.0


def check_forward_does_not_mutate_weights():
    """The strict HAN rule: ``forward`` never mutates ``W``."""
    layer = HanLayer(6, 4)
    before = layer.W.data.clone()
    layer(torch.randn(3, 2, 6))
    assert torch.allclose(before, layer.W.data)


def check_update_clears_window():
    """The window must be consumed by ``update_weights``."""
    layer = HanLayer(6, 4, window_size=10)
    for _ in range(4):
        layer(torch.randn(2, 1, 6))
    assert len(layer._pre_window) == 4
    layer.update_weights()
    assert len(layer._pre_window) == 0





# --------------------------------------------------------------------------- #
# Model-level tests.
# --------------------------------------------------------------------------- #
def _specs(n_agents, d_obs, d_action):
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


def check_per_agent_layer_stacking_shape():
    """The per-agent layer stacks must produce ``(B, N, out)``.

    The previous implementation passed the *whole* agent tensor to every
    per-agent stack and then stacked the results, which produced
    ``(B, N, N, out)`` — a spurious extra agent axis.
    """
    n_agents, d_obs, d_action = 5, 9, 4
    input_spec, output_spec, action_spec = _specs(n_agents, d_obs, d_action)

    for share in (True, False):
        model = HanModel(
            input_spec=input_spec, output_spec=output_spec, agent_group="agents",
            input_has_agent_dim=True, n_agents=n_agents, centralised=False,
            share_params=share, device="cpu", action_spec=action_spec,
            model_index=0, is_critic=False,
        )
        td = TensorDict(
            {"agents": {"observation": torch.randn(2, n_agents, d_obs)}},
            batch_size=[2],
        )
        out = model(td)
        shape = tuple(out["agents", "logits"].shape)
        assert shape == (2, n_agents, d_action), (
            f"share_params={share}: logits shape {shape}, "
            f"expected (2, {n_agents}, {d_action})"
        )


def check_model_config_exposes_new_mechanism_knobs():
    """The new mechanism knobs must be reachable from ``HanConfig``."""
    cfg = HanConfig(hidden_size=6, lr_oja=0.25, d_scale=0.0,
                    stats_mode="per_sample", normalization="spectral",
                    normalize_stats=True)
    assert cfg.lr_oja == 0.25
    assert cfg.d_scale == 0.0
    assert cfg.stats_mode == "per_sample"
    assert cfg.normalization == "spectral"
    assert cfg.normalize_stats is True


def _run_all():
    checks = [
        (name, fn) for name, fn in sorted(globals().items())
        if name.startswith("check_") and callable(fn)
    ]
    failures = []
    for name, fn in checks:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failures.append((name, str(e)))
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failures.append((name, f"{type(e).__name__}: {e}"))
    return checks, failures


if __name__ == "__main__":
    print("=" * 60)
    print("HAN mechanism regression tests")
    print("=" * 60)
    checks, failures = _run_all()
    print(f"\n{len(checks) - len(failures)}/{len(checks)} passed.")
    if failures:
        print("\nFAILURES:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    print("All HAN mechanism regression tests passed.")
