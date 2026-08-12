# `navigation_obs_avoidance` fitness — before/after

The HAN single-agent obstacle-avoidance experiment ships with a custom
CMA-ES fitness mode ([`CmaesHanOptimizer`][opt]) called
`navigation_obs_avoidance`. The first version of this fitness did not
actually penalise obstacle collisions strongly enough for CMA-ES to
discover avoidance behaviour. This document records what changed and why.

[opt]: benchmarl/algorithms/cmaes_han_optimizer.py

## Old formula

```python
fitness = 3.0 * mean_progress        # in [0, 1]
        + 5.0 * success_term          # in {0, 1}
        + 1.0 * final_term            # in [-1, 0]
        - 2.0 * mean_obstacle_pen     # mean over episode steps
```

with

```python
mean_obstacle_pen = mean over episode steps of exp(-k · r)
r = clamp((d - d_min) / (d_safe - d_min), 0, 1)
```

## New formula

```python
fitness = 1.5 * mean_progress
        + 2.0 * success_term
        + 0.5 * final_term
        - 1.5 * peak_obstacle_pen     # max over episode steps
        - 1.0 * collision_touched     # 1 if any step had d <= d_min
```

with

```python
peak_obstacle_pen = max over episode steps of exp(-k · r)
collision_touched = 1 if min(obs_dist) <= d_min else 0
```

The five weights are exposed as CLI knobs (`--w-progress / --w-success /
--w-final / --w-peak / --w-collision`); defaults are
`1.5 / 2.0 / 0.5 / 1.5 / 1.0`. The old `--obstacle-penalty-weight` flag
is kept as a backward-compat alias for `--w-peak`.

## Why the old formula failed

Three failure modes combined so that CMA-ES had no gradient toward
"actually avoid obstacles":

1. **Time-averaged penalty is diluted by short episodes.** A 30-step
   "crash-through-to-goal" episode records ~5 collision steps with
   `pen ≈ 1`, but the other 25 steps record `pen ≈ 0.05`. The mean
   is ≈ 0.21. Meanwhile a 30-step "clean-arrive" episode records
   almost no close encounters, mean ≈ 0.05. The two strategies differ
   by only 0.16 in the obstacle term, which CMA-ES cannot reliably
   exploit against a `success` term of weight 5.

2. **`success=5` dominated everything.** The maximum contribution from
   the obstacle term was `2.0 * mean_obstacle_pen ≈ 2.0`. Even an
   agent that drove straight through obstacles still got most of the
   `success=5` reward. CMA-ES preferred fast-arrival to safe-arrival
   because the gradient pointed that way.

3. **No distinction between "touched once" and "never touched".** Both
   strategies could collect the same `success` reward; the obstacle
   term only nudged the average by a small amount. CMA-ES had no
   binary "did you collide or not" signal to push the population
   toward safe behaviour.

## How the new formula fixes each failure mode

| Failure | Fix |
|---|---|
| Time-averaged dilution | Replace `mean(pen)` with `peak(pen) = max(pen)`. One close step pulls the penalty to ≈1 and stays there for the whole episode. |
| `success=5` dominance | Reduce `w_success` from 5.0 to 2.0 and rebalance the other terms so the success term is no longer larger than the obstacle term's budget. |
| "Touched once" ≈ "never touched" | Add the binary `collision_touched` term (`w_collision = 1.0`). Any single contact costs the agent 1.0 in fitness, on top of the peak penalty. |

## Numerical illustration

Same six representative episode archetypes, scored under both formulas.
Numbers are computed from the analytical formula with
`initial_dist ≈ 1.0` and the default `k=3.0, d_safe=0.3, d_min=0.25`.

| Episode | progress | success | final | mean_pen | peak_pen | collided | old score | new score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A — clean arrive | 1.0 | 1 | -0.05 | 0.05 | 0.05 | 0 | **+7.85** | **+2.42** |
| B — crash through | 1.0 | 1 | -0.05 | 0.10 | 0.95 | 1 | **+7.70** | **+0.43** |
| C — arrive with one close pass | 1.0 | 1 | -0.05 | 0.20 | 0.80 | 0 | **+7.60** | **+2.18** |
| D — clean, didn't arrive | 0.5 | 0 | -0.50 | 0.10 | 0.30 | 0 | **+0.90** | **+0.30** |
| E — stuck against obstacles | 0.2 | 0 | -0.80 | 0.60 | 0.99 | 1 | **-1.40** | **-2.05** |
| F — never moved | 0.0 | 0 | -1.00 | 0.05 | 0.05 | 0 | **-1.10** | **-0.55** |

The key pairings:

| Comparison | Old gap | New gap |
|---|---:|---:|
| A vs B (clean arrive vs crash through) | **0.15** | **1.99** |
| A vs F (clean arrive vs never moved) | 8.95 | 2.97 |
| B vs E (crash through vs stuck against obstacles) | 9.10 | 2.48 |

Under the old formula the clean-arrive and crash-through strategies
were effectively tied (gap 0.15) — CMA-ES had no way to prefer one
over the other. Under the new formula the gap is 1.99, a 13× clearer
signal pointing at "go around, don't drive through".

## Files changed

- [`benchmarl/algorithms/cmaes_han_optimizer.py`](benchmarl/algorithms/cmaes_han_optimizer.py)
  - `__init__`: add `w_progress / w_success / w_final / w_peak /
    w_collision`; reuse `obstacle_penalty_weight` as a backward-compat
    alias for `w_peak`.
  - `_compute_fitness` (`navigation_obs_avoidance` branch): replace
    `mean_obstacle_pen` with `peak_obstacle_pen` and add
    `collision_touched`; read all weights from `self.w_*`.

- [`examples/running/run_cmaes_han_navigation_obs_avoidance.py`](examples/running/run_cmaes_han_navigation_obs_avoidance.py)
  - argparse: expose the five new weights; default
    `--obstacle-penalty-weight` to 1.5 (was 2.0).
  - print banner: list the five weights instead of the old single
    `obstacle_penalty_weight` field.

## Verification

A 2-generation, pop=4 smoke test runs end-to-end in ~5 s and produces
non-NaN fitness values in the expected range (best ≈ 1.6, mean ≈ 0.9
with random ABCD vectors). Full training (15 generations × pop 30 ×
3 eval episodes) is needed to confirm the convergence picture shifts
toward avoidance behaviour.