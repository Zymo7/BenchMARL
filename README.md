# Hebbian MARL — A Research Toolkit for Plastic-Hebbian and Gradient-Based Multi-Agent Learning

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Built on BenchMARL](https://img.shields.io/badge/BenchMARL-fork-orange.svg)](https://github.com/facebookresearch/BenchMARL)

A research codebase for studying **biologically inspired, locally plastic
neural dynamics** in cooperative / competitive multi-agent tasks. The
project sits on top of [BenchMARL](https://github.com/facebookresearch/BenchMARL)
(reusing its env, algorithm, model and hydra-config scaffolding) but
adds a family of Hebbian-style actor networks, derivative-free (CMA-ES)
and gradient-based (IPPO+Hebbian) training pipelines, and a set of
custom VMAS scenarios for studying emergent coordination, encirclement,
hierarchical control and pursuit.

> **In one line:** compare gradient-trained policies, evolution-trained
> Hebbian policies, and policy-network plasticity in one library.

---

## Why this project?

Standard MARL benchmarks reward policies that **solve** the task. We are
also interested in policies that **stay solved** when individual agents
are knocked offline or pushed out of formation, and in policies whose
internal weights keep adapting at deployment time without backprop.
Concretely:

1. **Hand-designed policies carry hidden inductive biases.** When a
   static-weight MLP out-performs a Hebbian-plastic policy, is that
   because the Hebbian rule is wrong, or because the local Hebbian
   statistics are not aligned with what the env rewards?
2. **Robustness is rarely studied inside the training loop.** Most MARL
   work evaluates on a clean test env; we want explicit *disturbance*
   probes (freeze a teammate, push a teammate) and recovery metrics.
3. **Plastic networks make biological/online-learning claims
   measurable.** If "online Hebbian updates at test time" helps, we
   should see it. If it does not, the difference should be a number.

This repo is our scratchpad for asking those questions in a single,
reproducible place.

---

## Repository tour

```
benchmarl/
├── algorithms/
│   ├── cmaes_han.py              # CMA-ES outer-loop algorithm wrapping HAN
│   ├── cmaes_han_optimizer.py    # CMA-ES + fitness functions for plastic policies
│   ├── cmaes_static_mlp.py       # CMA-ES baseline: weight-only MLP (no plasticity)
│   ├── cmaes_static_mlp_optimizer.py
│   ├── ippo_hebbian.py           # IPPO augmented with online Hebbian layer
│   ├── hebbian.py                # legacy Hebbian baselines
│   └── …
├── environments/vmas/
│   ├── flocking.py               # 2-D cyclic encirclement around a target
│   ├── flocking_light.py         # encirclement + light source (extra obs dim)
│   ├── flocking_signal.py        # encirclement + dynamic target signal
│   ├── flocking_lf.py            # leader-follower flocking
│   └── simple_tag_v1.py          # obstacle-free pursuer-vs-evader (added)
├── models/
│   ├── han.py                    # Hebbian Attractor Network (HAN) — see below
│   ├── static_mlp.py             # weight-only MLP baseline (same param budget as HAN)
│   └── full_hebbian.py / hebbian.py  # older legacy plastic baselines
└── conf/                         # standard BenchMARL hydra configs

examples/running/
├── run_cmaes_han_flocking*.py            # CMA-ES HAN training on each flocking variant
├── run_cmaes_han_flocking_disturbance.py # freeze/push disturbance probe (HAN)
├── run_cmaes_static_mlp_flocking*.py     # static-MLP baselines + their disturbance probe
├── run_cmaes_han_simple_tag_v1.py        # CMA-ES HAN on pursuer-vs-evader (VMAS)
├── run_cmaes_han_pettingzoo_simple_tag.py # CMA-ES HAN on the PettingZoo MPE version
├── run_ippo_hebbian.py                   # IPPO with a Hebbian head on navigation
├── run_ippo_hebbian_dynamic_obs.py
├── run_comparison_eval.py                # geometric comparison HAN vs static-MLP
├── plot_comparison.py                    # T_stable, T_recover, quality vs N plots
└── flocking_patch.py                     # observation + collision patches for flocking
```

The per-task deep dives live as standalone markdown files at the repo
root:

| File                        | Contents |
|-----------------------------|----------|
| `HAN_README.md`             | HAN — design, mechanisms, API |
| `HAN_flocking.md`           | HAN applied to the four flocking variants |
| `HAN_lf.md`                 | Leader-follower flocking with HAN |
| `HAN_tag.md`                | HAN applied to simple-tag pursuit |
| `flocking_fitness.md`       | The orbit-fitness breakdown (At, Dt, Cg, S) |
| `flocking_orbit_session_summary.md` | head-to-head session log for flocking |

For a code-level map see `CODEBASE_GUIDE.md`.

---

## 1. The flagship study: HAN on flocking + disturbance robustness

This is the main body of work in the repo. The hypothesis we are
stress-testing is:

> *A Hebbian-plastic actor can be trained without gradients (CMA-ES)
> to orbit a moving target cooperatively; its online weight updates
> should help it recover from agent disturbances faster than a matched
> size static-weight baseline.*

### 1.1 Tasks

Four progressively harder flocking scenarios, all built on VMAS:

| Task             | n_agents | World                                                              |
|------------------|---------:|--------------------------------------------------------------------|
| `flocking`       |    4     | Cyclic encirclement of a single target at `(0, 0)`, radius 0.7.   |
| `flocking_light` |    4     | + an explicit light source the agents are attracted to (extra dim).|
| `flocking_signal`|    4     | + a moving signal source (non-static target).                     |
| `flocking_lf`    |   1+4    | One leader, four followers (leader-follower formation).            |

Each variant comes with the same training wrapper
(`run_cmaes_han_flocking_<variant>.py`) and the same observation
patch (`flocking_patch.py`).

### 1.2 Observation (10-D, fixed)

Every agent sees the same 10-D vector regardless of `n_agents`:

```
obs = [ self_pos (2), self_vel (2), target_rel (2),
        nearest_neighbor_rel_pos (2), nearest_neighbor_rel_vel (2) ]
```

- `self_pos`, `self_vel` — `agent.state.pos`, `agent.state.vel`.
- `target_rel` — agent's position relative to the encirclement target.
- nearest-neighbor — closest other policy-controlled agent within
  `_NEIGHBOR_RADIUS = 0.5`; if no one is in range, both slots are filled
  with zeros (no NaN, no inf).

The flocking env patches also remove the default Lidar sensor (which
would otherwise add 12 obs dims) and disable collisions with the target.

### 1.3 Fitness: an orbit quality index `F_orbit`

CMA-ES optimizes a **single scalar per episode**:

```
F_orbit = 1.5 · At + Dt + 0.2 · Cg + 0.8 · S
```

| Component | Range | Meaning |
|-----------|-------|---------|
| `At`      | [0,1] | Tangential alignment — how well each agent's velocity aligns with the counter-clockwise tangent to the orbit. |
| `Dt`      | [0.1,1] | Distance-from-band Gaussian — `exp(-(r − r★)² / (2 r_σ²))`, clamped to a floor. |
| `Cg`      | [0,1] | Cohesion — `1 / (# connected components in the neighbour graph)`. Prevents the "N independent circles" failure mode. |
| `S`       | [0,1] | Safety — `1 − (fraction of agents in collision)`. |

Weights are deliberately imbalanced: alignment and distance band carry
the cooperative geometry; cohesion is a cheap tie-breaker; safety is a
soft penalty. (HAN-side training uses `1.0 · S`, the static-MLP side
uses `0.8 · S`; the **evaluation** code (`run_comparison_eval.py`) always
re-scores with the static-MLP weights for a head-to-head comparison.)

The geometry constants are passed in by `flocking_patch.configure()`:

| Parameter        | Value |
|------------------|-------|
| `orbit_radius` (r★) | 0.7 |
| `orbit_radius_tolerance` (r_σ) | 0.3 |
| `neighbor_radius` | 0.5 |
| `safety_distance` | 0.15 |
| `dt_floor`        | 0.1   |
| `target_pos`      | (0, 0) |

### 1.4 HAN — Hebbian Attractor Network

`benchmarl/models/han.py` provides a small custom `nn.Module`
(`HanModel`) that interleaves `HanLayer`s where each weight matrix `W`
is updated by a generalized Hebbian ABCD rule:

```
Δw_{ij} = η · ( A_{ij} · x̄_pre[j] · x̄_post[i]
              + B_{ij} · x̄_pre[j]
              + C_{ij} · x̄_post[i]
              + D_{ij} )
```

HAN introduces three **hard** mechanisms that the legacy
`FullHebbianModel` did not have:

1. **Inference / weight-update decoupling.** `forward()` never modifies
   `W`; an explicit `update_weights()` call — gated by
   `ticks % (f_nn // f_hebb) == 0` — does.
2. **Sliding-window time average.** Each layer keeps two `deque`s of
   post-`tanh` activations; `update_weights()` first stacks the last
   `window_size` steps and *averages* before applying ΔW. No single-step
   noise can dominate.
3. **Hard per-layer normalization.** After every ΔW, the layer divides
   its entire weight matrix by `max|W|`, forcing `max(|W|) ≡ 1.0` for
   the lifetime of training. No weight decay, no clipping, no decay.

The CMA-ES outer loop treats the per-layer ABCD vectors as the only
search parameters — there are no gradients anywhere in the pipeline.
For the canonical 10→40→40→4 architecture this is 4 × (10·40 + 40·40 +
40·4) = 8,800 parameters (across all layers), but a much smaller
budget can be obtained by adjusting `hidden_size` (see
`run_cmaes_han.py --help`).

### 1.5 Disturbance probes (freeze & push)

Beyond clean evaluation we have an explicit robustness harness
(`run_cmaes_han_flocking_disturbance.py`,
`run_cmaes_static_mlp_flocking_disturbance.py`) that takes any trained
policy and applies one of two disturbances at a user-specified step:

| Disturbance     | What happens                                                         |
|-----------------|----------------------------------------------------------------------|
| `--disturbance-mode freeze` | The chosen agent's action is forced to zero and its position / velocity / force are pinned each step. Equivalent to "agent dies in place" — the swarm has to absorb the loss. |
| `--disturbance-mode push`   | A one-shot velocity impulse is applied to the chosen agent in either `radial-out` or `fixed-x` direction. The agent keeps acting afterwards; the swarm has to absorb a momentum shock. |

Each mode writes its own sub-folder (`disturbance_eval_freeze/`,
`disturbance_eval_push/`) containing:

- `trajectory.mp4` — a 20 fps video of the full episode.
- `fitness_curve.png` — `F_orbit` over time.
- `per_step_data.npz` — raw `At, Dt, Cg, S` time series for offline
  analysis.

### 1.6 Head-to-head metrics (HAN vs static-MLP)

Because raw `F_orbit` numbers reward whatever a policy was optimized
for, we added a more interpretable **comparison** suite
(`examples/running/run_comparison_eval.py`,
`examples/running/plot_comparison.py`,
`examples/run_eval.py`) that re-scores all candidates on the same
freshly-rolled-out episodes. The metrics are:

| Metric            | Definition |
|-------------------|------------|
| `T_stable`        | First time index at which the formation sustained the geometric stability rule for `stable_window = 20` consecutive steps. |
| `stable_quality`  | Composite in-window score: `0.4·radius + 0.3·angle + 0.2·direction + 0.1·eccentricity`. |
| `T_recover`       | After the disturbance step, time to next sustained-stability window **on the same remaining agents** (`n_keep ≥ 2`). Set to "never" if no recovery. |
| `error_vs_N`      | Same `stable_quality` averaged across `n_agents ∈ {4, 6, 8, 10, 12}` (a transfer test of the policy trained at `N=4`). |

"Never stable" and "never recovered" episodes are drawn as **hatched**
bars, so missing bars are explicit rather than zero-baselined.

#### Run the comparison

```bash
# 1. Run all (N, mode) combinations
python examples/run_eval.py \
    --han-experiments outputs/cmaeshan_flocking_hanmodel__*/ \
    --static-mlp-experiments outputs/cmaesstaticmlp_flocking_staticmlpmodel__*/ \
    --n-agents 4,6,8,10,12 \
    --disturbance-modes freeze push \
    --scenarios frozen_agent --max-steps 500

# 2. Aggregate + plot
python examples/running/plot_comparison.py --output-dir plots/
```

The output (under `plots/`):

- `t_stable.png`, `t_recover.png` — bar charts with hatched bars for
  never-stable / never-recovered.
- `f_quality.png` — `stable_quality` over (N, algo).
- `error_vs_N.png` — quality across N (transfer-robustness).

#### Probe a single disturbance episode with video

```bash
python examples/running/run_cmaes_han_flocking_disturbance.py \
    --experiment-path outputs/cmaeshan_flocking_hanmodel__<id> \
    --fitness-mode flocking_orbit \
    --disturbance-mode push \
    --disturbance-step 200 --frozen-agent-idx 2 \
    --push-magnitude 0.5 --push-direction radial-out \
    --max-steps 300
```

The same script exists for the static-MLP baseline
(`run_cmaes_static_mlp_flocking_disturbance.py`).

### 1.7 Heads-up: what the data currently shows

The disturbance harness and the four metrics above were the user's
explicit answer to *"the original comparison just showed HAN is worse
than static-MLP"*. With the new metrics, we now report the actual
behaviour honestly:

- On `T_stable` and transfer to larger `N`, the static-MLP baseline
  matches or beats HAN. We could not confirm the hypothesis "HAN is
  more N-robust than static-MLP".
- On **push** disturbance, both networks are surprisingly robust
  (`F_orbit` drop ≤ ~0.1 at `push_magnitude = 0.5`). HAN's online
  plasticity does not yet translate into a clearly faster
  `T_recover`.
- On **freeze**, the gap is real: `Cg` (cohesion) collapses once a
  neighbour is held in place. HAN's incremental rule partially
  compensates, but neither policy recovers to the
  pre-disturbance `At · Dt · S` triple.

This repository is honest about negative results — see
`HAN_flocking.md` and the in-tree session notes for the full numbers.

---

## 2. IPPO + Hebbian: gradient-trained, plastic at deployment

The other flagship direction is to keep PPO-style gradient training
but plug a Hebbian-plastic layer **inside** the policy so that weights
keep adapting at test time. The implementation lives in:

- `benchmarl/algorithms/ippo_hebbian.py` — IPPO variant whose actor
  backbone is the same `HanLayer`-style Hebbian layer, *plus* a
  standard PPO loss on top.
- `examples/running/run_ippo_hebbian.py` — single-process training
  driver.
- `examples/running/run_ippo_hebbian_dynamic_obs.py` — variant where
  obstacles move and re-spawn, so the policy must keep adapting.

### 2.1 Tasks

| Task                            | Notes |
|---------------------------------|-------|
| `vmas/navigation`               | Standard navigation-to-goal.         |
| `vmas/navigation_obs`           | With static obstacles.              |
| `vmas/navigation_dynamic_obs`   | Obstacles re-spawn at random each episode. The headline task — Hebbian layer has to update against a non-stationary obstacle layout. |
| `vmas/navigation_static_dynamic_obs` | Mix of static + dynamic obstacles. |

### 2.2 Why pair IPPO and Hebbian?

- PPO optimizes a value-loss surrogate that is local *in space* (one
  gradient step) but global *in time* (whole rollout).
- A Hebbian layer is local *in time* but produces statistics that PPO
  cannot discover by itself (covariance between hidden activations,
  resistance to weight drift, etc.).

The combination we are probing: let PPO set the high-level policy
direction, let the Hebbian layer absorb short-term non-stationarity. The
actuator is still a deterministic Gaussian whose parameters pass through
both heads.

### 2.3 Smoke run

```bash
python examples/running/run_ippo_hebbian_dynamic_obs.py \
    task=vmas/navigation_dynamic_obs \
    algorithm=ippo_hebbian \
    "model=han" "model.hidden_size=64" "model.window_size=10"
```

Look for the `friction`-like plateaus in `wandb` — the test we are
actually running is "does the policy still solve the env on episodes
with completely new obstacle layouts, even though it has never seen
them during training?"

---

## 3. In-progress / exploratory work

The repo also hosts two **early-stage** lines of work that we are
actively iterating on but haven't yet promoted to the headline section.

### 3.1 Leader-follower flocking (`flocking_lf`)

- `benchmarl/environments/vmas/flocking_lf.py` — a VMAS scenario with
  `n_leaders` guiding `n_followers`. Followers are rewarded for staying
  near the nearest leader; leaders are rewarded for spacing themselves.
- `examples/running/run_cmaes_han_flocking_lf.py` — CMA-ES HAN with a
  per-role fitness: leaders optimize spread + heading, followers
  optimize adjacency.
- Status: scenarios load, CMA-ES converges, but the *leader role*
  sharing is currently via two parallel `HanModel`s. We are considering
  switching to a single shared ABCD + a per-role observation tag.

See `HAN_lf.md` for the design log and known issues.

### 3.2 Pursuit with HAN (no-obstacle `simple_tag`)

A stripped-down variant of VMAS `simple_tag` (added as the new scenario
`simple_tag_v1`) lets us study isolated pursuit coordination without
lidar / obstacles muddying the picture:

| Component | Description |
|-----------|-------------|
| Scenario  | `benchmarl/environments/vmas/simple_tag_v1.py` — 1 "good" agent + N adversaries in `[-bound, bound]²`. No landmarks, no obstacles. Episode ends when any pursuer touches the good (or after `max_steps`). |
| Observation | Fixed 8-D: `[self_pos(2), self_vel(2), nearest_neighbor(2), nearest_neighbor_vel(2), nearest_good(2)]`. Independent of `n_agents`. |
| Fitness    | `simple_tag_capture`: `catch_reward − proximity_weight · mean_pursuer_to_good_distance − timeout_penalty`. Episode truncated on first catch to keep evaluation cheap. |
| Training   | `examples/running/run_cmaes_han_simple_tag_v1.py` (VMAS) and `examples/running/run_cmaes_han_pettingzoo_simple_tag.py` (PettingZoo MPE baseline for cross-check). |

Status: end-to-end pipeline works (1 episode ≈ 0.4 s on CPU, 2
episodes ≈ 0.3 s once early termination kicks in). Fitness signal is
discriminative (-4.76 timeout vs +1.63 catch on the same untrained
weights). Behavioural quality of the trained policy is still being
characterized; see `HAN_tag.md`.

### 3.3 PettingZoo MPE cross-check

`examples/running/run_cmaes_han_pettingzoo_simple_tag.py` is a
PettingZoo `simple_tag_v3` version of the same training pipeline. It
exists primarily to confirm that any result we see on VMAS `simple_tag`
is not a quirk of the VMAS observation/action wrapper. (Status:
running, no significant gap observed yet, but early-stage.)

### 3.4 Custom-fitness flocking variants

`run_cmaes_han_flocking_custom.py` /
`run_cmaes_static_mlp_flocking_custom.py` are the entry points for
plugging in a user-defined fitness function while reusing the rest of
the HAN / CMA-ES scaffolding. They were used to develop the four
geometric metrics in §1.6 and to build the wind-perturbed variants.

### 3.5 `wind_flocking` (periodic environmental forcing)

A separate VMAS scenario lives at
`benchmarl/environments/vmas/wind_flocking.py` and its YAML config in
`benchmarl/conf/task/vmas/wind_flocking.yaml`. It applies a small
sinusoidal wind force to the flock and lets us test whether HAN's
online Hebbian updates can ride out periodic non-stationarity (a
lighter-weight version of §2). Currently only the static-MLP baseline
has been run on it — adding HAN is a TODO in `HAN_flocking.md`.

---

## 4. Installation

The repo is a **fork** of [BenchMARL](https://github.com/facebookresearch/BenchMARL).
You can either install upstream BenchMARL and overlay our custom code,
or install this fork directly.

### 4.1 Clone and editable install

```bash
git clone <this-repo-url>           # e.g. HebbianMARL
cd HebbianMARL
pip install -e .
```

This pulls in:

- `torch`, `torchrl` (matching the BenchMARL version pinned in
  `setup.cfg`).
- `vmas`, `pettingzoo`, `mpe` — the env suites we use.
- `cma` — the covariance-matrix-adaptation library used by our CMA-ES
  optimizer.
- `hydra-core`, `wandb` — config & logging.

### 4.2 Optional: VMAS + MPE extras

```bash
pip install "vmas" "pettingzoo[all]" "mpe"
```

### 4.3 Hardware

- Training is single-process CPU. CMA-ES evaluation is `pop_size ×
  max_gens × n_eval_episodes` short rollouts in serial — a typical
  50-gen, 30-pop, 3-episode flocking run completes in a few hours on a
  modern laptop CPU.
- Optional `cuda` works for the static-MLP path; HAN's custom ABCD
  matrices are tiny and CPU is usually faster.

---

## 5. Quick start

The fastest possible entry into the codebase, in three commands:

```bash
# 1. Train HAN on the simplest flocking scenario (≈ 30 min on CPU)
python examples/running/run_cmaes_han_flocking.py \
    --cmaes-gens 30 --pop-size 20 --n-eval-episodes 2

# 2. Train the static-MLP baseline under the same protocol
python examples/running/run_cmaes_static_mlp_flocking_disturbance.py \
    --experiment-path outputs/cmaesstaticmlp_flocking_staticmlpmodel__<id> \
    --fitness-mode flocking_orbit \
    --max-steps 300 --num-episodes 1
```

(or use the dedicated static-MLP training entry
`run_cmaes_static_mlp_flocking_custom.py` if you have one)

```bash
# 3. Compare them
python examples/run_eval.py \
    --han-experiments         outputs/cmaeshan_flocking_hanmodel__*/ \
    --static-mlp-experiments  outputs/cmaesstaticmlp_flocking_staticmlpmodel__*/ \
    --n-agents 4,6,8 \
    --disturbance-modes freeze \
    --max-steps 400
python examples/running/plot_comparison.py --output-dir plots/
```

You should now have `plots/t_stable.png`, `plots/t_recover.png`,
`plots/f_quality.png`, `plots/error_vs_N.png`.

To try the pursuit task:

```bash
python examples/running/run_cmaes_han_simple_tag_v1.py \
    --num-good-agents 1 --num-adversaries 3 \
    --cmaes-gens 30 --pop-size 20 --n-eval-episodes 2
```

---

## 6. Contributing

This is a **research scratchpad**, not a stable product. Issues,
pull-requests and experiments that contradict our conclusions are all
welcome — please include:

- the commit hash you ran on,
- the exact CLI invocation,
- the resulting `summary.json` (for the comparison suite) or the
  resulting `han_results/results.json` (for a training run).

The biggest known TODOs are listed at the top of `CODEBASE_GUIDE.md`
and again in the "Status:" line of each experiment above.

---

## 7. Citation

If you use this codebase in academic work, please cite the upstream
BenchMARL paper:

```bibtex
@article{bettini2024benchmarl,
  author  = {Matteo Bettini and Amanda Prorok and Vincent Moens},
  title   = {BenchMARL: Benchmarking Multi-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {217},
  pages   = {1--10},
  url     = {http://jmlr.org/papers/v25/23-1612.html}
}
```

For work specifically on **HAN** (Hebbian Attractor Network), please
cite our forthcoming write-up — placeholder until the preprint lands.

## 8. License

MIT — see [LICENSE](LICENSE). The HAN / static-MLP / IPPO-Hebbian
code in this repository is original to this fork; everything in
`benchmarl/` that mirrors the upstream BenchMARL API is MIT-licensed by
Meta Platforms, Inc. and affiliates (see upstream `LICENSE`).

---

## About this codebase

This repository is a **fork of [BenchMARL](https://github.com/facebookresearch/BenchMARL)**
(Meta Platforms, licensed under MIT) that extends the original library
with:

- The **HAN** model family (`benchmarl/models/han.py`,
  `benchmarl/algorithms/cmaes_han*.py`).
- A **static-MLP** baseline with the same parameter budget
  (`benchmarl/models/static_mlp.py`,
  `benchmarl/algorithms/cmaes_static_mlp*.py`).
- An **IPPO+Hebbian** actor-critic
  (`benchmarl/algorithms/ippo_hebbian.py`).
- Custom VMAS scenarios for **flocking**, **leader-follower**, and
  **obstacle-free pursuit** (`benchmarl/environments/vmas/flocking*.py`,
  `benchmarl/environments/vmas/simple_tag_v1.py`).
- Disturbance-robustness evaluation probes and comparison plots
  (`examples/running/run_cmaes_han_flocking_disturbance.py`,
  `examples/running/run_cmaes_static_mlp_flocking_disturbance.py`,
  `examples/running/run_comparison_eval.py`,
  `examples/running/plot_comparison.py`).

Everything else — env wrappers, hydra config, TorchRL bindings, the
public API of `Experiment` and `Benchmark` — is unchanged from
upstream BenchMARL and should remain compatible with any external
algorithm/model that targets the same API surface.

To upgrade against upstream BenchMARL you can `git fetch upstream
main && git rebase upstream/main`; collisions are restricted to the
files we explicitly added (`cmaes_*`, `han.py`, `static_mlp.py`,
`flocking*.py`, `simple_tag_v1.py`, plus the example scripts under
`examples/running/run_cmaes_*`).
