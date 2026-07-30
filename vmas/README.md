# Local VMAS scenarios

This directory mirrors VMAS scenario files maintained alongside the
BenchMARL repository. Scenarios here are tracked in git; they need to be
kept in sync with the VMAS install used at runtime (currently
`miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/`).

## Files

| Scenario | Purpose | Synced from VMAS |
|---|---|---|
| `navigation_obs_avoidance.py` | Single-holonomic-agent navigation with N static obstacles; 9-dim observation `[pos, vel, goal_rel, nearest_obstacle_rel, has_flag]`. | new — added 2026-07-30 for the HAN obstacle-avoidance experiment |

## Sync workflow

When you edit a scenario file:

```bash
SRC=/home/zhaozeming/BenchMARL/vmas/scenarios/<scenario>.py
DST=/home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/<scenario>.py
cp "$SRC" "$DST"
```

And re-register in `vmas/__init__.py` (the `scenarios` sorted list) if it is a
new scenario. The BenchMARL runtime always imports the VMAS package from
site-packages — files under `vmas/scenarios/` in this repo are the
git-tracked source of truth.
