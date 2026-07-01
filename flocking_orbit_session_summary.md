# 6/23 - 6/26 对话总结

## 项目背景

在 BenchMARL 中为 HAN(可塑性 Hebbian 网络)训练 flocking 任务,验证 HAN 的 `flocking_orbit` 任务表现,并最终通过 **frozen agent 扰动实验** 测试 HAN 的动态适应能力。

## 核心进展

### 1. flocking_orbit 任务设计

定义了一个新的 fitness 模式,公式为:

```
F_orbit = 1.5×At + Dt + 0.2×Cg + 0.8×S   (方案 E)
```

其中:
- `At`: 切线对齐 (target→agent 的切线方向 vs agent 速度)
- `Dt`: 距 target 的高斯距离带 (中心 `orbit_radius=0.7`, σ=0.3)
- `Cg`: 1/连通分量数(全局连通性)
- `S`: 1-碰撞率(分离度)

权重经过多轮调试,发现 `Cg` 权重过高会导致 HAN 卡在"聚团"局部最优。

### 2. 观测模式(nn/lidar 二选一)

| 模式 | 维度 | 组成 |
|------|------|------|
| nn (默认) | 10 | pos(2)+vel(2)+target_rel(2)+nn_rel_pos(2)+nn_rel_vel(2) |
| lidar | 18 | pos(2)+vel(2)+target_rel(2)+lidar(12) |

lidar 模式有 12 条射线检测其他 agent 的距离,nn 模式只感知最近 1 个邻居。

**最终采用 nn 模式**。lidar 模式已被回退。

### 3. 共享 patch 模块 `flocking_patch.py`

新文件,统一处理:
- 静止 target (覆盖 `action_script_creator`)
- target 居中 (覆盖 `reset_world_at`,默认位置 (0, 0))
- target 无实体状态 (`_patched_make_world` 中 `collide=False`,避免智能体撞到 target)
- 10 维 nn 观测 (覆盖 `observation`)

训练和评估脚本都通过 `from flocking_patch import configure` 共享配置。

### 4. 关键超参(基于 b140e5f5 成功实验)

| 参数 | 值 |
|------|---|
| n_agents | **4** (5+ 会失败) |
| max_steps | 800 |
| n_obstacles | 0 |
| hidden_size | 10 |
| n_eval_episodes | 2 |
| pop_size | 30 |
| sigma0 | 0.3 (默认) |
| cmaes_gens | 30 |
| neighbor_radius | 0.5 |
| orbit_radius | 0.7 |
| lr_hebb | 0.01 |
| target_pos | (0.0, 0.0) |

### 5. 成功实验 vs 失败实验

| 实验 | obs_mode | n_agents | max_steps | best_fitness | 状态 |
|------|----------|----------|-----------|--------------|------|
| **b140e5f5** | nn (10-dim) | 4 | 800 | **3.088** | ✅ |
| 2731a1e9 | lidar (18-dim) | 5 | 1600 | 2.435 | ❌ |
| d49d4b5e | nn (10-dim) | 5 | 1600 | 2.645 | ❌ |

**失败原因**:5 agent 弦长 ≈ 0.82,超出 neighbor_radius=0.5,Cg 项几乎永远=0.2,梯度太弱导致发散。

## Frozen Agent 扰动实验

### 目的

测试 HAN 训练出的策略是否具有**动态适应能力**——在 episode 中途突然冻结一个 agent,看剩下 3 个 agent 能否继续 flocking。

### 关键 bug 修复

1. **disturbance-step 单次生效 bug**: `if step == disturbance_step` 改为 `if step >= disturbance_step` (持续 override)
2. **td[action_key][frozen_idx] 索引错误**: `[:, frozen_idx]` 而非 `[frozen_idx]`(否则冻结 env 0 所有 agent)
3. **frozen agent 被推开**: 强制每步还原 `state.pos/vel/force`
4. **target 有物理碰撞**: 用 `_patched_make_world` 改 `collide=False`
5. **环境不一致**(target 动/有障碍物/obs_mode 不匹配): 统一通过 `flocking_patch` 解决

### 实现

`run_cmaes_han_flocking_disturbance.py`:
- 每代步骤累加
- frozen agent 锚定到 step=disturbance_step 时的位置
- 强制 `state.pos/vel/force=0`
- 输出: per_step_data.npz, fitness_curve.png (2 子图), trajectory.mp4

### 关键结果(用 b140e5f5 模型,disturbance_step=400, frozen agent #0)

```
Baseline [0..400):    Fg=2.067±0.381
Immediate post [400..500):  Fg=2.309±0.100
Long post [500..800):      Fg=2.361±0.054
Full post [400..800):      Fg=2.348±0.072

Fitness drop after disturbance: -0.281
→ HAN is ROBUST to the disturbance (fitness barely changed).
```

**frozen agent 在 400 步内漂移 0.000000**(完美静止),HAN 行为从 2.07 升到 2.35(+0.28),说明 HAN 学到了 robust 策略。

## 已回退的改动

6/23 后我做了很多改动,对话末尾你要求**回退雷达功能**,目前状态:

| 改动 | 状态 |
|------|------|
| target 无实体状态 (`_patched_make_world`) | **保留** |
| target 居中 (`_patched_reset_world_at`) | **保留** |
| frozen agent 当障碍物(disturbance 脚本) | **保留** |
| 10 维 nn 观测 | **保留** |
| `--target-pos-x/y` CLI | **保留** |
| `--hidden-size=10` 默认 | **保留** |
| 18 维 lidar 模式 | **已删除** |
| `--obs-mode lidar/--lidar-*` CLI | **已删除** |
| `_lidar_agent_detection()` 函数 | **已删除** |
| 方案 E 权重(1.5×At + 0.2×Cg + 0.8×S) | **保留** |
| `_patched_load` + `configure()` | **保留** |
| `experiment_config.loggers = []` (关 wandb) | **保留** |
| `sensors=None`(删 VMAS Lidar) | **保留** |

## 当前环境

- **conda env**: benchmarl
- **NumPy 版本**: 1.26.4 (从 2.2.6 降级)
- **PyTorch**: 2.x (用 NumPy 1.x ABI 编译)

**关键依赖修复**:
```bash
pip install "numpy<2"
```

(原因: torch 2.x + numpy 2.x 不兼容,触发 `RuntimeError: Numpy is not available`)

## 关键文件

| 文件 | 作用 |
|------|------|
| `examples/running/flocking_patch.py` | 共享 patch 模块(target 静止+居中+无实体+10 维 nn 观测) |
| `examples/running/run_cmaes_han_flocking_custom.py` | 训练脚本(替代原版 run_cmaes_han.py) |
| `examples/running/run_cmaes_han_flocking_disturbance.py` | 扰动实验评估脚本 |
| `benchmarl/algorithms/cmaes_han_optimizer.py` | 包含 `_compute_flocking_orbit_fitness` (方案 E 权重) |

## 性能优化

1. **关 wandb**: `experiment_config.loggers = []` —— 避免每步网络同步
2. **关 VMAS Lidar**: `sensors=None` —— 避免每步 12 条射线计算
3. **单代时间**: pop_size=30, n_eval=2, max_steps=800 → **~50-90 秒**(干净 CPU 下)

## 待解决 / 待验证

1. **未跑原版(无 patch)做对比**:怀疑之前的 22 分钟/代是机器其他用户抢 CPU 导致(实测 load avg 40,被严重抢占),但我没完成这个对比实验。
2. **failure 5-agent 问题未解决**:扩展智能体数量时需要调整 neighbor_radius 或 orbit_radius,而不是简单增加 n_agents。
3. **lidar 模式已被回退**:如果将来想用 18 维观测,需要重新启用 `_lidar_agent_detection` 和相关 CLI 参数。

## 复现 b140e5f5 训练的命令

```bash
cd /home/zhaozeming/BenchMARL
source /home/zhaozeming/miniconda3/etc/profile.d/conda.sh && conda activate benchmarl
cd examples
python running/run_cmaes_han_flocking_custom.py \
  --task flocking --fitness-mode flocking_orbit \
  --cmaes-gens 30 --pop-size 30 --n-eval-episodes 2 --sigma0 0.3 \
  --hidden-size 10 --window-size 10 --f-nn 4 --f-hebb 1 \
  --lr-hebb 0.01 \
  --neighbor-radius 0.5 --safety-distance 0.15 \
  --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \
  --target-pos-x 0.0 --target-pos-y 0.0 \
  --n-final-eval 10 --max-video-frames 400 --fps 20
```

## 复现 b140e5f5 disturbance 评估的命令

```bash
cd /home/zhaozeming/BenchMARL
source /home/zhaozeming/miniconda3/etc/profile.d/conda.sh && conda activate benchmarl
cd examples
python running/run_cmaes_han_flocking_disturbance.py \
  --experiment-path outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8agents \
  --fitness-mode flocking_orbit \
  --hidden-size 10 --window-size 10 --f-nn 4 --f-hebb 1 \
  --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \
  --neighbor-radius 0.5 --safety-distance 0.15 \
  --disturbance-step 400 --frozen-agent-idx 0 \
  --target-pos-x 0.0 --target-pos-y 0.0 \
  --n-final-eval 10 --max-video-frames 800 --fps 20
```

## 实验结果速查

| 实验 | 路径 | best_fitness | 备注 |
|------|------|--------------|------|
| b140e5f5 (成功) | `examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8agents/` | 3.088 | 4 agent, 800 step, 30 gen |
| 2731a1e9 (失败) | `examples/outputs/cmaeshan_flocking_hanmodel__2731a1e9_26_06_25-19_16_31/` | 2.435 | 5 agent, 1600 step, lidar 18-dim |
| d49d4b5e (失败) | `examples/outputs/cmaeshan_flocking_hanmodel__d49d4b5e_26_06_25-19_16_32/` | 2.645 | 5 agent, 1600 step, nn 10-dim |
| 5a5d0cd1 (中途中断) | `examples/outputs/cmaeshan_flocking_hanmodel__5a5d0cd1_26_06_25-11_08_08/` | N/A | 只 config.pkl,无 results.json |
| 8c466930 (1代训练) | `examples/outputs/cmaeshan_flocking_hanmodel__8c466930_26_06_24-19_33_17/` | 1.906 | 只 1 代, lidar 18-dim |

## 关键设计决策

1. **方案 E 权重(1.5×At + 0.2×Cg + 0.8×S)** vs 默认 (1.0×At + 0.5×Cg + 0.5×S):
   - 默认权重让 HAN 卡在"聚团"局部最优(Fg ≈ 1.85)
   - 方案 E 把绕圈 vs 聚团差距从 1.05 拉大到 1.52,CMA-ES 梯度更强

2. **At 耦合速度幅度**(原版 At 在静止时 = 0.5 不合理):
   - 旧: `At = mean((dot+1)/2)`,静止时为 0.5
   - 新: `At = mean((dot+1)/2 * speed_factor)`,静止时为 0
   - 加 `speed_threshold=0.02`,低速时 At 线性衰减

3. **n_agents=4 vs 5**:
   - 4 agent 弦长 0.99,刚好超 neighbor_radius=0.5,Cg=0.25 经常出现
   - 5 agent 弦长 0.82,全部超,Cg=0.20 几乎永远,**fitness 梯度太弱**
