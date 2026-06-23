# BenchMARL 代码库使用指南

本指南整合仓库内全部算法、模型、任务、实验配置与运行命令，并按主题分章节组织。

---

## 目录

1. [算法](#1-算法)
2. [任务与环境](#2-任务与环境)
3. [模型](#3-模型)
4. [实验配置](#4-实验配置)
5. [快捷运行命令](#5-快捷运行命令)
6. [常见问题](#6-常见问题)

---

## 1. 算法

仓库内算法分两大类：

- **基于梯度（PPO/SAC/Q 系列）**：使用 PyTorch + TorchRL 进行 on-policy 或 off-policy 训练，通过 Hydra 入口 `benchmarl/run.py` 运行；
- **基于演化（CMA-ES 系列）**：直接优化模型参数（不依赖梯度），通过 `examples/running/run_*.py` 入口运行。

### 1.1 算法总览

| 算法 | 配置文件 | 类型 | 适用动作空间 | 训练范式 | 入口脚本 |
|------|----------|------|-------------|----------|----------|
| **MAPPO** | `mappo.yaml` | On-policy | 连续 + 离散 | PPO | `benchmarl/run.py` |
| **IPPO** | `ippo.yaml` | On-policy | 连续 + 离散 | PPO | `benchmarl/run.py` |
| **IPPO-Hebbian** | `ippo_hebbian.yaml` | On-policy | 连续 + 离散 | PPO + CMA-ES 两阶段 | `examples/running/run_ippo_hebbian.py` |
| **MADDPG** | `maddpg.yaml` | Off-policy | 连续 | DDPG | `benchmarl/run.py` |
| **IDDPG** | `iddpg.yaml` | Off-policy | 连续 | DDPG | `benchmarl/run.py` |
| **MASAC** | `masac.yaml` | Off-policy | 连续 + 离散 | SAC | `benchmarl/run.py` |
| **ISAC** | `isac.yaml` | Off-policy | 连续 + 离散 | SAC | `benchmarl/run.py` |
| **QMIX** | `qmix.yaml` | Off-policy | 离散 | Q-learning | `benchmarl/run.py` |
| **VDN** | `vdn.yaml` | Off-policy | 离散 | Q-learning | `benchmarl/run.py` |
| **IQL** | `iql.yaml` | Off-policy | 离散 | Q-learning | `benchmarl/run.py` |
| **CmaesFullHebbian** | `cmaes_hebbian.yaml` | — | 连续 | CMA-ES（单阶段） | `examples/running/run_cmaes_hebbian.py` |
| **CmaesHan** | `cmaes_han.yaml` | — | 连续 | CMA-ES（单阶段） | `examples/running/run_cmaes_han.py` |

> 算法配置文件位于 `benchmarl/conf/algorithm/`，注册入口在 `benchmarl/algorithms/__init__.py`。

### 1.2 通用算法参数

所有 PPO 系（MAPPO / IPPO）共享以下核心参数：

```yaml
share_param_critic: True       # 是否共享 Critic 参数
clip_epsilon: 0.2              # PPO 裁剪 epsilon
entropy_coef: 0.0              # 熵系数
critic_coef: 1.0               # Critic 损失系数
loss_critic_type: "l2"         # Critic 损失类型
lmbda: 0.9                     # GAE lambda
use_tanh_normal: True          # 是否使用 tanh 正态分布
```

### 1.3 IPPO-Hebbian（两阶段：PPO + CMA-ES）

IPPO-Hebbian 把前两层 MLP 用 PPO 训练，最后一层用赫布规则在线更新，并通过 CMA-ES 优化赫布 ABCD 参数。

**网络架构**：
```
输入 (obs_dim) → MLP1 → MLP2 → Hebbian输出层 → 动作分布
                                    ↑
                   ABCD 参数通过 CMA-ES 优化
                   W 在执行过程中按 ABCD 规则在线更新
```

**赫布学习层（Hebbian Layer）权重更新规则**：
$$\Delta w_{ij} = A \cdot x_i \cdot y_j + B \cdot x_i + C \cdot y_j + D$$
其中 $x_i$ 是前层神经元激活，$y_j$ 是输出神经元激活，A、B、C、D 是每个连接的四个学习参数。

**训练阶段**：
1. **Phase 1**：用 PPO 训练前两层 MLP，Hebbian 层固定（W 保持初始值）。
2. **Phase 2**：冻结 MLP 层，用 CMA-ES 优化 Hebbian 层的 ABCD 参数。

### 1.4 CmaesFullHebbian（Full HNN + CMA-ES）

**网络结构**（3 层赫布层）：
```
输入 → HanLayer(hidden) → HanLayer(hidden) → HanLayer(output)
```

每一层权重 W 都按赫布规则在线更新；ABCD 参数全部由 CMA-ES 优化（单阶段训练）。碰撞惩罚可选 `navigation_avoidance` 模式。

### 1.5 CmaesHan（HAN：Hebbian Attractor Network + CMA-ES）

HAN 是相对 Full HNN 的三机制强化版本。详细机制见 [§3.3](#33-hanmodel-赫布吸引子网络)。该算法用 CMA-ES 优化 HAN 的 ABCD 参数（单阶段训练，ABCD 单调收敛即可）。

**核心参数**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--hidden-size` | 18 | 单隐层宽度 |
| `--window-size` | 10 | 滑动窗口长度 M |
| `--f-nn` | 4 | 推理频率（每 f_nn 步 = 1 步 forward） |
| `--f-hebb` | 1 | 权重更新频率（每 f_hebb 步 = 1 次 update） |
| `--lr-hebb` | 0.01 | 赫布学习率 η |
| `--cmaes-gens` | 50 | 进化代数 |
| `--pop-size` | 30 | 种群规模 |
| `--sigma0` | 0.5 | 初始步长 |
| `--n-eval-episodes` | 3 | 每个 ABCD 候选的评估 episode 数 |

**Fitness 模式**（`--fitness-mode`）：

| 模式 | 公式 | 适用场景 |
|------|------|----------|
| `navigation` | `episode_reward` | 直接最大化环境奖励 |
| `navigation_avoidance` | `episode_reward - λ × collision_count` | 简单加障碍物碰撞惩罚 |
| `navigation_v2` | `3·progress + 5·success + 1·(-final_dist)` | 推荐默认；分项奖励 |
| `navigation_avoidance_v2` | `3·progress + 5·success + 1·(-final_dist) - λ · mean(agent_collision_ratios)` | **新增**：含智能体间避碰 |
| `flocking_global` | `(1/T)·Σ(Cg+S+Ag) + M`，见 [§ 5.4.4.1](#5441-flocking_global--ramos-2019-全局-flocking) | Ramos 2019 Global flocking;梯度弱,推荐 baseline |
| `flocking_orbit` | `(1/T)·Σ(At+Dt+0.5Cg+0.5S)`，见 [§ 5.4.4.2](#5442-flocking_orbit--绕-target-飞推荐) | **推荐**:绕 target 飞,强梯度,适合 HAN 学习验证 |

`navigation_avoidance_v2` 模式新增参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--safety-distance` | 0.15 | 智能体间距离 < 该值即视为碰撞 |
| `--neighbor-radius` | 0.5 | 邻居感知半径（只考虑该半径内的邻居） |
| `--collision-penalty-weight` | 2.0 | 智能体间碰撞权重 λ |

> **世界尺寸参考**：navigation_static_dynamic_obs 任务的 world 是 `[-1.0, +1.0] × [-1.0, +1.0]`（即 2.0×2.0），智能体半径 0.1，3 个智能体。
> 推荐参数：`--safety-distance 0.25`（约 2.5× agent 半径）、`--neighbor-radius 0.5`（约 25% 地图宽）。

---

## 2. 任务与环境

### 2.1 支持的环境

| 环境 | 安装命令 | 任务数 | 向量化 |
|------|----------|--------|--------|
| **VMAS** | `pip install vmas` | 27 | Yes |
| **SMACv2** | 见 [安装指南](README.md#smacv2) | 15 | No |
| **PettingZoo** | `pip install "pettingzoo[all]"` | 10 | No |
| **MeltingPot** | `pip install dm-meltingpot` | 49 | No |
| **MAgent2** | `pip install git+https://github.com/Farama-Foundation/MAgent2` | 1 | No |

### 2.2 VMAS 任务列表

| 类别 | 任务名 |
|------|--------|
| 平衡类 | `balance`, `wheel` |
| 导航类 | `navigation`, `navigation_obs`, `navigation_dynamic_obs`, `navigation_static_dynamic_obs`, `discovery`, `flocking`, `wind_flocking` |
| 通道类 | `passage`, `joint_passage`, `joint_passage_size`, `ball_passage` |
| 对抗类 | `simple_tag`, `simple_push`, `simple_adversary` |
| 合作类 | `simple_spread`, `simple_reference`, `simple_speaker_listener` |
| 传输类 | `transport`, `reverse_transport`, `football` |
| 其他 | `buzz_wire`, `dropout`, `sampling`, `dispersion`, `give_way`, `multi_give_way`, `ball_trajectory` |

### 2.3 navigation_static_dynamic_obs 任务详解

本任务基于 `navigation` 扩展，支持可配置的静态 + 动态障碍物，适合测试算法在动态、时变环境中的避障与适应能力。

#### 与其他导航任务的对比

| 特性 | `navigation` | `navigation_obs` | `navigation_dynamic_obs` | `navigation_static_dynamic_obs` |
|------|--------------|------------------|------------------------|---------------------------------|
| 障碍物 | 无 | 静态 | 动态 | 静态 + 动态 |
| 障碍物运动模式 | — | — | 直线往复 / 圆周 / 随机 | 同左 |
| 智能体动力学 | 全向 | 全向 / 差速 / 自行车 | 全向 | 全向 |
| 任务目标 | 导航 | 避开静态障碍 | 避开动态障碍 | 避开静态 + 动态障碍 |

#### 配置文件位置

- **任务配置**：`benchmarl/conf/task/vmas/navigation_static_dynamic_obs.yaml`
- **TaskConfig**：`benchmarl/environments/vmas/navigation_static_dynamic_obs.py`
- **VMAS 场景**：`vmas/scenarios/navigation_static_dynamic_obs.py`（在 pip 包内）

#### 配置参数说明

```yaml
# 智能体配置
n_agents: 3                 # 智能体数量
agent_radius: 0.1           # 智能体半径

# 障碍物配置
n_static_obstacles: 2       # 静态障碍物数量
n_dynamic_obstacles: 0      # 动态障碍物数量
static_obstacle_radius: 0.15
dynamic_obstacle_radius: 0.1
obstacle_modes:             # 动态障碍物运动模式
  - linear
  - circular
  - random

# 动态运动参数
obstacle_speed: 0.02
obstacle_linear_range: 0.5
obstacle_circular_radius: 0.3
obstacle_random_noise: 0.1

# 世界
world_spawning_x: 1.0      # spawn 范围 [-1, 1]
world_spawning_y: 1.0

# 奖励
agent_collision_penalty: -2.0
obstacle_collision_penalty: -1.0
final_reward: 0.01
shared_rew: False
```

> **关键事实**：世界边界 `[-1, +1] × [-1, +1]`（2.0×2.0），agent 半径 0.1。在使用 `navigation_avoidance_v2` 时推荐 `--safety-distance 0.25`、`--neighbor-radius 0.5`。

---

## 3. 模型

### 3.1 标准模型

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| **MLP** | 多层感知机 | 默认选择 |
| **Hebbian** | 单层赫布学习层（PPO 训练 ABCD） | 在线自适应 |
| **FullHebbian** | 全赫布多层网络 | 端到端赫布 |
| **HanModel** | 赫布吸引子网络（HAN） | 端到端赫布 + 严格边界 |
| **GRU** | 门控循环单元 | 时序依赖 |
| **LSTM** | 长短期记忆网络 | 时序依赖 |
| **GNN** | 图神经网络 | 多智能体通信 |
| **CNN** | 卷积神经网络 | 视觉输入 |
| **Deepsets** | 深集合网络 | 置换不变性 |

> 模型文件位于 `benchmarl/models/`，注册入口在 `benchmarl/models/__init__.py`。

### 3.2 Hebbian 系列对比

| 维度 | `HebbianLayer` | `FullHebbianModel` | `HanModel` |
|------|----------------|-------------------|------------|
| 层数 | 1 | 3 | 2（单隐层） |
| 隐层宽度 | 由 `out_features` 决定 | 9（默认） | 18（默认） |
| forward 是否改 W | ✅ 是 | ✅ 是 | ❌ 否 |
| 权重更新函数 | 内联 | 内联 | 独立 `update_weights()` |
| 触发频率 | 每步 | 每步 | `f_NN // f_hebb` 步可配 |
| 激活值来源 | 瞬时 | 瞬时 | 滑动窗口时间平均 |
| 权重上限 | `w_max` 软裁剪 | `w_max` 软裁剪 | **逐层硬归一化到 max=1.0** |
| 优化器 | PPO（CMA-ES 仅在 Phase 2） | CMA-ES（单阶段） | CMA-ES（单阶段） |
| 适合任务 | 简单在线适应 | 多层抽象 | 难任务（避碰、动态） |

### 3.3 HanModel：赫布吸引子网络

HAN 在传统 HNN 基础上引入三条**严格机制**：

#### 机制 1：推理与权重更新解耦
- `HanLayer.forward(x)` 只做 `output = x @ W`，**绝不修改 W**。
- 网络（`HanModel`）维护 `self.ticks` 计数器，每次 `_forward`（即每次环境步）自增 1。
- 仅当 `ticks % (f_NN // f_hebb) == 0` 时调用 `update_weights()`；否则 **W 严格保持静态**。
- 若 `f_hebb > f_NN` 或任一为 0，自动禁用更新（不崩溃）。

#### 机制 2：滑动窗口 + 时间平均
- 每层维护两个 `deque(maxlen=M)`：`pre_window`、`post_window`。
- `forward()` 只 `deque.append`，**不计算任何 ΔW**。
- `update_weights()` 内部取 `stack(window).mean(dim=0)` 得 $\overline{x}_{pre}$ / $\overline{x}_{post}$，代入广义 ABCD：
  $$\Delta w_{ij} = \eta \cdot (a_{ij} \cdot \overline{x}_{pre,j} \cdot \overline{x}_{post,i} + b_{ij} \cdot \overline{x}_{pre,j} + c_{ij} \cdot \overline{x}_{post,i} + d_{ij})$$
- 更新完成后 **清空 deque**，确保下一 M 步是全新窗口。
- **绝对禁止使用当前步的瞬时值**做权重更新。

#### 机制 3：逐层硬归一化
```python
max_abs = W_new.abs().max()
if max_abs.item() > 0.0:
    W_new = W_new / max_abs   # 每层独立除以自己的 max|W|
```
- 归一化后，**每层权重矩阵的绝对值最大值严格等于 1.0**。
- 替代 HNN 原先的 Oja / Weight Decay，既防发散又防萎缩。

#### HanModel 关键 API

| 方法 | 作用 |
|------|------|
| `HanModel.ticks` | 当前环境步数（自上次 `reset_all_weights()` 起） |
| `HanModel._update_interval` | 触发间隔 = `f_NN // f_hebb`；禁用时为 `None` |
| `HanModel.get_all_han_layers()` | 返回所有 `HanLayer` |
| `HanModel.get_abcd_vector()` | 把所有层 ABCD 展平成一个向量 |
| `HanModel.set_abcd_from_vector(v)` | 从向量设置 ABCD |
| `HanModel.reset_all_weights()` | 重置 W、清空所有 deque、`ticks=0` |
| `HanLayer.update_weights()` | 用窗口时间平均计算 ΔW 并应用；清空 deque |

#### 端到端使用示例

```python
import torch
from torchrl.data import Composite, Unbounded
from benchmarl.models.han import HanConfig

cfg = HanConfig(
    hidden_size=18, lr_hebb=0.01, weight_init=1.0,
    window_size=10, f_nn=4, f_hebb=1,
    activation_class=torch.nn.Tanh,
)
model = cfg.get_model(
    input_spec=..., output_spec=...,
    n_agents=2, share_params=True, device="cpu",
)
# 推 5 步：内部自动 ticks += 1，并在 ticks%4==0 时触发 update_weights()
for _ in range(5):
    td = ...
    model(td)
```

---

## 4. 实验配置

所有 PPO 系算法的实验参数统一在 `benchmarl/conf/experiment/` 下。核心通用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_n_frames` | 3,000,000 | 最大训练帧数 |
| `max_n_iters` | null | 最大迭代次数 |
| `lr` | 0.00005 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `evaluation_interval` | 120,000 | 评估间隔（帧） |
| `evaluation_episodes` | 10 | 评估 episodes 数 |
| `checkpoint_interval` | 0 | 保存检查点间隔 |

### 4.1 On-policy 参数（MAPPO, IPPO, IPPO-Hebbian）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `on_policy_collected_frames_per_batch` | 6000 | 每次收集的帧数 |
| `on_policy_n_envs_per_worker` | 10 | 环境数量 |
| `on_policy_n_minibatch_iters` | 45 | 每个 batch 的训练轮次 |
| `on_policy_minibatch_size` | 400 | 小批量大小 |

### 4.2 Off-policy 参数（SAC, DDPG, QMIX 等）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `off_policy_collected_frames_per_batch` | 6000 | 每次收集的帧数 |
| `off_policy_n_envs_per_worker` | 10 | 环境数量 |
| `off_policy_n_optimizer_steps` | 1000 | 优化器步数 |
| `off_policy_train_batch_size` | 128 | 训练批量大小 |
| `off_policy_memory_size` | 1,000,000 | Replay buffer 大小 |

---

## 5. 快捷运行命令

### 5.1 标准 PPO 算法（Hydra 入口）

```bash
# MAPPO + Navigation
python benchmarl/run.py algorithm=mappo task=vmas/navigation

# IPPO + Navigation
python benchmarl/run.py algorithm=ippo task=vmas/navigation

# QMIX + SMACv2
python benchmarl/run.py algorithm=qmix task=smacv2/terran_5_vs_6

# 多任务对比
python benchmarl/run.py algorithm=mappo task=vmas/balance,vmas/navigation

# 多算法 × 多任务 × 多 seed 笛卡尔积
python benchmarl/run.py \
    algorithm=mappo,qmix,masac \
    task=vmas/balance,vmas/navigation \
    seed=0,1,2

# 使用不同模型
python benchmarl/run.py algorithm=mappo task=vmas/navigation model=gnn
python benchmarl/run.py algorithm=mappo task=vmas/navigation model=lstm
```

### 5.2 IPPO-Hebbian（两阶段训练）

```bash
# 默认配置
python examples/running/run_ippo_hebbian.py

# 动态障碍物版
python examples/running/run_ippo_hebbian_dynamic_obs.py
```

### 5.3 CmaesFullHebbian（CMA-ES 优化 Full HNN）

```bash
# 默认训练
python examples/running/run_cmaes_hebbian.py \
  --task navigation_static_dynamic_obs \
  --fitness-mode navigation_v2 \
  --cmaes-gens 50 --pop-size 30

# 仅评估
python examples/running/run_cmaes_hebbian.py \
  --evaluate-only \
  --experiment-path outputs/<你的实验文件夹> \
  --fitness-mode navigation_v2 \
  --n-final-eval 20
```

### 5.4 CmaesHan（CMA-ES 优化 HAN）

#### 5.4.1 默认训练

```bash
python examples/running/run_cmaes_han.py \
  --task navigation_static_dynamic_obs \
  --fitness-mode navigation_v2 \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --lr-hebb 0.01 \
  --cmaes-gens 50 --pop-size 30 --sigma0 0.5 \
  --n-eval-episodes 3 \
  --n-final-eval 10 \
  --max-video-frames 400
```

#### 5.4.2 含智能体间避碰（navigation_avoidance_v2）

```bash
# 无环境障碍物 + 智能体间避碰
python examples/running/run_cmaes_han.py \
  --task navigation_static_dynamic_obs \
  --fitness-mode navigation_avoidance_v2 \
  --n-static-obstacles 0 \
  --n-dynamic-obstacles 0 \
  --safety-distance 0.25 \
  --neighbor-radius 0.5 \
  --collision-penalty-weight 2.0 \
  --cmaes-gens 50 --pop-size 30
```

> **世界尺寸参考**：`navigation_static_dynamic_obs` 的 world 是 `[-1, +1] × [-1, +1]`（2.0×2.0），智能体半径 0.1，3 个智能体。
> 推荐：`--safety-distance 0.25`（2.5× agent 半径）、`--neighbor-radius 0.5`（25% 地图宽）、`--collision-penalty-weight 2.0`。

#### 5.4.3 仅评估

```bash
python examples/running/run_cmaes_han.py \
  --evaluate-only \
  --experiment-path outputs/<你的实验文件夹> \
  --fitness-mode navigation_v2 \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --n-final-eval 20 \
  --max-video-frames 400
```

> ⚠️ 评估时 `--hidden-size`、`--window-size`、`--f-nn`、`--f-hebb` 必须与训练时一致，否则 `policy_state.pt` 形状或 deque 长度会失配。
> 评估命令也需带上 `--task <与训练时相同>`。例如训练时 `--task flocking`，评估时同样要 `--task flocking`，否则 test_env 会构造不同的 scenario，`policy_state.pt` 输入维度可能失配。

#### 5.4.4 Flocking 任务

`flocking` 任务基于 VMAS 内置的 flocking scenario（`vmas/scenarios/flocking.py`）：4 个 agent 在 `[-1,1]×[-1,1]` 平面里做 holonomic 2D 运动，5 个红色静态障碍物，1 个由 `cos(t/30), sin(t/30)` 驱动的绿色 target。环境本身的奖励是 shaping 形式（与 flocking 任务目标不一致），CMA-ES 通过下面两种 fitness mode 之一重新定义目标函数，**完全脱离 RL 奖励**。

> **关键实现细节（两种 mode 都用到）**：VMAS flocking 的 agent 默认是 Holonomic，`state.rot` 不会被动力学自动更新（永远为 0）。`CmaesHanOptimizer._run_one_episode` 每步后用 `atan2(vel_y, vel_x)` patch `state.rot`，再据此计算朝向角。这是一个**仅读取 + 局部副作用**的小修改，不改 scenario，holonomic 动作空间不变。
> **世界尺寸参考**：VMAS flocking 的 `x_dim = y_dim = 1.0`，世界是 `[-1, 1] × [-1, 1]`，智能体最大可能位移 ≈ `√2 ≈ 1.414`，target 初始位置 `(0, -1)`。

##### 5.4.4.1 `flocking_global` —— Ramos 2019 全局 flocking

**fitness 公式（Ramos et al. 2019, Global setup, Eq.11）**：

```
Fg = (1/T) · Σ_t [ Cg(t) + S(t) + Ag(t) ] + M
```

| 项 | 含义 | 实现 | 范围 |
|----|------|------|------|
| `Ag(t)` | 全局对齐 | `| (1/N)·Σ exp(j·θᵢ) |`，θ 由 `atan2(vel_y, vel_x)` 得到 | [0, 1] |
| `Cg(t)` | 全局内聚 | `1 / 连通分量数`，基于 `neighbor_radius` 无向图 | (0, 1] |
| `S(t)` | 分离 | `1 - (碰撞 agent 比例)`，判定 `dist < safety_distance` | [0, 1] |
| `M` | 移动奖励 | `min(平均位移 / movement_target_displacement, 1)` | [0, 1] |

`Fg` 取值范围 `[0, 4]`：4 表示理想 flock（全局对齐 + 单连通分量 + 无碰撞 + 移动达目标），0 表示完全散乱。

**已知问题**：`flocking_global` 对"散向边界(乱飞)"和"群体同向飞"都给 3.0+ 分数，**梯度弱**，CMA-ES 难以区分"真 flocking"和"自由飘飞"（已实测：5 代 CMA-ES 几乎不进步）。适合作为 baseline / 论文复现，不推荐作为 HAN 学习能力验证的首选。

**训练命令**：

```bash
python examples/running/run_cmaes_han.py \
  --task flocking \
  --fitness-mode flocking_global \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --lr-hebb 0.01 \
  --neighbor-radius 0.5 \
  --safety-distance 0.15 \
  --movement-target-displacement 1.0 \
  --cmaes-gens 60 --pop-size 30 --sigma0 0.5 \
  --n-eval-episodes 3 \
  --n-final-eval 10 \
  --max-video-frames 400
```

默认 `neighbor_radius=0.5`、`safety_distance=0.15`、`movement-target-displacement=1.0` 在该世界尺寸下是合理默认，可直接用。

##### 5.4.4.2 `flocking_orbit` —— 绕 target 飞(推荐)

**设计动机**：`flocking_global` 不能区分"自由飘飞"和"绕 target 转圈"。`flocking_orbit` 引入**径向切线对齐**和**距离带**,把"围 target 旋转"直接编码进 fitness,让 CMA-ES 有清晰梯度。

**fitness 公式**：

```
F_orbit = (1/T) · Σ_t [ At(t) + Dt(t) + 0.5·Cg(t) + 0.5·S(t) ]
```

| 项 | 含义 | 实现 | 范围 |
|----|------|------|------|
| `At(t)` | 径向切线对齐 | 每个 agent 的速度方向 vs (target→agent) CCW 90° 切线;`At = mean( (dot+1)/2 )` | [0, 1] |
| `Dt(t)` | 距 target 距离带 | `exp(-(d - r*)² / 2σ²)`,r*=`orbit_radius`,σ=`orbit_radius_tolerance`,下限 `dt_floor` | [0, 1] |
| `Cg(t)` | 全局连通(同 flocking_global) | (0, 1] |
| `S(t)` | 分离(同 flocking_global) | [0, 1] |

`F_orbit` 取值范围 `[0, 4]`。

**关键差异**(`flocking_orbit` vs `flocking_global`):

| 行为 | `flocking_global` | `flocking_orbit` |
|------|-------------------|------------------|
| 散向边界(乱飞) | ~3.0(被误奖) | **~1.7**(被惩罚) |
| 群体同向飞(不绕圈) | ~3.5(高分) | ~2.0(中分) |
| 绕 target 转圈 | ~3.5 | **2.6+(高分)** |
| **梯度跨度** | 0.3(弱) | **1.0+(强)** |

**推荐参数**(适配 VMAS flocking 世界,target 在 `(0,-1)`,agent-to-target 距离典型 1~2):
- `--orbit-radius 0.7`(agent 距 target 期望距离)
- `--orbit-radius-tolerance 0.3`(高斯 σ,宽松区间 `[0.4, 1.0]`)
- `--dt-floor 0.1`(Dt 下限,防止 agent 飘远时信号归零)
- `--sigma0 0.3`(从 0.5 改 0.3,1584 维 ABCD 空间更细搜索)

**训练命令**：

```bash
python examples/running/run_cmaes_han.py \
  --task flocking \
  --fitness-mode flocking_orbit \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --lr-hebb 0.01 \
  --neighbor-radius 0.5 \
  --safety-distance 0.15 \
  --orbit-radius 0.7 \
  --orbit-radius-tolerance 0.3 \
  --dt-floor 0.1 \
  --cmaes-gens 30 --pop-size 20 --sigma0 0.3 \
  --n-eval-episodes 2 \
  --n-final-eval 10 \
  --max-video-frames 400 --fps 20
```

**实测 baseline**(5 代烟雾测试, `pop=12, gens=5`):
```
Gen 1: best=1.48, mean=1.26
Gen 3: best=1.71, mean=1.28
Gen 5: best=1.65, mean=1.49   ← 仍然在爬升,未收敛
```
- 训练结束时 `mean_fitness ≈ 1.7`(乱飞 baseline)
- 完整 30 代目标 `mean_fitness ≥ 2.0`,收敛 60+ 代目标 `≥ 2.4`

**仅评估(flocking_orbit)**：

```bash
python examples/running/run_cmaes_han.py \
  --evaluate-only \
  --experiment-path outputs/<你的 flocking 实验文件夹> \
  --task flocking \
  --fitness-mode flocking_orbit \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \
  --neighbor-radius 0.5 --safety-distance 0.15 \
  --n-final-eval 10 --max-video-frames 800 --fps 20
```

##### 5.4.4.3 自定义 target 运动

VMAS flocking 的 target 由 `vmas/scenarios/flocking.py` 的 `action_script_creator()` 驱动,默认走 `cos(t/30), sin(t/30)` 圆周。要自定义 target 运动,用 fork 脚本 `examples/running/run_cmaes_han_flocking_custom.py`,它通过 monkey-patch `vmas.scenarios.load` 在每次 load flocking scenario 后把 `Scenario.action_script_creator` 替换成我们定义的版本。**改 target 运动只需编辑脚本顶部 `custom_target_action_script` 函数体**。

**关键技术点**(实现细节,如不需要改实现可跳过):
- `vmas/__init__.py` 把 `scenarios` 重赋值为 sorted list,遮蔽子包引用。patch 必须用 `importlib.import_module("vmas.scenarios")` 拿到真子包,才有 `load` 方法。
- VMAS 每次 `make_env(scenario="flocking")` 都用 `importlib` 重新 exec 一次 scenario 文件,所以必须 wrap `vmas.scenarios.load` 函数而不是只 patch 单个 import 副本。
- VMAS 调用链:`make_world` 里 `action_script=self.action_script_creator()`(self = Scenario),返回的闭包被存到 `agent._action_script`,每步由 `agent.action_callback` 调成 `agent._action_script(agent, world)`。所以 `custom_target_action_script` 的 `agent` 参数实际上是 target agent(不是 Scenario),Scenario 实例通过 closure 传进来。

**常用 target 运动模板**(替换 `custom_target_action_script` 函数里 `# === EDIT BELOW ===` 那段):

```python
# 默认: 圆周(VMAS 原版)
u_x = torch.cos(t)
u_y = torch.sin(t)

# 静止
u_x = torch.zeros_like(t)
u_y = torch.zeros_like(t)

# X 轴来回往复(正弦)
u_x = torch.sin(t)
u_y = torch.zeros_like(t)

# 直线匀速 +X
u_x = torch.ones_like(t)
u_y = torch.zeros_like(t)

# 椭圆(更慢的 Y 振荡)
u_x = torch.cos(t)
u_y = 0.5 * torch.sin(t)

# 更快角速度(×3)
u_x = torch.cos(t * 3.0)
u_y = torch.sin(t * 3.0)

# 8 字轨迹
u_x = torch.sin(t)
u_y = torch.sin(t * 2.0)
```

> **物理注意**:VMAS flocking target 默认是 Holonomic dynamics,`u` 是加速度(不是速度)。`u_x=1, u_y=0` 会让 target 缓慢向 +X 加速,400 步累计位移比想象的少。如果想让 target 走出**精确的圆/直线**,需要在公式里乘 `dt` 补偿,或者直接修改 `vmas/scenarios/flocking.py` 把 target 换成 Kinematic dynamics。

**训练命令**(用 custom 脚本):

```bash
/home/zhaozeming/miniconda3/envs/benchmarl/bin/python \
  examples/running/run_cmaes_han_flocking_custom.py \
  --task flocking \
  --fitness-mode flocking_orbit \
  --hidden-size 18 \
  --window-size 10 --f-nn 4 --f-hebb 1 \
  --lr-hebb 0.01 \
  --orbit-radius 0.7 --orbit-radius-tolerance 0.3 --dt-floor 0.1 \
  --neighbor-radius 0.5 --safety-distance 0.15 \
  --cmaes-gens 30 --pop-size 20 --sigma0 0.3 \
  --n-eval-episodes 2 \
  --n-final-eval 10 --max-video-frames 400 --fps 20
```

> **重要**:评估时 target 运动脚本**必须与训练时一致**,否则 fitness 不可比(同样的 ABCD 在不同 target 轨迹下分数会变)。

##### 5.4.4.4 输出与可观察量(两种 mode 共用)

- 训练中终端打印 `best` 和 `mean` fitness(每代一次)。
- `--n-final-eval` 个 episode 完成后会打印 `Mean fitness` 与 `Mean reward`(注:reward 是 VMAS 原始 shaping reward,与 fitness mode 不一致,二者解耦)。
- `outputs/<实验>/han_results/results.json` 里有 `best_fitness`(为 fitness 标量,范围 [0, 4])、`fitness_mode` 等元信息。
- `outputs/<实验>/videos_han/eval_han_<ep>.mp4` 是带渲染的 rollout 视频,可视检查是否涌现 flocking / 绕圈 行为。

### 5.5 修改训练参数（Hydra）

```bash
# 最大训练帧数
python benchmarl/run.py algorithm=mappo task=vmas/navigation max_n_frames=5_000_000

# 学习率
python benchmarl/run.py algorithm=mappo task=vmas/navigation lr=0.001

# 评估间隔
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation_interval=100_000

# 设备
python benchmarl/run.py algorithm=mappo task=vmas/navigation sampling_device=cpu train_device=cuda

# 禁用视频记录
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation.save_video=False
```

### 5.6 评估 / 恢复训练

```bash
# 评估已训练模型
python benchmarl/evaluate.py <checkpoint_path>

# 恢复中断的训练
python benchmarl/resume.py <checkpoint_path>
```

### 5.7 输出位置

训练结果保存在 `outputs/<experiment_name>/`：

| 文件/目录 | 内容 |
|-----------|------|
| `checkpoints/` | 模型检查点 |
| `logs/` | 训练日志 |
| `wandb/` | Wandb 日志（如启用） |
| `*.json` | marl-eval 格式结果 |
| `han_results/`（HAN 训练）| ABCD 参数 + policy state |
| `videos_han/`（HAN 评估） | 评估视频 |

---

## 6. 常见问题

### 6.1 torchvision 版本兼容

如果遇到 `AttributeError: module 'torchvision.io' has no attribute 'write_video'`，禁用视频记录：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation.save_video=False
```

### 6.2 CUDA 版本问题

如果看到 CUDA 版本警告但仍能运行，强制使用 CPU：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation sampling_device=cpu train_device=cpu
```

### 6.3 Wandb 离线模式

网络不稳定时使用离线模式：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation loggers=[csv] wandb_extra_kwargs.mode=offline
```

### 6.4 视频过短

1. **增加环境步数**：
   ```bash
   python benchmarl/run.py algorithm=mappo task=vmas/navigation task.max_steps=200
   ```
   `max_steps` 在 `benchmarl/conf/task/vmas/navigation.yaml` 默认 100。

2. **降低帧率**：
   ```bash
   python benchmarl/run.py algorithm=mappo task=vmas/navigation wandb_extra_kwargs.video_fps=10
   ```

3. **两者结合**：
   ```bash
   python benchmarl/run.py algorithm=mappo task=vmas/navigation \
     task.max_steps=200 wandb_extra_kwargs.video_fps=10
   ```

### 6.5 IPPO-Hebbian 与 run.py 的关系

`run.py`（Hydra 入口）不支持 IPPO-Hebbian，因为它需要自定义网络配置（MLP + Hebbian 层组合）。必须用独立脚本：

```bash
python examples/running/run_ippo_hebbian.py
```

### 6.6 navigation 任务无障碍物

从代码可见，VMAS 的 `navigation` scenario 默认没有障碍物。如需障碍物，可：

- **方案 1（不推荐）**：直接修改 vmas 包内 `navigation.py`；
- **方案 2（推荐）**：使用自定义任务 `navigation_static_dynamic_obs`（详见 §2.3），通过 CLI 调整 `n_static_obstacles` / `n_dynamic_obstacles`；
- **方案 3**：参考 `examples/extending/task/` 创建自己的 navigation 任务。

### 6.7 CMA-ES 训练的 HAN 评估时参数必须一致

评估时 `--hidden-size`、`--window-size`、`--f-nn`、`--f-hebb` 必须与训练时一致，否则 `policy_state.pt` 的形状对不上，或滑动窗口与触发周期失配。

### 6.8 训练智能体跑出地图

`navigation_static_dynamic_obs` 的世界边界是 `[-1, +1] × [-1, +1]`，默认 ABCD=0 初始化时未训练的网络可能产生极端动作使 agent 飞出地图。这是训练初期的正常现象；CMA-ES 通过 fitness 选择会逐步淘汰这些候选。

### 6.9 inter-agent 碰撞检测依赖绝对位置

HAN 优化器的 inter-agent 碰撞检测通过 VMAS core env 拿 `agent.state.pos`（绝对位置），**不是**用 `obs[:2]`，因为 `obs[:2] = agent.pos - agent.goal`（每个 agent goal 不同，差值不能抵消 goal 偏移）。
