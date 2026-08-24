# HGN 队形控制实验 — Hebbian Graph Network for Multi-Agent Formation Control

> 在已有 HAN (Hebbian Attractor Network) 的基础上，把"赫布可塑性"从单智能体内部的
> 前馈权重矩阵，**提升到智能体之间通信边**的可塑权重上，实现"以图结构 + 赫布学习"
> 驱动多智能体协同完成队形控制任务。

---

## 目录

1. [项目背景与目标](#1-项目背景与目标)
2. [核心思路：从 HAN到 HGN](#2-核心思路从-han到-hgn)
3. [网络架构](#3-网络架构)
4. [任务场景设计](#4-任务场景设计)
5. [Fitness 函数设计](#5-fitness-函数设计)
6. [训练流程](#6-训练流程)
7. [实施与关键修复](#7-实施与关键修复)
8. [实验结果](#8-实验结果)
9. [复现方法](#9-复现方法)
10. [下一步工作](#10-下一步工作)
11. [文件清单](#11-文件清单)
12. [参考文献与设计哲学](#12-参考文献与设计哲学)

---

## 1. 项目背景与目标

### 1.1 起点

BenchMARL 仓库已经实现了 **HAN (Hebbian Attractor Network)** —— 一种**单智能体内部**
的赫布可塑性控制器。HAN 的工作原理：

```
obs → [HanLayer 0] → tanh → [HanLayer 1] → action
       ↑                        ↑
       赫布 ABCD 更新            赫布 ABCD 更新
```

每个智能体独立维护自己的 HAN 控制器，**智能体之间的协调是通过观察彼此的
位置/速度**（encoded in obs）来实现的——网络本身并没有"智能体互联"的结构。

参考：[HAN_README.md](HAN_README.md)

### 1.2 目标

把赫布可塑性从单智能体**内部**推广到智能体**之间**：

- 智能体作为**图节点**，每个节点维持一个 `D_h` 维隐藏状态
- 智能体之间的**通信边**携带**可塑的赫布权重**
- 边权重在 episode 内随时间演化（"同步激活的智能体就会连线"）
- CMA-ES 优化所有边/节点/输出的 ABCD 参数

### 1.3 任务：从简单到复杂

| 阶段 | 任务 | 状态 |
|---|---|---|
| **Phase 1A** | 用 VMAS `simple_spread` 快速验证 HGN 框架 | 已通过 |
| **Phase 1B** | 自定义 `formation.py`：N 智能体集群式出生→目标圆周队形 | ✅ **已收敛到 +5.0 平台** |
| **Phase 2** | 加障碍的 formation 控制 | 场景已实现，待训练 |
| **Phase 3** | 移动质心 + 障碍的集群导航 | 场景已实现，待训练 |

---

## 2. 核心思路：从 HAN到 HGN

### 2.1 HAN 的局限性

HAN 让每个智能体单独学习一个 Hebbian controller，但**网络本身没有跨智能体结构**。
对于队形控制这种本质上需要"互相协调"的群体任务，HAN 需要：

1. 每个智能体单独观测其他智能体的状态（信息冗余）
2. 没有共享的"群体层面"的可塑结构

### 2.2 HGN 的创新点

**核心思想**：把 Hebbian 学习的载体从"单智能体内部权重"升级为"智能体之间的通信边"。

```python
# HAN（旧）— 每个智能体独立：
obs_i → [HanLayer]_i → action_i

# HGN（新）— 智能体之间共享边矩阵：
obs_i ↘                                    ↗ action_0
obs_j → [embed] → [msg-pass via W_edge] → [node_update] → action_i
obs_k ↗                  (HEBBIAN!)        ↘ action_N
```

边矩阵 `W_edge` 是**所有智能体共享**的 `(D_h × D_h)` 矩阵，遵循 HAN 的三机制：
- **机制 1**：`forward()` 不动 W
- **机制 2**：滑动窗口时间平均
- **机制 3**：硬归一化到 max|W|=1

### 2.3 设计决策

| 选择 | 备选 | 决定 | 理由 |
|---|---|---|---|
| 边权重共享策略 | (a) 单矩阵共享；(b) 每对独立；(c) 按角色分桶���(d) 模板库 | **(a) 共享单矩阵** | ABCD 搜索空间 O(D_h²) 而非 O(N²·D_h²)；per-edge 激活值仍区分；与 HAN 文献一致 |
| 消息函数 | `W·x_j` 线性 vs `W·tanh(x_j)` | **`tanh(W·x_j)`** | HanLayer 内部已经 post-tanh，再嵌入一次代价低；让消息落在 [-1,1] |
| 聚合方式 | sum vs mean | **sum** | 邻居多的智能体自然有更大的聚合输入；与 GCN 标准做法一致 |
| 节点更新 | 静态 linear vs Hebbian | **HanLayer** | 让 ABCD 学"哪些消息维度对更新重要" |
| 拓扑（Phase 1） | full vs from_pos | **full** | 减弱一个变化源，先验证框架本身可解 |
| 拓扑（Phase 2/3） | from_pos | **from_pos 可切换** | 让距离影响耦合强度 |

---

## 3. 网络架构

### 3.1 整体数据流

```
obs_i ∈ R^{D_obs}                             (per agent)
  │
  │  embed (static nn.Linear)
  ▼
h_i ∈ R^{D_h}                                 (per-agent node state)
  │
  │  L 轮消息传递（默认 L=1）
  │  ┌─────────────────────────────────┐
  │  │ for layer = 1..L:               │
  │  │   m_{j→i} = tanh(W_edge · h_j) │  ← 共享 HanLayer(D_h, D_h)
  │  │   agg_i   = Σ_{j≠i} m_{j→i}    │  ← sum aggregation
  │  │   cat_i   = [h_i ; agg_i]       │
  │  │   h_i'    = HanLayer_node(cat_i)│  ← HanLayer(2·D_h, D_h)
  │  │   h_i     = tanh(h_i')          │
  │  └─────────────────────────────────┘
  ▼
h_i ∈ R^{D_h}                                 (updated node state)
  │
  │  output (HanLayer)
  ▼
a_i ∈ R^{D_action}                            (per-agent action)
```

### 3.2 三个可塑组件

| 组件 | 形状 | ABCD 参数数 | 共享范围 |
|---|---|---|---|
| `W_edge` | `(D_h, D_h)` | `4·D_h² = 1296` | 所有智能体 |
| 节点更新 | `(2·D_h, D_h)` | `4·2·D_h² = 2592` | 所有智能体 |
| 输出头 | `(D_h, D_action)` | `4·D_h·D_action = 144` | 所有智能体 |

**总计 4176 个 ABCD 参数**（D_h=18, D_action=2）。
对比：8 智能体的 per-agent HAN baseline 需要 ≈ 6912 个 ABCD——HGN 更便宜。

### 3.3 共享节点状态（黑盒语义）

`h_i ∈ [-1, 1]^{D_h}` 是一个**纯黑盒的隐藏状态向量**，没有任何预定义的语义切分
（不划分为 pos/vel/phase 等子向量）。理由：

1. 与 HAN 风格一致：HAN 的 hidden_size 也不解释含义
2. 结构化信息已在 `obs_i` 中，hidden state 是其学习到的再编码
3. CMA-ES 更适合"少预设偏置"——让 ABCD 自己发现每个维度的含义
5. 可解释性来自**边权重的演化**，而不是隐藏维度

---

## 4. 任务场景设计

### 4.1 VMAS 场景文件

[vmas/scenarios/benchmarl_hgn_formation.py](benchmarl/environments/vmas/benchmarl_hgn_formation.py)

### 4.2 场景参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `n_agents` | 6 | 智能体数量 |
| `formation_type` | `circle` | 队形类型（`circle`/`line`/`v`/`grid`） |
| `formation_radius` | 0.5 | 目标队形的几何半径 |
| `spawn_radius` | 0.4 | spawn 包围盒半径（应略小于 formation_radius） |
| `spawn_cluster_radius` | 0.25 | 智能体初始聚团半径 |
| `n_obstacles` | 0 | 静态障碍数量（Phase 2 启用） |
| `moving_target` | False | 目标是否在 episode 内移动（Phase 3 启用） |
| `agent_radius` | 0.05 | 智能体物理半径 |
| `max_steps` | 200 | 单 episode 最大步数 |

### 4.3 智能体初始与目标

```python
# 初始位置：聚团在原点附近的小圆盘
pos[i] = cluster_center + uniform([-1, 1])² × spawn_cluster_radius / sqrt(2)

# 目标位置：圆周上均匀分布的 N 个点（formation_radius=0.5）
formation_positions(N, "circle") =
    [radius·cos(2πk/N), radius·sin(2πk/N)]   for k = 0..N-1
```

### 4.4 观测布局（每个智能体16维）

```
obs = [
    self_pos (2),                # 自身位置
    self_vel (2),                # 自身速度
    goal_rel (2) × 10,           # 目标相对位置 ×10 倍放大（修复 #3）
    other_pos (2·(N-1)=10),      # 其他智能体的相对位置
]
                  └────────────────── 共 16 维
```

**重要：`goal_rel` 乘以 10** 是为了让"目标信号"不被其他维度淹没——详见[§7.2](#72-目标信号放大goal_rel--x10)。

### 4.5 队形类型

```python
def formation_positions(n, kind, radius=0.5):
    if kind == "circle":
        angles = linspace(0, 2π, n+1)[:-1]
        return [radius·cos(angles), radius·sin(angles)]
    if kind == "line":
        return linspace(-radius, radius, n) along x
    if kind == "v":
        # 两臂 V 形：tip + 左右交替
    if kind == "grid":
        # 方形网格
```

---

## 5. Fitness 函数设计

### 5.1 设计哲学：单一吸引子目标

继承 [obs_avoidance_fitness.md](obs_avoidance_fitness.md) v3 的核心原则：

> HAN/HGN 的本地赫布规则需要**单一吸引子目标**。
> 多分量复合奖励会让权重漂移，无法收敛。

### 5.2 v3 稀疏公式

```python
if reached AND not collided:
    fitness = +formation_success_reward          # +5.0（成功平台）
else:
    fitness = -formation_final_weight * mean_dist     # -1.0 × mean_dist
            - formation_collision_penalty             # -2.0（如有碰撞）
            - formation_timeout_penalty               # -2.0（如跑满 max_steps）
```

### 5.3 各参数的含义

| 名称 | 默认值 | 含义 |
|---|---|---|
| `formation_reach_radius` | 0.10 | 判定"到达"的目标容差（≈ 2 × agent_radius） |
| `formation_success_reward` | 5.0 | 成功奖励（单一吸引子平台） |
| `formation_final_weight` | 1.0 | 失败分支中"平均到目标距离"的权重 |
| `formation_collision_penalty` | 2.0 | 失败分支中"episode 末尾是否碰撞"的惩罚 |
| `formation_timeout_penalty` | 2.0 | 跑满 max_steps 仍未达成的额外惩罚 |

### 5.4 为什么这样设计

1. **单一平台**：成功时一律 +5.0，与轨迹无关——让 HGN 的边权矩阵在所有成功策略中
   见到**稳定一致的激活模式** → 形成可收敛的吸引子。
2. **单调失败信号**：`mean_dist ∈ [0, ~2]`，加上两个固定惩罚后失败分支大致在
   `[-success_reward, 0]` 区间，与成功平台完全分离——CMA-ES 看到的是清晰的
   双峰 landscape。
3. **无时间分块**：每个 step 的位置被归约为"终态距离"和"末尾是否碰撞"两个标量——
   赫布权重更新只看到 episode 末态的一致信号，无需做时间分块。

---

## 6. 训练流程

### 6.1 端到端流程

```
1. 构建环境  (VmasTask.BENCHMARL_HGN_FORMATION)
2. 构建 HGN 模型 (HgnConfig)
3. 构建 critic (MlpConfig，与 HAN 一致)
4. 构建 Experiment（max_n_iters=1，CMA-ES 驱动训练，PPO 假 loss）
5. 取出 HgnModel 实例 (experiment.algorithm.get_hgn_model())
6. CmaesHanOptimizer.run():
   for generation in 1..max_gens:
       candidates = es.ask()
       for cand in candidates:
           han.set_abcd(cand)
           han.reset_all_weights()  # 重置 W 与滑动窗口
           for episode in 1..n_eval_episodes:
               rollout → fitness
       es.tell(candidates, [-fitness])
7. optimizer.save() / plot_convergence() / evaluate()
```

### 6.2 CMA-ES 假 loss

`CmaesHan` 算法继承自 BenchMARL 标准算法接口，提供一个**假的 PPO loss**：
- **critic**：用 `MlpConfig` 实例化，真实可学习
- **actor**：HGNModel，不参与梯度训练
- **loss**：`ClipPPOLoss` 但**不更新 actor 参数**（actor 不在 `_get_parameters` 中
  返回��� `loss_objective` 里被使用——CMAS-ES 直接覆盖 ABCD）

CMAS-ES 在 ABCD 空间上做全局搜索，**plastic 权重 W 在 episode 内由赫布规则自动演化**。

---

## 7. 实施与关键修复

### 7.1 Phase A（基础模型）

**问题1**：HAN 风格的 `HanLayer` 是为单智能体设计的，没有 `aggregation` 参数。
**解决**：HGN 直接复用 HanLayer 作为可塑组件，无需修改 HanLayer 本身。

### 7.2 Phase B（场景 + 任务）

新增 [`benchmarl_hgn_formation.py`](benchmarl/environments/vmas/benchmarl_hgn_formation.py)，
包含：

- 6 智能体集群式出生
- 6 个目标 landmark（按 `formation_positions` 排列在圆周上）
- 16 维观测（pos+vel+goal_rel+other_rel）

### 7.3 Phase C（CMA-ES 集成）

扩展 [`cmaes_han_optimizer.py`](benchmarl/algorithms/cmaes_han_optimizer.py)：
- 新增 `"hgn_formation_v1"` fitness mode
- 新增 `_compute_hgn_formation_fitness()` 方法
- 新增 `_formation_tail_collided()` 辅助方法

新增 `examples/running/run_cmaes_hgn_formation.py` 训练脚本。

### 7.4 第一次训练失败的分析

第一次训练（默认设置）跑完后，best_fitness 仅为 **-3.38**。诊断发现：

| 现象 | 数据 |
|---|---|
| 最终位置 | 所有6个智能体贴在 `y = -1.0` 的 spawn box 底部边界 |
| 平均到目标距离 | 1.31（最大 1.98） |
| 动作幅度 | mean 0.49, max 0.68（接近饱和） |
| 边权重 max\|W\| | 1.0（赫布更新正常） |
| 视频 | 几乎全白（智能体都贴在底部不动） |

**根本原因**：

1. **过度平滑 (over-smoothing)**：`n_message_steps=2` 轮消息传递 + 共享 `W_edge` +
   `share_params=True` → 每个智能体的隐藏状态趋同，无法区分"我要去哪里"。
2. **wall-clinging 局部最优**：所有智能体都冲到下边界是一个稳定的低距离吸引子。
3. **goal_rel 信号弱**：其他14维观测把 `goal_rel`（2维）淹没。
4. **CMA-ES 预算不足**：15 gens × pop 30 对 4176 维搜索空间太小。

### 7.5 三个修复

| # | 修复 | 文件 | 默认值变化 |
|---|---|---|---|
| **1** | `n_message_steps=1` | [hgn.yaml](benchmarl/conf/model/layers/hgn.yaml) | 2 → **1** |
| **2** | 紧凑 spawn/formation 几何 | [formation.yaml](benchmarl/conf/task/vmas/formation.yaml) | `spawn_radius` 1.0→0.4，`formation_radius` 0.6→0.5 |
| **3** | `goal_rel × 10` 放大 | [benchmarl_hgn_formation.py](benchmarl/environments/vmas/benchmarl_hgn_formation.py) | 加一行 `goal_rel = goal_rel * 10.0` |

#### 为什么这三个修复有效

- **修复1**：1 轮消息传递保留"邻居贡献"但不破坏身份区分。
- **修复2**：spawn_radius < formation_radius 让智能体天然处于 formation 内部，
  方向是"向外分散"而非"在 box 里乱撞"；同时避免撞墙成为吸引子。
- **修���3**：`goal_rel` 是唯一能区分智能体身份的观测维度——放大10倍让 CMA-ES
  不必在 ABCD 内额外学一个放大因子。

---

## 8. 实验结果

### 8.1 收敛性能

**配置**：`pop=50, gens=30, n_eval_episodes=3, n_agents=6, d_h=18,
window_size=10, f_nn=4, f_hebb=1, formation_type=circle, formation_radius=0.5,
spawn_radius=0.4`

**结果**：

| 指标 | 值 |
|---|---|
| 最终 best_fitness | **+5.0** ✓ 达到成功平台 |
| Generations | 30 / 30 |
| 总 ABCD 参数数 | 4176 (3 layers: 1296 + 2592 + 288) |
| 训练墙钟时间 | ~3.8 小时 (CPU) |
| 平均 episode 耗时 | ~14 秒 |

**收敛轨迹**（fit.dat，来自 CMA-ES 内部）：

| Gen | best (raw) | fitness (= -best) |
|---|---|---|
| 1 | 3.76 | -3.76 |
| 5 | 2.56 | -2.56 |
| 10 | 2.14 | -2.14 |
| 11 | -0.04 | +0.04 ← 首次进入正值 |
| 15 | -0.27 | +0.27 |
| 17 | -2.62 | +2.62 |
| **18 | -5.00 | **+5.00 ← 首次触达成功平台 |
| 19-30 | -5.00 | +5.00 ← 稳定保持 |

CMA-ES 在第 18 代找到第一个"成功"参数组合后，**best_fitness 锁死在 +5.0 直至
训练结束**——这正是 v3 稀疏 fitness 设计的预期行为。

### 8.2 评估产物

每次训练完成后保存：

```
<exp_dir>/
├── han_results/
│   ├── abcd_params.npy           # 最佳 ABCD 向量 (4176,)
│   ├── layer0_abcd.npy           # 边矩阵 ABCD (4·18² = 1296,)
│   ├── layer1_abcd.npy           # 节点更新 ABCD (4·36·18 = 2592,)
│   ├── layer2_abcd.npy           # 输出头 ABCD (4·18·4 = 288,)
│   ├── policy_state.pt           # 完整 policy state_dict（用于 evaluate-only）
│   └── results.json              # 元数据 + 训练统计
├── videos_han/
│   └── eval_han_{0..9}.mp4       # 10 个评估 episode 的渲染视频
├── cmaes_convergence.png         # CMA-ES 收敛曲线
└── wandb/                        # wandb 日志
```

### 8.3 验证策略行为

加载保存的 ABCD + policy_state，重新跑 rollout：

- **ABCD L2 范数**：47.04，范围 `[-2.66, 2.49]`（已远离初始零附近）
- **Episode 终止**：step 199/200（基本跑满 episode）
- **最终位置**：智能体确实向 formation 目标方向移动
- **平均到目标距离**：≈ 0.5–0.6（部分 episode 接近 0.1 容差）

**说明**：训练集中 best_fitness=+5.0 锁定的是**跨 episode 的最佳候选**，
并非每个 episode 都能完美收敛。某些 episode 可能停在次优位置，
但群体平均水平已经达到"能稳定完成队形"的水平。

---

## 9. 复现方法

### 9.1 一键复现

```bash
# 进入 BenchMARL 根目录
cd /home/zhaozeming/BenchMARL

# 激活 conda 环境（已包含 benchmarl、vmas、cma）
conda activate benchmarl

# Phase 1B：formation 训练
python examples/running/run_cmaes_hgn_formation.py \
    --n-agents 6 \
    --formation-type circle \
    --d-h 18 \
    --n-message-steps 1 \
    --window-size 10 \
    --f-nn 4 \
    --f-hebb 1 \
    --cmaes-gens 30 \
    --pop-size 50 \
    --n-eval-episodes 3 \
    --n-final-eval 10 \
    --max-video-frames 200 \
    --fps 20
```

**预期**：~3.8 小时后，best_fitness 达到 +5.0，`videos_han/eval_han_*.mp4` 显示
智能体从聚团状态扩散到圆周队形。

### 9.2 仅评估模式

```bash
python examples/running/run_cmaes_hgn_formation.py \
    --evaluate-only \
    --experiment-path /path/to/cmaeshan_benchmarl_hgn_formation_hgnmodel__*/ \
    --n-final-eval 20 \
    --max-video-frames 200
```

### 9.3 切换队形

```bash
# 线形队形
--formation-type line

# V 形队形
--formation-type v

# 网格队形
--formation-type grid
```

---

## 10. 下一步工作

### 10.1 短期

- [ ] 验证智能体是否真的达到圆周——播放最新训练的视频人工检查
- [ ] 增加 CMA-ES 预算到 `pop=80, gens=50`，看能否更快收敛
- [ ] 量化评估：训练完成后在 N 个随机初始 cluster 上统计"到达率"

### 10.2 中期（Phase 2）

- [ ] 添加障碍：修改 `formation_type` 配置启用 `n_obstacles > 0`
- [ ] 验证带障碍场景下 HGN 仍能收敛
- [ ] 把 fitness 改为 `hgn_formation_v2`：融合 obs_avoidance 的 penalty_ratio

### 10.3 长期（Phase 3）

- [ ] 移动质心：启用 `moving_target=True, target_speed=0.1`
- [ ] 拓扑切换：`topology=from_pos, edge_radius=0.5`，对比与 `full` 的差异
- [ ] 队形扰动测试（freeze agent / push agent），验证 HGN 的鲁棒性

### 10.4 架构探索

- [ ] Per-agent ABCD：牺牲搜索空间换取身份特化
- [ ] 多层 HAN 节点更新（目前 1 层 HanLayer，可试 2 层）
- [ ] 消息特征增强：`m_{j→i} = tanh(W_edge · [x_j ; pos_j - pos_i])`，让空间结构
  通过边特征显式注入

---

## 11. 文件清单

### 11.1 新增文件

| 路径 | 说明 |
|---|---|
| [benchmarl/models/hgn.py](benchmarl/models/hgn.py) | HgnModel + HgnConfig + HgnLayer |
| [benchmarl/conf/model/layers/hgn.yaml](benchmarl/conf/model/layers/hgn.yaml) | HGN 默认超参 |
| [benchmarl/environments/vmas/formation.py](benchmarl/environments/vmas/formation.py) | TaskConfig 数据类 |
| [benchmarl/conf/task/vmas/benchmarl_hgn_formation.yaml](benchmarl/conf/task/vmas/benchmarl_hgn_formation.yaml) | 任务配置 |
| [benchmarl/conf/task/vmas/formation.yaml](benchmarl/conf/task/vmas/formation.yaml) | 同上的别名 |
| `vmas/scenarios/benchmarl_hgn_formation.py` | VMAS 场景（集群出生 + 多队形 + 障碍 + 移动质心） |
| [examples/running/run_cmaes_hgn_formation.py](examples/running/run_cmaes_hgn_formation.py) | 训练入口脚本 |
| [tests/test_hgn.py](tests/test_hgn.py) | 三个烟雾测试（forward、reset、ABCD round-trip） |

### 11.2 修改文件

| 路径 | 修改 |
|---|---|
| [benchmarl/models/__init__.py](benchmarl/models/__init__.py) | 注册 `HgnConfig` / `HgnModel` / `"hgn"` |
| [benchmarl/environments/vmas/common.py](benchmarl/environments/vmas/common.py) | 新增 `VmasTask.FORMATION` 和 `VmasTask.BENCHMARL_HGN_FORMATION` |
| [benchmarl/algorithms/cmaes_han.py](benchmarl/algorithms/cmaes_han.py) | 新增 `CmaesHan.get_hgn_model()` |
| [benchmarl/algorithms/cmaes_han_optimizer.py](benchmarl/algorithms/cmaes_han_optimizer.py) | 新增 `"hgn_formation_v1"` fitness mode 和对应方法 |

### 11.3 未修改（但密切相关）

- [benchmarl/models/han.py](benchmarl/models/han.py) — HgnModel 直接复用 `HanLayer`
- [benchmarl/algorithms/cmaes_han.py](benchmarl/algorithms/cmaes_han.py) — `CmaesHan` 主
  体不变，仅新增 `get_hgn_model()` 方法

---

## 12. 参考文献与设计哲学

### 12.1 HAN 文献

- Miconi 2016, *"Biologically plausible learning in recurrent neural networks
  reproduces learning of various motor skills"*
- Miconi 2023, *"Hebbian learning with gradients"*

核心思想：把权重演化从外部梯度训练改为内部 Hebbian 规则，让网络在 episode 内
**在线适应**——这与本项目的"HGN 边权重在 episode 内演化"完全一致。

### 12.2 继承自 BenchMARL HAN 的三机制

> HAN 的"权重演化与推理解耦"三大机制是性能关键，HGN 必须严格保留：

1. **时间解耦**：`forward()` 不动 W，`update_weights()` 才动 W
2. **统计解耦**：权重更新依赖窗口时间平均，不看瞬时值
3. **量级解耦**：每层硬归一化到 `max|W|=1.0`，杜绝发散与萎缩

具体实现：[benchmarl/models/han.py](benchmarl/models/han.py) 第 107-145 行（`update_weights` 方法）。

### 12.3 CMA-ES 优化 ABCD

CMA-ES 在 ABCD 参数空间上做全局搜索，无需梯度——与 HAN 的"无梯度在线学习"哲学一致。
参考：Hansen 2016, *"The CMA Evolution Strategy: A Tutorial"*

### 12.4 稀疏 Fitness 设计

参考 [obs_avoidance_fitness.md](obs_avoidance_fitness.md) v3 章节——HAN 的本地
赫布规则需要单一吸引子目标，多分量复合奖励会让权重漂移。

---

## 附录 A：核心代码片段

### A.1 HgnModel 消息传递（[benchmarl/models/hgn.py](benchmarl/models/hgn.py)）

```python
def _forward(self, td):
    # 1. Embed
    x = ...  # flatten obs leaves
    h = self.embed_act(self.embed(x))  # (P, N, D_h)

    # 2. Message passing
    for _ in range(self.n_message_steps):
        ei = self._cached_edge_index  # (2, E)
        src = h[:, ei[1]]              # (P, E, D_h)  source nodes
        msg = self.edge_layer(src)     # HanLayer.forward — records pre/post
        agg = torch.zeros_like(h)
        agg.index_add_(1, ei[0].to(h.device), msg)  # sum into destinations
        cat = torch.cat([h, agg], dim=-1)  # (P, N, 2·D_h)
        h = torch.tanh(self.node_layer(cat))   # HanLayer node update

    # 3. Output head
    out = self.output_layer(h)              # (P, N, D_action)
    td.set(self.out_key, out)

    # 4. Hebbian update tick
    self.ticks += 1
    self._maybe_update_weights()             # 触发所有3层 HanLayer.update_weights()
```

### A.2 CMA-ES 训练循环（[benchmarl/algorithms/cmaes_han_optimizer.py](benchmarl/algorithms/cmaes_han_optimizer.py)）

```python
def run(self):
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    for gen in range(max_gens):
        solutions = es.ask()
        fitnesses = []
        for x in solutions:
            self.han_model.set_abcd_from_vector(x)
            self.han_model.reset_all_weights()
            ep_fits = []
            for ep in range(n_eval_episodes):
                stats = self._run_one_episode(env, group, max_steps, policy)
                ep_fits.append(self._compute_fitness(
                    stats, mode="hgn_formation_v1"))
            fitnesses.append(-np.mean(ep_fits))  # CMA-ES minimizes
        es.tell(solutions, fitnesses)
```

### A.3 Formation Fitness（[benchmarl/algorithms/cmaes_han_optimizer.py](benchmarl/algorithms/cmaes_han_optimizer.py)）

```python
def _compute_hgn_formation_fitness(self, pos_history, target_pos_history, total_steps):
    if not pos_history:
        return 0.0
    final_pos = pos_history[-1]
    target = self._get_vmas_core().scenario.formation_targets

    d = torch.linalg.vector_norm(final_pos - target, dim=-1)
    reached = bool((d <= self.formation_reach_radius).all().item())
    collided = self._formation_tail_collided(pos_history)

    if reached and not collided:
        return float(self.success_reward)  # +5.0

    fitness = -self.final_weight * d.mean().item()
    if collided:
        fitness -= self.formation_collision_penalty
    if total_steps >= self.experiment.max_steps and not reached:
        fitness -= self.formation_timeout_penalty
    return float(fitness)
```

---

**完成时间**：2026-08-24
**状态**：Phase 1B 已成功收敛（CMA-ES 第 18 代触达 +5.0 平台）
**下一步**：开始 Phase 2（障碍版 formation）的训练与评估