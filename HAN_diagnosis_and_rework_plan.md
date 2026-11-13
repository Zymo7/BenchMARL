# HAN 避障失败诊断与改造计划

> 分支：`HAN`（基线 commit `3dac9c6`）
> 方法：代码审计 + 对训练好的模型做真实 rollout 插桩测量。
> 文中每条结论都有实测数据支撑；复现脚本见 §7。

---

## 0. 一句话结论

**当前 HAN 在避障任务上"表现不好"，根因不是网络容量不足，也不是"需要更多的环境交互"，
而是三个可测量的实现缺陷叠加：**

1. **训练目标坏了** —— `success` 语义把 horizon 截断当成"到达目标"，
   导致 fitness 实际上**不测避障**，随机权重网络的得分比训练好的网络还高；
2. **Hebbian 更新退化成固定矩阵** —— D 项（常数漂移源）占 |ΔW| 的 **80.6%**，
   真正体现输入-输出关联的 A 项只占 **1.3%**，
   W 的不动点实测等于 `normalize(-D)`，**与环境数据几乎无关**；
3. **多智能体统计被平均抹掉** —— 每个 agent 的差异在更新前就被平均，
   且 per-agent 层堆叠存在 shape bug。

因此 HAN 目前**并没有在与环境交互**，它做的是一次性离线矩阵搜索然后冻住。
这与观察完全吻合：flocking（obs 即全部信息）表现好，避障（需要交互）表现差。

---

## 1. 核心发现 A：`success` 语义 bug（fitness 几乎没在测避障）

### 1.1 机制

`benchmarl/algorithms/cmaes_han_optimizer.py:1344-1350`：

```python
done_t = td.get(("next", "done"))
...
if done:
    success = True
```

但 VMAS 的 `VmasEnv(max_steps=800)` 在**到达 horizon 时也会置 `done=True`**。
`torchrl/envs/libs/vmas.py:130-134` 明确写着：

> If `max_steps` is specified, the scenario is also terminated
> (and the `"terminated"` flag is set) whenever this horizon is reached.
> Unlike gym's `TimeLimit` transform ..., this argument will **not** set
> the `"truncated"` entry in the tensordict.

链路已逐环确认：
`task.yaml: max_steps=800` → `benchmarl/environments/vmas/common.py:26` 传 `**config` →
`VmasEnv(max_steps=800)` → horizon 到期 `done=True`。

### 1.2 实测

跟踪 `done` 与真实目标距离（目标半径 0.10）：

```
step 798: done=False goal_dist=1.0455  <-- 到达目标? False
step 799: done=True  goal_dist=1.0455  <-- 到达目标? False
step 800: done=True  goal_dist=1.0455
first done at step 799 (horizon=800)
```

**agent 离目标还有 1.04 个单位没到，也被判为 success。**

### 1.3 后果

`cmaes_han_optimizer.py:469`：

```python
if success and not collided:
    return float(self.success_reward)   # +5.0
```

`success` 恒为 True ⇒

- 失败分支**只能靠 `collided=True` 进入**；
- `nav_timeout_penalty` 是**永远走不到的死代码**（`cmaes_han_optimizer.py:484`）；
- fitness 实际只测了一件事："这局有没有碰到障碍"。

真值表：

| success | collided | at_horizon | fitness |
|---|---|---|---|
| True | False | True | **+5.00（平台）** |
| True | False | False | **+5.00（平台）** |
| True | True | 任意 | −final·d − penalty·ratio |
| False | 任意 | True | −final·d − penalty·ratio − timeout（**不可达**） |

### 1.4 决定性证据：任务不区分好坏策略

同一份初始权重、逐 episode 配对（每个 episode 三种变体面对**完全相同的世界布局**）：

| 变体 | 平均 fitness | 触到 +5 平台 |
|---|---|---|
| 训练好的 ABCD（`edab186b`） | **+1.896** | 8/12 |
| 把 D 项清零 | **+4.519** | 11/12 |
| ABCD 全零（随机权重） | **+3.565** | 9/12 |

**随机权重网络 > 训练好的网络。**

在当前 fitness 下，CMA-ES 优化的方向与"学会避障"没有任何关系 ——
它甚至在破坏策略。这也解释了 `edab186b`（best_fitness=5.0）为什么不是好结果：
5.0 只是撞上了 +5 平台，不代表避障成功。

### 1.5 修法

`success` 必须由**目标的到达**定义，而非 `done`：

```python
reached = final_goal_dist < goal_radius          # 用 simulator 状态
timeout = (step >= max_steps) and not reached
```

并让 fitness 对"到达距离"连续敏感，而不是只有 0/5 两个平台。

---

## 2. 核心发现 B：W 的不动点几乎完全由 D 决定

### 2.1 ABCD 更新

`benchmarl/models/han.py:125-130`：

```
ΔW = A·(x̄_pre ⊗ x̄_post) + B·x̄_pre + C·x̄_post + D
```

### 2.2 项贡献实测（训练好的 `edab186b`，真实 rollout 插桩）

```
mean |term|:  A(outer)=0.00645  B(pre)=0.05089  C(post)=0.04112  D=0.40833
share:       A=1.3%            B=10.0%         C=8.1%          D=80.6%
```

D 项（常数项 = 纯随机漂移源）占 80.6%；真正体现输入-输出关联的 A 项只占 **1.3%**。

**量纲原因**：Hebbian 外积 `x̄_pre ⊗ x̄_post` 元素量级约 0.20
（激活经 tanh 后 `|x̄_pre| ≈ 0.44`），而 D 元素量级约 0.41 ——
**D 单项就压过 A 项 25 倍**。

### 2.3 决定性验证：不动点 = `normalize(-D)`

D-only 动力学的解析不动点是 `normalize(-D)`
（`W ← (W + lr·D)/max|W|` 是收缩到 `-D` 方向的仿射迭代）。实测：

```
cos(normalize(-D), D驱动不动点) = -0.9999999     ← 解析结论验证
cos(W_real_final, normalize(-D)) = -0.9800       ← 真实 rollout 验证
cos(W_real_final, D驱动不动点)   =  0.9314       ← 数据只贡献 ~7% 偏离
```

对两层分别验证（真实 rollout 结束时的 W）：

```
layer0: cos(W_final, normalize(-D1)) = -0.9800
layer1: cos(W_final, normalize(-D2)) = -0.9818
```

### 2.4 含义

**网络的有效权重矩阵在训练结束后是一个固定的、由 CMA-ES 决定的矩阵，
与"环境里障碍怎么分布"没有任何关系。**

HAN 在这个任务上根本没有在与环境交互 —— 它做的是一次性离线矩阵搜索，
然后把结果冻住。

> 对照：`han_dynamic_advantage_task_design.md` 的 PROBE 1 证明
> Hebbian 项确实有"重塑权重"的能力（`lr_hebb=0.01` 时恢复 135~247 步）。
> 能力是有的，**但 D 项把它彻底压过去了**。

---

## 3. 其他已确认的实现缺陷

### 3.1 `weight_init` 是死参数

`han.py:52` 存了 `weight_init`，但 `han.py:63` 永远用
`torch.randn(in_features, out_features) / sqrt(in_features)` 初始化 W。

实测：`weight_init=0.1` 与 `weight_init=99.0` 得到的 W **完全相同**。

### 3.2 `window_size=10` 实际只用了 4 步

模型层每 `f_nn // f_hebb = 4` 步清空一次窗口（`han.py:144`），
所以滑动窗口**永远不会超过 4 个元素**。

```
window len just before update at tick 4 = 4      (参数写的是 10)
```

→ 文档里的 `window_size` 与真实行为不符；**时间分辨率比预期粗 2.5 倍**。
这与 `han_dynamic_advantage_task_design.md` §P4 的窗口匹配分析直接相关 ——
该分析假设窗口真的能积累 10 步。

### 3.3 多智能体时，每个 agent 的统计被平均抹掉

`han.py:92` 用 `x.mean(dim=所有非特征维)` 压缩成**一个**向量：

```
recorded pre_vec: tensor([0.8015, 0.6788, 0.5921, 0.6089, 0.8037])
agent0 alone    : tensor([5., 5., 5., 5., 5.])
```

**3 个 agent 的差异在 Hebbian 更新前就被平均掉了。**
`navigation_static_dynamic_obs`（n_agents=3）正是这种情况 ——
所有 agent 共享一个 W、共享一份被平均的统计量。

（单 agent 任务无此问题，所以 `navigation_obs_avoidance` 相对好一些。）

### 3.4 per-agent 层堆叠的 shape bug

`han.py:291-297` 在 `share_params=False` 时对每个 agent 跑一遍完整层栈再 stack。
实测得到**错误 shape**：

```
stacked over N agent stacks -> (1, 5, 5, 4)     (expected (1, 5, 4))
```

当前默认 `share_policy_params: True`（`base_experiment.yaml:15`）所以没暴露，
但一旦改成不共享参数就会崩。

### 3.5 `max-abs` 归一化让一个元素决定整层的增益

`han.py:137-139` 把整层 `max|W|` 拉到 1.0。实测：

```
global rescale factor applied to all entries: 0.872
```

→ 矩阵里**最大的那一个元素决定其余所有元素的尺度**。
归一化本身保持 shape（cos = 1.0），但任何更新都会同时改变 shape 和 scale，
而 CMA-ES 在 plastic 模式下**搜索空间只有 ABCD，无法控制 W 的尺度**。

### 3.6 部分"避障"实验根本没有环境障碍

从各实验的 `config.pkl` 解出实际配置：

| 实验 | n_static | n_dynamic | n_agents | 备注 |
|---|---|---|---|---|
| `navigation_static_dynamic_obs__2f9903f6` | **0** | **0** | 3 | **无任何环境障碍** |
| `navigation_static_dynamic_obs__34409b50` | **0** | **0** | 3 | **无任何环境障碍** |
| `navigation_static_dynamic_obs__19267bd6` | 2 | 0 | 3 | |
| `navigation_static_dynamic_obs__623852fa` | 2 | 0 | 3 | |

`2f9903f6` / `34409b50` 只有 inter-agent 碰撞，
**不能作为"环境避障能力"的证据**。

### 3.7 多智能体导航 obs 缺速度

`navigation_static_dynamic_obs` 的 obs =
`[pos-goal(2), 静态障碍(2×N), 动态障碍(2×N), lidar(12)]` ——
**没有速度**。而 `navigation_obs_avoidance` 有 `vel(2)`。

速度对避障是必需的（无法做制动/绕行预测）。

---

## 4. 改造建议（按优先级）

### 建议 0（前置）：修 fitness + 建立"可测性"门槛

**在动网络之前必须先做**，否则改造效果无法测量。

1. **修 `success` 语义**（§1.5），并让 fitness 对到达距离连续敏感。
2. **验证任务可测性**：用 `--static-mode`（训练一个真静态网络）做对照。
   若 static 分数接近 plastic，说明任务测不出东西，不要继续训 plastic。
   （`moving_target_adaptation.md` §4.1 / §6.1 已确立这个方法论。）

### 建议 1：让 Hebbian 更新数据驱动主导，去掉 D 的支配地位

**最核心的一条。** 三种做法，侵入性递增：

**(a) 最小改动 —— 约束 D 的量纲**
把 D 改成逐元素衰减项，或给 D 一个小的固定缩放。
D 的作用本应是提供"漂移方向"（PROBE 1 里让 W 主动遗忘的机制），
但它不该压过数据。

**(b) 加入 Oja 稳定项（推荐）**
ABCD 里缺了经典 Hebbian 的稳定器 `W·post²`。
这正是代码要用 max-abs 归一化这种粗暴替代的原因。补上后：

```
ΔW = A·(x̄_pre ⊗ x̄_post) + B·x̄_pre + C·x̄_post + D − η·W·(x̄_post ⊗ x̄_post)
```

Oja 项让 W 沿**当前激活主方向**增长、沿正交方向衰减 ——
天然实现 PROBE 1 的"重塑/遗忘"，而且**不再需要 D 来提供漂移**。
可把 D 置零，让 CMA-ES 去优化 A/B/C/η。

**(c) 加衰减（两时间尺度）**
```
W ← (1−γ)·W + lr·ΔW
```
γ 提供显式遗忘，让 D 变得可选。γ 是标量，搜索空间增加极小。

### 建议 2：`max-abs` 归一化 → 按行/按列归一化

```python
# 现在：整层一个数决定所有元素的尺度
new_W = new_W / new_W.abs().max()

# 改成：每个输出神经元独立归一化
new_W = new_W / new_W.norm(dim=1, keepdim=True).clamp_min(1e-8)
```

实测差异：

```
max-abs:  [1.0,   0.100, 0.100, 0.100, ...]        ← 1.0 支配全层
row-norm: [0.990, 0.099, 0.099, 0.577, ...]        ← 每个输出单元独立增益
```

→ 每个输出单元能独立调整增益，不会出现"一个幸运元素绑架整层"。

### 建议 3：多智能体 —— 保留 per-agent 统计 + 修 shape bug

**(a) 统计量按 agent 保留**
把 `han.py:92` 的 `x.mean(dim=所有非特征维)` 改成**按 agent 轴保留**
（保留 `(n_agents, in_features)`），让 `ΔW` 用**每个 agent 自己的**外积
再跨 agent/时间平均。这样 agent 差异才能进入 Hebbian 更新。

**(b) 修 per-agent 层堆叠的 shape bug**（§3.4）。

**(c) 给多智能体导航任务补 `vel`**（§3.7）。

### 建议 4：把 reward 引入更新（reward-modulated Hebbian）

当前 Hebbian 只是**相关**（correlation），不是**目标导向**。经典 R-STDP：

```
ΔW = A·(x̄_pre ⊗ x̄_post) + β·r·(x_pre ⊗ x_post)
```

**这一步是让 HAN "与环境交互"的关键** —— 当前实现里环境对 W 的影响是零。

可行性已验证：`td.get(("next","agents","reward"))` 在 rollout 中可取到
（上一步的 reward，shape `(10,1,1)`）。
但 policy 在 `env.step` **之前**调用，需要：

- 在 `HanModel` 里缓存上一步 reward，或
- 把 reward 作为额外 state 喂进 policy（改 `HanModel._forward` 的 in_keys）

推荐前者：给 `HanModel` 加 `self._last_reward` 成员，
在 optimizer 的 rollout 循环里 `env.step` 之后回填。改动很小，
但把更新从"无监督相关"变成"有监督相关"。

### 建议 5：更新频率与窗口 —— 让 `window_size` 名副其实

当前 `f_nn=4, f_hebb=1` ⇒ 每 4 步更新、窗口只用 4 步（§3.2）。建议：

- 窗口**跨更新周期不清空**（改成 EMA），让 `window_size` 真正生效，且对噪声更鲁棒；
- 或显式解耦：`window_size` = 统计平滑窗口，`update_interval` = 更新频率。
  当前两者被 `f_nn//f_hebb` 隐式绑定，是**语义混淆**。

EMA 实现简单，且天然提供遗忘：
```
x̄ ← (1−α)·x̄ + α·x_t        # α = 1/window_size
```

若保留窗口清空语义，则至少把 `window_size` 设为 ≥ 4×update_interval。

### 建议 6：让 CMA-ES 能控制 W 的初值

plastic 模式下 W 每次运行都从 `randn/sqrt(in)` 开始（`han.py:63`），
**CMA-ES 完全无法控制它** —— 最终行为仍取决于一个随机初始矩阵。

两个选择：

- **搜索空间加上 W_init**（W 只有 140 维，可接受）；
- 或者，做完建议 1 之后，W 的不动点由 ABCD 唯一确定（不再是随机的）——
  **这本身就是建议 1 的额外好处**。

顺带：`weight_init` 参数应真正生效（§3.1）。

---

## 5. 实施顺序

| 阶段 | 内容 | 目的 | 状态 |
|---|---|---|---|
| **0** | 修 `success` 语义；`static-mode` 对照验证任务可测 | 建立可信度量 | **已完成** |
| **1** | D 项约束 + Oja 稳定项 + 归一化改造（建议 1b + 2） | 让 W 的不动点由数据决定 | **已完成** |
| **2** | reward-modulated 更新（建议 4） | 让环境真正影响 W | 待办 |
| **3** | per-agent 统计 + 补 vel（建议 3） | 多智能体导航前提 | 部分完成（shape bug 已修） |
| **4** | EMA 窗口 / 更新频率解耦（建议 5） | 时间分辨率 | 待办 |
| **5** | W_init 进入搜索空间（建议 6） | 消除随机种子依赖 | 待办（`weight_init` 已生效） |

**验证方式**：每一步都用 `--static-mode` 作对照，
报告 **plastic vs static** 在同一 task 配置、同一 seed 下的差值。

`moving_target_adaptation.md` §9.3 已在 evader 场景用这个方法测出
plastic 决定性胜出（稳态 +0.64，S 0.996 vs 0.189）。
**同样的方法应该先在避障任务上跑一遍** —— 预期在当前实现下会看到
plastic ≈ static 甚至更差，这本身就是对上述诊断的检验。

---

## 5.1 已落地的改动（阶段 0 + 1）

### 阶段 0 —— fitness 语义

`benchmarl/algorithms/cmaes_han_optimizer.py`：

- `success` 不再取自 `done`，改由**目标距离**判定
  （用场景自身的 `goal.shape.radius`，与 VMAS 的 `on_goal` 定义一致）；
- 新增 `reached` / `timed_out` 两个显式字段，一并放进 `stats`；
- `nav_timeout_penalty` 从"死代码"变成可达分支。

实测（随机权重，修复前全部 `success=True` 拿 +5 平台）：

| | 修复前 | 修复后 |
|---|---|---|
| ep0 | success=True, +5.00 | reached=False, timed_out=True, −3.17 |
| ep1 | success=True, +5.00 | reached=False, timed_out=True, −2.59 |
| … | … | … |

`run_cmaes_han_navigation_obs_avoidance.py` 新增 `--static-mode`
（此前该脚本没有对照组入口），并在 evaluate-only 路径下正确加载 `w_params.npy`。

### 阶段 1 —— Hebbian 机制

`benchmarl/models/han.py`：

| 改动 | 参数 | 默认 | 作用 |
|---|---|---|---|
| 逐样本外积 | `stats_mode` | `per_sample` | 修正 `E[xy^T] ≠ E[x]E[y]^T` |
| 激活 RMS 归一 | `normalize_stats` | `True` | 让外积无量纲，可与 D 竞争 |
| Oja 稳定项 | `lr_oja` | `0.0`（可选） | `none` 归一化下唯一的稳定器 |
| 漂移项开关 | `d_scale` | `0.0` | D 项归零 → 更新纯数据驱动 |
| 归一化方式 | `normalization` | `spectral` | 约束层增益但保留单元间相对增益 |
| `weight_init` 生效 | — | — | 之前是死参数 |
| per-agent 堆叠 shape | — | — | 修 `(B,N,N,out)` → `(B,N,out)` |

**为什么归一化选 `spectral` 而不是 per-column**：
per-column（`column_l2` / `column_maxabs`）会把所有输出单元拉到**相同**增益，
抹掉网络想学的单元间相对增益；而 `spectral`（除以最大奇异值）
只约束整层增益，保留相对结构。

**实测效果**（`tests/test_han_mechanism.py`，11/11 通过）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| W 的不动点 vs `normalize(-D)` 的 cos | 0.98 | 已解耦（D 归零） |
| 不同随机初值的收敛 W 之间 cos | 0.88 | **0.034** |
| \|ΔW\| 中 A 项占比 | 1.3% | **74.6%** |
| 不同障碍数（3 vs 8）的收敛 W 之间 cos | 0.916 | **0.705** |

最后一行是关键指标：**收敛后的 W 现在随环境改变**，
即 HAN 真的在与环境交互，而不是在解冻一个与环境无关的矩阵。

**回归测试**：`tests/test_han_mechanism.py`（11 项，独立脚本，
本环境无 pytest，按仓库既有约定写成可直接运行的脚本）。
`tests/test_hgn.py` 的 max|W| 断言已同步更新为新不变量
（谱范数 == 1）。

**PROBE 1 仍然成立**：`examples/running/han_mechanism_probe.py` 实测
输入分布突变后 W 仍会主动重塑（|W[0,new_dim]| +0.1105，静态网络 +0.0000），
说明"快速权重重塑"这一 HAN 核心能力没有被改动破坏。

---

## 6. 预期与风险

**预期**：阶段 0+1 完成后，HAN 的 W 不动点不再由 D 单独决定，
`A` 项占比应从 1.3% 上升到可观的量级，
`cos(W_final, normalize(-D))` 应从 0.98 显著下降。
这是**机制层面的可验证指标**，比 fitness 更直接。

**风险**：

- 修 `success` 语义后，fitness 数值会**大幅下降**（旧的 +5 平台消失）。
  这不是回归，是之前的高分本来就没有意义。
- 归一化方式改变会改变所有已有实验的数值，
  旧实验结果**不可与新结果直接比较**，需要重跑基线。
- `d_scale=0` 后网络不再有任何"漂移"来源。若某个任务确实需要
  常数偏置，CMA-ES 仍可通过优化 `A/B/C` 的常数分量部分覆盖，
  但这需要重训验证。

---

## 7. 复现

诊断脚本（本次分析临时编写，未入库）：

| 脚本 | 验证内容 |
|---|---|
| `probe2.py` | 窗口实际长度、D 项支配、max-abs 归一化行为 |
| `probe6.py` | 多智能体统计被平均抹掉 |
| `probe16/20/21.py` | 配对 episode 的 ABCD 消融 |
| `probe24/25.py` | W 不动点 = `normalize(-D)`（合成 + 真实 rollout） |
| `probe17/18/19.py` | `success` 在 horizon 被误触发 |
| `e2e2.py` | 修复前后收敛 W 对环境的依赖性对比 |

**已入库的回归测试**：

```bash
python tests/test_han_mechanism.py    # 11 项 HAN 机制回归测试
python tests/test_hgn.py               # HGN 冒烟测试（断言已同步）
```

**验证脚本**：

```bash
python examples/running/han_mechanism_probe.py   # PROBE 1/2/3
```


---

## 8. 相关文件

| 文件 | 说明 |
|---|---|
| `benchmarl/models/han.py` | HanLayer / HanModel，待改造主体 |
| `benchmarl/algorithms/cmaes_han_optimizer.py` | fitness 与 rollout 循环，`success` 语义所在 |
| `examples/running/han_mechanism_probe.py` | 已有机制探针（PROBE 1/2/3） |
| `han_dynamic_advantage_task_design.md` | 任务设计原则 P1-P4 |
| `moving_target_adaptation.md` | 静态对照方法论（§4.1/§6.1/§9.3） |
| `benchmarl/models/hgn.py` | HGN，复用 HanLayer，改造时需同步考虑 |
