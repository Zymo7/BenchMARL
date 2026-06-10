# HAN — Hebbian Attractor Network（赫布吸引子网络）

> 在传统纯赫布网络（HNN）基础上引入的全新机制：
> **推理与权重更新解耦 + 时间窗口平均 + 逐层硬归一化**。
>
> 本文档涵盖：网络性质、文件清单、配置项、命令行使用方法、代码级使用示例、关键实现细节与单元测试结果。

---

## 1. 背景与动机

传统 HNN 在每次 `forward()` 内部直接基于当前步瞬时激活值 $x_i, x_j$ 写入 $\Delta W$。这种实现存在三个问题：

1. **更新频率被锁定在推理频率上**，无法解耦；
2. **逐拍瞬时值噪声大**，单步相关容易被干扰；
3. **权重无硬性边界**，长 episode 下容易发散到极端值甚至 NaN。

HAN（Hebbian Attractor Network）通过三个**严格机制**解决上述问题。

---

## 2. 三大核心机制

### 机制 1：推理与权重更新解耦（独立 `update_weights()` + `ticks` 计数器）

- `HanLayer.forward(x)` 只做矩阵乘法 `x @ W`，**绝不修改 W**。
- 网络（`HanModel`）内部维护一个环境步数计数器 `self.ticks`，每次 `_forward` 调用（即每次环境步）自增 1。
- 由网络根据 `f_NN`（推理频率）和 `f_hebb`（更新频率）控制何时调用每层的 `update_weights()`：
  - 触发条件：`self.ticks % (f_NN // f_hebb) == 0`；
  - 满足时调用 `update_weights()`；不满足时 **权重矩阵严格保持静态**；
  - 若 `f_hebb > f_NN` 或任一为 0，自动禁用更新（不会除零崩溃）。

> 关键点：**forward 路径上没有任何 W 写入**，所有权重变化都来自显式调用 `update_weights()`。

### 机制 2：固定长度滑动窗口 + 时间平均

- 每一层内部维护两个 `collections.deque(maxlen=M)` 缓冲区，分别记录：
  - `pre_window`：本层输入经过 `tanh` 后的激活值（`x̄_pre` 的来源）；
  - `post_window`：本层输出经过 `tanh` 后的激活值（`x̄_post` 的来源）。
- `forward()` 内部只做 `deque.append(...)`，**不计算任何 $\Delta W$**。
- 当外部触发 `update_weights()` 时：
  1. 将两个 deque 中的 $M$ 步张量沿时间维 `stack`，得到 `(M, in_features)` 与 `(M, out_features)`；
  2. 取 `mean(dim=0)` 得到 $\overline{x}_{pre}$（长度 `in_features`）与 $\overline{x}_{post}$（长度 `out_features`）；
  3. 代入广义 ABCD 赫布公式：
     $$\Delta w_{ij} = \eta \cdot (a_{ij} \cdot \overline{x}_{pre,j} \cdot \overline{x}_{post,i} + b_{ij} \cdot \overline{x}_{pre,j} + c_{ij} \cdot \overline{x}_{post,i} + d_{ij})$$
  4. 更新完成后 **清空两个 deque**，确保下一次收集的 $M$ 步是全新的窗口。
- **绝对禁止使用当前步的瞬时值**做权重更新。

### 机制 3：逐层硬归一化（绝对值最大值严格为 1.0）

在 `update_weights()` 内部完成 $W_{new} = W_{old} + \Delta W$ 后，**立刻**执行：

```python
max_abs = W_new.abs().max()
if max_abs.item() > 0.0:
    W_new = W_new / max_abs
```

- **逐层独立**进行：每一层各自除以自己的 `max(|W|)`；
- 归一化完成后，**每一层权重矩阵的绝对值最大值严格等于 1.0**；
- 这一步替代了 HNN 原先的 Oja/Weight Decay，既防止发散又防止无限萎缩。

---

## 3. 整体架构

```
输入 obs
   ↓
[HanLayer #0]  W₀: (in_features, hidden_size)   ← 记录 pre/post 到 deque₀
   ↓  tanh
[HanLayer #1]  W₁: (hidden_size, hidden_size)   ← 记录 pre/post 到 deque₁
   ↓  tanh
[HanLayer #2]  W₂: (hidden_size, output_features) ← 记录 pre/post 到 deque₂
   ↓
动作 logits (loc, scale)
```

- **推理（inference）**：每步环境都发生，W 完全冻结。
- **权重更新（update）**：每 `f_NN // f_hebb` 步发生一次，使用滑动窗口的时间平均。

---

## 4. 文件清单

| 路径 | 角色 | 说明 |
|---|---|---|
| `benchmarl/models/han.py` | **核心** | `HanLayer`、`HanModel`、`HanConfig` |
| `benchmarl/models/__init__.py` | 注册 | 已加入 `HanModel`、`HanConfig` 与 `"han"` 名字 |
| `benchmarl/algorithms/cmaes_han.py` | 算法骨架 | `CmaesHan` + `CmaesHanConfig`（提供 CMA-ES 兼容的假 PPO loss） |
| `benchmarl/algorithms/cmaes_han_optimizer.py` | **CMA-ES 优化器** | 优化 ABCD 参数；提供 `fitness()`、`run()`、`save()`、`evaluate()`、`plot_convergence()` |
| `benchmarl/algorithms/__init__.py` | 注册 | 已加入 `CmaesHan`、`CmaesHanConfig` 与 `"cmaes_han"` 名字 |
| `benchmarl/conf/algorithm/cmaes_han.yaml` | 算法配置 | `scale_mapping`、`use_tanh_normal` |
| `benchmarl/conf/model/layers/han.yaml` | 模型配置 | `window_size`、`f_nn`、`f_hebb`、其他超参 |
| `examples/running/run_cmaes_han.py` | **命令行入口** | 与 `run_cmaes_hebbian.py` 对称；支持训练 / 评估两种模式 |

---

## 5. 配置项

### 5.1 模型配置 `HanConfig`（`benchmarl/models/han.py`）

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `hidden_size` | int | 9 | 隐层宽度 |
| `lr_hebb` | float | 0.01 | 赫布学习率 $\eta$ |
| `weight_init` | float | 1.0 | 权重初始化缩放 |
| `window_size` | int | 10 | 滑动窗口长度 $M$（机制 2） |
| `f_nn` | int | 1 | 推理频率（单位：环境步） |
| `f_hebb` | int | 1 | 权重更新频率（单位：环境步） |
| `activation_class` | nn.Module | `nn.Tanh` | 层间激活 |
| `activation_kwargs` | dict | `None` | 激活类初始化参数 |
| `num_feature_dims` | int | 1 | 展平维度数 |

> **关键约束**：必须 `0 < f_hebb <= f_NN`，否则更新被自动禁用（`update_interval = None`），不会崩溃。

### 5.2 算法配置 `CmaesHanConfig`（`benchmarl/algorithms/cmaes_han.py`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `scale_mapping` | `"biased_softplus_1.0"` | 策略分布的尺度映射 |
| `use_tanh_normal` | `True` | 是否使用 `TanhNormal` 分布 |

### 5.3 CMA-ES 相关（命令行）

| 参数 | 默认 | 含义 |
|---|---|---|
| `--fitness-mode` | `navigation_v2` | 适应度函数；可选 `navigation` / `navigation_avoidance` / `navigation_v2` |
| `--cmaes-gens` | 50 | 进化代数 |
| `--pop-size` | 30 | 种群规模 |
| `--sigma0` | 0.5 | 初始步长 |
| `--n-eval-episodes` | 3 | 每个候选 ABCD 的评估 episode 数 |

---

## 6. 命令行使用方法

### 6.1 训练模式

```bash
python examples/running/run_cmaes_han.py \
  --task navigation_static_dynamic_obs \
  --fitness-mode navigation_v2 \
  --hidden-size 9 \
  --lr-hebb 0.01 \
  --weight-init 1.0 \
  --window-size 10 \
  --f-nn 4 \
  --f-hebb 1 \
  --cmaes-gens 50 \
  --pop-size 30 \
  --sigma0 0.5 \
  --n-eval-episodes 3 \
  --n-final-eval 10 \
  --max-video-frames 400
```

### 6.2 仅评估模式

```bash
python examples/running/run_cmaes_han.py \
  --evaluate-only \
  --experiment-path outputs/<你的实验文件夹> \
  --fitness-mode navigation_v2 \
  --n-final-eval 20 \
  --max-video-frames 400
```

> 评估时使用 `--f-nn` 与 `--f-hebb` 必须与训练时一致，否则滑动窗口与触发周期会失配。

### 6.3 推荐超参（起点）

| 场景 | window_size | f_nn | f_hebb | 说明 |
|---|---|---|---|---|
| 噪声大 / 障碍多 | 10 | 4 | 1 | 默认推荐；用 4 步平均后再更新 |
| 快速变化场景 | 4 | 1 | 1 | 每步都用 4 步窗口更新 |
| 极稳定场景 | 20 | 10 | 1 | 慢更新、强平滑 |

---

## 7. 代码级使用示例

### 7.1 直接构造一个 `HanModel`

```python
import torch
from torchrl.data import Composite, Unbounded
from benchmarl.models.han import HanConfig

cfg = HanConfig(
    hidden_size=9,
    lr_hebb=0.01,
    weight_init=1.0,
    window_size=10,
    f_nn=4,
    f_hebb=1,
    activation_class=torch.nn.Tanh,
)

n_agents = 2
input_spec = Composite({"agents": Composite({"observation": Unbounded(shape=(n_agents, 3))}, shape=(n_agents,))})
output_spec = Composite({"agents": Composite({"logits": Unbounded(shape=(n_agents, 2))}, shape=(n_agents,))})
action_spec = Composite({"agents": Composite({"action": Unbounded(shape=(n_agents, 2))}, shape=(n_agents,))})

model = cfg.get_model(
    input_spec=input_spec, output_spec=output_spec,
    n_agents=n_agents, centralised=False, input_has_agent_dim=True,
    agent_group="agents", share_params=True, device="cpu",
    action_spec=action_spec,
)

# 推 5 步
for _ in range(5):
    td = ...  # 构造你的 TensorDict
    model(td)   # 内部自动 ticks += 1，并在 ticks%4==0 时触发 update_weights()
```

### 7.2 通过 CMA-ES 优化 ABCD 参数

```python
from benchmarl.algorithms.cmaes_han import CmaesHanConfig
from benchmarl.algorithms.cmaes_han_optimizer import CmaesHanOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

task = VmasTask.NAVIGATION_STATIC_DYNAMIC_OBS.get_from_yaml()
experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.max_n_iters = 1

experiment = Experiment(
    task=task,
    algorithm_config=CmaesHanConfig.get_from_yaml(),
    model_config=HanConfig(...),
    critic_model_config=MlpConfig(num_cells=[64, 64],
                                  activation_class=torch.nn.Tanh,
                                  layer_class=torch.nn.Linear),
    seed=0,
    config=experiment_config,
)
experiment._setup()

han_model = experiment.algorithm.get_han_model()
optimizer = CmaesHanOptimizer(
    experiment=experiment,
    han_model=han_model,
    fitness_mode="navigation_v2",
    pop_size=30, sigma0=0.5, max_gens=50, n_eval_episodes=3,
    device=experiment.config.train_device,
)
best_abcd = optimizer.run()
optimizer.save(output_dir=str(experiment.folder_name))
optimizer.plot_convergence(output_dir=str(experiment.folder_name))
optimizer.evaluate(output_dir=str(experiment.folder_name),
                   n_episodes=10, fps=20, max_video_frames=400)
```

### 7.3 关键 API

| 方法 | 作用 |
|---|---|
| `HanModel.ticks` | 当前环境步数（自上次 `reset_all_weights()` 起） |
| `HanModel._update_interval` | 触发间隔（`f_NN // f_hebb`），若禁用则为 `None` |
| `HanModel.get_all_han_layers()` | 返回所有 `HanLayer`（含 per-agent 情形） |
| `HanModel.get_abcd_vector()` | 把所有层 ABCD 展平成一个向量 |
| `HanModel.set_abcd_from_vector(v)` | 从向量设置 ABCD |
| `HanModel.reset_all_weights()` | 把所有 W 重置为初始值，**并清空所有 deque** 与 `ticks` |
| `HanLayer.update_weights()` | 用当前 deque 时间平均计算 $\Delta W$ 并应用；之后清空 deque |

---

## 8. 与传统 HNN 的差异对照

| 维度 | 传统 HNN（`FullHebbianModel`） | HAN（`HanModel`） |
|---|---|---|
| 推理路径是否改 W | **是**（在 forward 内更新） | **否**（forward 纯推理） |
| 权重更新函数 | 隐式在 `forward` 中 | 显式 `update_weights()` |
| 更新触发频率 | 每步 | 每 `f_NN // f_hebb` 步（可配置） |
| 激活值来源 | 当前步瞬时值 | **窗口内 M 步时间平均** |
| 激活值形式 | 原始 `x`、`output = x @ W` | 经过 `tanh` 后存入 deque |
| 权重上限 | `w_max` 软裁剪到 $\pm 1$ | **逐层硬归一化到 max=1.0** |
| 权重下限 | 可能无限萎缩（若用 decay） | 由硬归一化自然保护 |
| 滑窗/状态 | 无 | 每层独立 `deque(maxlen=M)` |

---

## 9. 关键实现细节

### 9.1 缓冲区的存储

```python
self._pre_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
self._post_window: Deque[torch.Tensor] = deque(maxlen=self.window_size)
```

存的是 **post-`tanh` 的 1 维向量**（`mean` 掉了非特征维），每个元素是 `(in_features,)` 或 `(out_features,)`。

### 9.2 时间平均的实现

```python
pre_stack = torch.stack(list(self._pre_window), dim=0)    # (M, in_features)
post_stack = torch.stack(list(self._post_window), dim=0)  # (M, out_features)
x_bar_pre = pre_stack.mean(dim=0)      # (in_features,)
x_bar_post = post_stack.mean(dim=0)     # (out_features,)
```

`deque(maxlen=M)` 自动丢弃最早的元素；当新元素追加超过 M 时，旧的自动弹出。

### 9.3 逐层硬归一化

```python
new_W = self.W.data + self.lr_hebb * delta_W
max_abs = new_W.abs().max()
if max_abs.item() > 0.0:
    new_W = new_W / max_abs
self.W.data = new_W.clone()
```

`max_abs > 0` 的守卫避免全新初始化的零矩阵触发除零。

### 9.4 触发计数器

`HanModel._forward` 的最后：

```python
self.ticks += 1
self._maybe_update_weights()
```

`_maybe_update_weights()` 检查 `ticks % _update_interval == 0`，对**所有层**调用 `update_weights()`。

> **重要**：多智能体（`input_has_agent_dim and not share_params`）情形下，每个智能体有独立的层栈与独立 deque；
> `ticks` 仍只在 `_forward` 末尾自增 1 次（不会因 per-agent 循环而多次自增）。

### 9.5 与框架的兼容性

- `HanModel` 继承自 `Model`（BenchMARL 标准基类），与 `CmaesHan` 算法、`ProbabilisticActor` 包装器、`TensorDictModuleSequential` 链路完全兼容。
- `_policies_for_loss` 内的 `get_han_model()` 用来在外部（如优化器）拿到模型实例。

---

## 10. 验证结果（已通过的单元测试）

下面这些断言都已经在仓库内的 Python 进程中实际跑过：

| 断言 | 结果 |
|---|---|
| `forward()` 不修改 W | ✅ `torch.equal(W_before, W_after) == True` |
| `forward()` 后 deque 长度 +1 | ✅ |
| `update_weights()` 后 deque 被清空 | ✅ `len(window) == 0` |
| `update_weights()` 后 `max\|W\| == 1.0` | ✅（多次循环仍稳定） |
| 滑动窗口时间平均等于解析期望 | ✅（`tanh([1,2,3,4]).mean()` 完全匹配） |
| 6 步、interval=2、3 层 → 9 次 `update_weights()` 调用 | ✅ |
| `f_hebb > f_NN` 不会崩溃，更新被禁用 | ✅ |
| `f_hebb == f_NN` 每步都更新 | ✅ |
| 端到端 50 步 episode 后 `ticks=50`、`max\|W\|=1.0` | ✅ |
| 端到端 2 代、pop=4 的 CMA-ES 跑通，返回合法 best_abcd | ✅ |

---

## 11. 训练产物

训练结束后（`optimizer.save` 写入到 `<experiment_folder>/han_results/`）：

| 文件 | 内容 |
|---|---|
| `abcd_params.npy` | 最佳 ABCD 向量（numpy） |
| `layer{i}_abcd.npy` | 各层 ABCD 分别保存 |
| `policy_state.pt` | 完整 `policy.state_dict()`（用于 `--evaluate-only`） |
| `results.json` | 元信息：fitness、层数、层形状、`window_size`、`f_nn`、`f_hebb`、`update_interval` 等 |

视频写到 `<experiment_folder>/videos_han/eval_han_{ep}.mp4`。
CMA-ES 收敛图写到 `<experiment_folder>/cmaes_convergence.png`。

---

## 12. 常见问题

**Q1. 为什么不直接用 `tanh(x) @ W` 替换为新的输出？**
A. 推理路径（动作输出）不能动；动 W 的代码必须与动 output 的代码严格分离。这是机制 1 的硬要求。

**Q2. 缓冲区用 `deque` 会不会在 `state_dict` 序列化/反序列化时丢数据？**
A. 不会。`deque` 是 Python 对象，不参与 `state_dict`；`reset_all_weights()` 会在每个新 episode 起点清空缓冲区，因此跨 episode 不需要保留内容。

**Q3. 滑动窗口填不满 M 步会怎样？**
A. 仍然按当前长度取平均。`M` 越大、噪声越平滑，但响应越慢；建议 `M ∈ [4, 20]`。

**Q4. `w_max` 软裁剪还要不要保留？**
A. 在 HAN 中已被硬归一化替代，不需要再保留 `w_max`。

**Q5. 多智能体不共享参数时，CMA-ES 优化的是几份 ABCD？**
A. 每个智能体的层栈有独立的 ABCD；`total_abcd_params` 也会变成 N 倍。在 `share_params=True`（推荐用于 CMA-ES）下，每个智能体共享同一份 ABCD，节省搜索空间。

---

## 13. 总结

HAN 用三条**互相正交**的硬性规则，把"动作推理"与"权重演化"在时间、统计、量级上完全解耦：

1. **时间解耦**：`forward` 不动 W，`update_weights` 才动 W；
2. **统计解耦**：权重更新依赖窗口时间平均，不看瞬时值；
3. **量级解耦**：每层硬归一化到 `max|W|=1.0`，杜绝发散与萎缩。

配合 CMA-ES 在 ABCD 参数空间上的全局搜索，可以得到一个完全无梯度、可解释、长 episode 稳定的可塑权重策略网络。
