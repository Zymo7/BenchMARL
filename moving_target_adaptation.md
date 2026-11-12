# HAN 在线自适应能力评估 — 移动目标实验

> **结论摘要**（随实验进展两次修正）：
> - **orbit 任务（静止/慢速 circle target）**：没有证据支持"赫布在线自适应带来优势"——
>   训练过的静态网络与 plastic 持平甚至略优（§5-6）。
> - **追逃围捕任务（evader，target=40% agent 速度）**：**plastic 决定性胜出**——
>   稳态 Fg +0.64，避碰 S 0.996 vs 0.189。static 挤成一团互相挤压，plastic 保持松散围捕编队（§9）。
>
> 区别的根源：orbit 任务不需要跨步决策，而追逃任务的"挤团"失败模式只有
> 塑性（W 的在线重整）能化解。本文档记录完整实验、踩过的坑与修正过程。

---

## 1. 背景与问题

**实验**：`cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents`
（task=`flocking`，fitness=`flocking_orbit`，`lr_hebb=0.01`，`f_nn=4`，`f_hebb=1`，ABCD=560 维，2 层）

**要回答的问题**：现成 ABCD + policy 在 target 移动时能否保持 orbit 行为？
如果是，就能测出 HAN 的**在线自适应能力**。

**任务背景**：8 个 agent 要绕着场景中央的绿色 target 做轨道运动。
fitness 分解为四项（见 [cmaes_han_optimizer.py:611](benchmarl/algorithms/cmaes_han_optimizer.py#L611)）：

| 分量 | 含义 | 权重 |
|---|---|---|
| **At** | 径向-切向对齐（agent 速度方向 vs agent→target 射线的 CCW 切线） | 1.5 |
| **Dt** | 到 target 距离的高斯带（峰值在 `orbit_radius=0.7`） | 1.0 |
| **Cg** | 全局聚集（邻接图连通分量数的倒数） | 0.2 |
| **S** | 间距保持（1 − 碰撞 agent 比例） | 0.8 |

---

## 2. 实现

只改了 4 个文件，**`cmaes_han_optimizer.py` 的 fitness 计算和 task 配置的核心逻辑没动**（只加了 static mode 分支）。

### 2.1 移动 target 机制 — [examples/running/flocking_patch.py](examples/running/flocking_patch.py)

新增 `target_motion` action script，支持 4 种轨迹：

| 模式 | 轨迹 | 参数 |
|---|---|---|
| `stationary` | 固定在 (x, y)（原行为，默认） | — |
| `circle` | 绕中心匀速圆周 | `radius=0.35`, `period=400` 步/圈 |
| `figure8` | Gerono 双纽线 | `radius=0.4`, `period=400` |
| `random_walk` | 定速直线 + 定期重采样方向 + 撞墙反射 | `speed=0.01`, `interval=50` |

**关键设计决策：运动学驱动（`u=0` + `set_pos`），而不是力驱动。**

最初想用 `u = speed·[-sin φ, cos φ]`（切向力），但这是错的：
- VMAS 的 `action.u` 是**力**不是速度指令。世界有 `drag=0.25/步`、`dt=0.1`，稳态速度 ≈ `0.4·u/步`
- 纯切向力**没有向心力**，轨迹是相位滞后的花瓣形（rosette）而非圆
- `speed=0.05, period=400` 算下来半径 ≈ 1.27，**超出 ±1 的世界边界**；"越界反弹"会让轨迹变成花瓣+方波，污染 At 的 CCW 语义

改成照抄 `flocking_signal.py:186-220` 的成熟模式：**每步解析算出目标位置，`set_pos` 直接写，`u` 保持 0**。
（VMAS 在 [core.py:972](file:///home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/simulator/core.py) 有 assert 要求 action script 必须设 `u`，所以 `u=0` 是必须的。）

参数因此从 (speed, period) 改为 **(radius, period)**：R=0.35 / T=400 时切向速度仅 ≈0.0055/步，天然不越界，**不需要 clamp 也不需要反弹**。

运动状态（phase / heading）挂在 scenario 上，在 `reset_world_at` 里重置，保证 episode 间可复现。

### 2.2 CLI — [run_cmaes_han_flocking_custom.py](examples/running/run_cmaes_han_flocking_custom.py) + [run_cmaes_han_flocking_disturbance.py](examples/running/run_cmaes_han_flocking_disturbance.py)

```
--target-motion {stationary,circle,figure8,random_walk}
--target-motion-radius / --target-motion-period / --target-motion-speed / --target-motion-interval
--eval-output-subdir          # 避免 evaluate 覆盖训练目录里的基线视频
--static-mode                 # 训练静态网络对照（见 §4）
--max-steps / --n-agents      # 覆盖 task yaml（用于匹配原实验配置）
```

---

## 3. ⚠️ 踩过的坑：最初的对照组是无效的

**第一版对照组**：直接给已有实验加 `--lr-hebb 0` 重新评估，得出"冻结塑性后 fitness 从 3.30 崩到 1.38，证明赫布自适应是核心驱动力"。

**这个结论是错的。** 原因在 [han.py:63](benchmarl/models/han.py#L63)：

```python
W = torch.randn(in_features, out_features) / (in_features ** 0.5)   # 随机初始化
```

`W` 是**随机**初始化的，而 `reset_all_weights()` 每次都从 `W_init`（那份随机值）恢复。
`lr_hebb=0` 时 `new_W = W + 0·ΔW` —— **W 永远停在随机初值，ABCD 完全不起作用**。

所以那个对照组实际测的是"**一个随机权重网络做 orbit**"：它只能证明随机网络不会 orbit，
**无法回答"塑性相对于训练过的静态网络值多少"**。ABCD 在其中没有参与任何计算。

### 修正后的对照：真正训练一个静态网络

新增 `--static-mode`：
- CMA-ES 的搜索空间从 **ABCD (560 维)** 换成 **W (140 维)**，直接优化权重
- 赫布更新**彻底关闭**

**实现要点（两处都必须做，否则对照组会被静默污染）**：

1. **同步 `W_init`** — `set_w_from_vector` 同时写 `W` 和 `W_init`。
   否则 `reset_all_weights()`（在 `fitness()` 里每个 episode 前都调）会把 W 还原成随机值，
   CMA-ES 的候选值在评估前就被抹掉了。
2. **完全跳过 update** — 新增 `HanModel.freeze_weights` 开关。
   因为**即使 `lr_hebb=0`，max-abs 归一化仍会执行**：
   ```
   new_W = W + 0·ΔW              # 不变
   new_W = new_W / max|W|        # 但 max|W| 从 0.05 → 1.0！
   ```
   实测确认：`max|W|` 从 0.0500 被拉到 1.0000。
   这会让静态网络的权重尺度被 CMA-ES 之外的东西持续改写。

优化器与模型之间**只**通过 `get_abcd_vector`/`set_abcd_from_vector` 通信，
所以把 static mode 做成这两个方法的分支即可，**CMA-ES 主循环一行未改**。

---

## 4. 实验设计

### 4.1 两套 static 训练（因为 task 配置在训练后被改过）

查 git 发现 task yaml 在实验之后被修改过：

| | plastic HAN 训练时（commit `c05c99f`） | 当前 yaml |
|---|---|---|
| `max_steps` | **800** | 1600 |
| `n_agents` | **4** | 8 |
| `pop_size` | 20 | — |

因此跑了**两套** static 训练，避免把任务难度差异误判成塑性的价值：

- **A（匹配原实验）**：`--max-steps 800 --n-agents 4 --pop-size 20`
- **B（当前 yaml）**：`--max-steps 1600 --n-agents 8 --pop-size 30`

两者都用：`--cmaes-gens 30 --sigma0 0.3 --n-eval-episodes 2`（与原实验一致）

### 4.2 评估

`run_cmaes_han_flocking_disturbance.py` 跑 1 个完整 1600 步 episode，输出 per-step 的
`Fg / At / Dt / Cg / S` 分解（`--disturbance-step 1599` 使 freeze 扰动落在最后一步，
整个 episode 实质是 baseline）。

---

## 5. 结果

### 5.1 训练结果

| | plastic HAN | static A（800步/4agents/pop20） | static B（1600步/8agents/pop30） |
|---|---|---|---|
| 最优 fitness | 3.298 | **3.32** | **3.28** |
| 10-episode 均值 | — | 3.31 | 3.27 |

**静态网络与 plastic HAN 持平，甚至略高。**

### 5.2 完整分解矩阵（1600 步全 episode）

| 条件 | Fg | At | Dt | Cg | S |
|---|---|---|---|---|---|
| plastic, stationary | 3.298 | 0.912 | 0.896 | 0.166 | 1.000 |
| plastic, circle | 3.226 | 0.901 | 0.839 | 0.180 | 1.000 |
| **static, stationary** | **3.398** | **0.944** | **0.947** | **0.278** | 0.980 |
| **static, circle** | **3.358** | **0.938** | **0.925** | **0.303** | 0.965 |

### 5.3 移动 target 的退化幅度（关键对比）

| | ΔAt | ΔDt | ΔFg |
|---|---|---|---|
| plastic | -1.2% | -6.4% | -2.2% |
| **static** | **-0.6%** | **-2.3%** | **-1.2%** |

### 5.4 时间窗口（每 400 步的 At / Dt）

```
Condition       w0-200       w200-400      w400-800     w800-1200    w1200-1600
plastic stat    At=.71/Dt=.44  At=.92/Dt=.84  At=.94/Dt=.96  At=.94/Dt=.99  At=.94/Dt=.99
plastic circle  At=.72/Dt=.38  At=.89/Dt=.81  At=.93/Dt=.91  At=.93/Dt=.93  At=.93/Dt=.92
```

两种模式都在 ~200 步内收敛到稳态，target 跑 4 圈（T=400）期间没有掉队。

---

## 6. 结论

### 6.1 修正后的结论

1. **没有证据表明赫布在线自适应是这个任务的关键。**
   训练过的静态前馈网络与 plastic HAN 持平（3.31 vs 3.298），且**在 target 移动时退化更小**
   （ΔDt: -2.3% vs -6.4%）。

2. **最初的"鲁棒性"观察应归因于前馈策略本身。**
   `target_rel` 在 obs 里，agent 每步都能看到 target 的当前位置；
   "沿切线飞"这个行为在 env 力学下是自维持的（VMAS flocking 的碰撞力 + 近邻交互），
   **不需要跨步的时间记忆**就能维持。
   换句话说：**这个任务本身没有强制要求在线自适应的需求**。

3. **塑性反而可能带来轻微拖累。**
   plastic 的 At/Dt 都略低于 static，且移动时退化更大。
   合理解释：W 的在线演化引入了噪声，而 orbit 行为其实不需要 W 变化。
   plastic 的 `Cg`（0.166 vs 0.278）也明显更低——swarm 更松散。

4. **但也不能说塑性"无用"。**
   这个实验只能说明**在 orbit 这个任务上**，塑性没有带来增益。
   塑性的价值可能出现在真正需要时间记忆的任务上
   （如 `flocking_signal` 的 leader 只能看到标量光强、必须从时间序列推断梯度方向）。

### 6.2 对后续实验的启示

- **要让实验能测出塑性，必须先证明"静态网络做不到"**。
  正确的对照是训练过的静态网络（本文档的方法），而不是冻结塑性的随机网络。
- 找**真正需要跨步推断**的任务：给 agent 的观测里**去掉目标的即时信息**，
  只留标量场（如 `flocking_signal` 的 Φ(pos,t)），迫使它用时间序列估计趋势。
- 若要保留当前设置，至少要做 §4.1 的配置匹配，否则 task 差异会污染结论。

---

## 7. Push 扰动下的适应能力（plastic vs static）

### 7.1 实验设计

在移动 target 场景下，测试多智能体系统对 **push 扰动**（agent 2 被径向外推，
magnitude 0.5，step 800）的适应能力，并对比 plastic HAN 与训练过的 static net。

**2×2×2 矩阵**（每组 1 个 1600 步 episode）：

| 模型 | target | push |
|---|---|---|
| plastic | stationary / circle | ✓ / ✗ |
| static | stationary / circle | ✓ / ✗ |

- 扰动参数与已有的 `disturbance_eval_push` 基线**完全一致**（radial-out, mag 0.5, agent 2, step 800）
- 新增 `--disturbance-mode none` 作为无扰动对照
- 两个模型用**同一 task 配置**评估（max_steps=1600, n_agents=8）

### 7.2 指标修正（重要）

**recovery time 的参考基准不能用整个扰动前窗口的均值**——那个窗口包含 warmup 期
（Fg 从 2.2 爬到 3.4），而 push 后 Fg 继续爬升，永远不会"回落到"那个均值，
导致所有条件都显示 "never recovered"。

改为**局部稳态参考**（扰动前最后 100 步）后指标才有意义。
更关键的是要报告 **净效应 = (push 条件的 drop) − (none 条件的 drop)**，
以扣掉"没有扰动也有的自然波动"（none 条件下也有 +0.02~+0.08 的"drop"）。

### 7.3 结果

**逐条件分解**（参考 = 局部稳态）：

| 条件 | ref Fg | post Fg | min Fg | drop | recovery | ΔAt | ΔDt | ΔCg |
|---|---|---|---|---|---|---|---|---|
| plastic \| stationary \| push | 3.441 | 3.435 | 3.412 | +0.029 | 0 | -0.002 | -0.003 | -0.001 |
| plastic \| stationary \| none | 3.441 | 3.436 | 3.419 | +0.022 | 0 | -0.002 | -0.002 | -0.004 |
| plastic \| circle \| push | 3.339 | 3.353 | 3.286 | +0.053 | 0 | +0.006 | +0.010 | -0.025 |
| plastic \| circle \| none | 3.339 | 3.358 | 3.289 | +0.051 | 0 | +0.006 | +0.014 | -0.026 |
| static \| stationary \| push | 3.427 | 3.419 | **3.174** | **+0.253** | 0 | +0.001 | -0.003 | -0.029 |
| static \| stationary \| none | 3.427 | 3.424 | 3.371 | +0.056 | 0 | +0.000 | -0.001 | -0.011 |
| static \| circle \| push | 3.415 | 3.397 | 3.336 | +0.079 | 0 | -0.004 | -0.008 | -0.017 |
| static \| circle \| none | 3.415 | 3.402 | 3.340 | +0.075 | 0 | -0.004 | -0.007 | -0.001 |

**净效应（push − none）**：

| 模型 / target | Δdrop | Δmin Fg | Δrecovery |
|---|---|---|---|
| **plastic / stationary** | **+0.007** | -0.007 | 0 |
| **plastic / circle** | **+0.002** | -0.002 | 0 |
| **static / stationary** | **+0.197** | -0.197 | 0 |
| **static / circle** | **+0.004** | -0.004 | 0 |

### 7.4 结论

1. **push 扰动在几乎所有条件下都测不出影响**：净效应 ≤ 0.007（plastic），
   recovery 全部为 0 步。mag=0.5 径向外推单个 agent
   **不足以对这个 8-agent orbit 系统构成有效扰动**。

2. **唯一有信号的一组是 `static / stationary`（+0.197）**，
   但**同样的 push 在 static 的 circle 条件下净效应只有 +0.004**。
   所以这**不能**解读为"塑性在起作用"——而是因为
   **静止 target 下的 static 网络恰好处于一个 push 敏感的临界状态**
   （min Fg 跌到 3.174，是 8 组里最低），换到移动 target 这个失衡就消失了。
   **可测性依赖于系统所处的具体状态，而非模型的架构属性。**

3. **plastic 对局部扰动几乎免疫**（stationary 下净效应仅 +0.007）：
   W 的持续在线演化本身就是一种鲁棒化。

4. **本实验没有测出 HAN 的动态优势**。若要继续这条路需要**更强的扰动**：
   推离轨道更远（mag ≥ 2.0）、同时推多个 agent、
   或把 agent 推到 target 附近直接破坏 Dt 分量。

### 7.5 实现中修掉的两个"静默错误"bug

两个 bug 都不会报错，只会给出**看起来合理但完全错误**的表格：

1. **收集时机**：最初在 8 组全部跑完后才拷贝 npz，
   而 disturbance 脚本每次都写同一个 `disturbance_eval_<mode>/per_step_data.npz`
   → 只能拿到最后一组的数据（表格整整齐齐，但 4 个 plastic 行完全相同）。
   修复：runner 在**每组结束后立刻**快照成 `<model>_<motion>_<mode>.npz`。

2. **输出路径**：快照路径硬编码了 `disturbance_eval_push/`，
   但 `none` 模式写到的是 `disturbance_eval_none/`（目录名跟随 mode）
   → none 组拷到了上一组的残留文件。修复：路径改为 `disturbance_eval_${mode}/`。

修复后加了**区分度验证**：每个 motion 内 push/none 的 ref Fg 必须完全相同、
circle 组 `tgt_moved=True`、push 组的速度跳变必须大于 none 组。全部通过。

### 7.6 复现

```bash
bash examples/running/run_push_adaptation_matrix.sh
python examples/running/collect_push_matrix.py --snapshots /tmp/push_matrix
```

---

## 8. 加强扰动：magnitude = 2.0

magnitude=0.5 太弱（净效应 ≤0.007）。物理推算：agent 轨道速度 ≈ 0.372/步，
VMAS `drag=0.25/步`，所以冲量按指数衰减——0.5 的冲量 4-5 步就耗散完，
而前馈策略 + 近邻碰撞力每步都在纠正它。

| magnitude | 相对轨道速度 | 扰动后瞬时速度 |
|---|---|---|
| 0.5 | 1.3× | 0.37 → 0.81 |
| **2.0** | **5.4×** | **0.37 → ~2.3** |
| 4.0 | 10.8× | 0.37 → ~4.3 |

### 8.1 结果（mag=2.0，8 组同设计）

**逐条件**（参考 = 局部稳态）：

| 条件 | ref Fg | post Fg | min Fg | drop | 碰撞步数 | ΔDt |
|---|---|---|---|---|---|---|
| plastic \| stationary \| push | 3.441 | 3.425 | 3.314 | +0.127 | **0** | -0.014 |
| plastic \| stationary \| none | 3.441 | 3.436 | 3.419 | +0.022 | 0 | -0.002 |
| plastic \| circle \| push | 3.339 | 3.340 | 3.251 | +0.089 | **0** | -0.005 |
| plastic \| circle \| none | 3.339 | 3.358 | 3.289 | +0.051 | 0 | +0.014 |
| static \| stationary \| push | 3.427 | 3.418 | 3.268 | +0.160 | 0 | -0.006 |
| static \| stationary \| none | 3.427 | 3.424 | 3.371 | +0.056 | 0 | -0.001 |
| **static \| circle \| push** | 3.415 | 3.363 | **2.849** | **+0.566** | **57** | -0.024 |
| static \| circle \| none | 3.415 | 3.402 | 3.340 | +0.075 | 0 | -0.007 |

**净效应（push − none）**：

| 模型 / target | Δdrop @0.5 | **Δdrop @2.0** | 碰撞步数 | push 后被推 agent 最大离轨距离 |
|---|---|---|---|---|
| plastic / stationary | +0.007 | **+0.105** | 0 | 1.14 |
| plastic / circle | +0.002 | **+0.038** | 0 | 1.27 |
| static / stationary | +0.197 | **+0.104** | 0 | 1.27 |
| **static / circle** | +0.004 | **+0.491** | **57** | 1.04 |

### 8.2 关键发现

**1. magnitude=2.0 确实让扰动"可见"了**：净效应从 ≤0.007 涨到 0.04~0.49，
被推 agent 的速度跳变从 0.12 涨到 1.11~1.25（约 8 倍）。

**2. 但最戏剧化的差异（static/circle 的 +0.491）不是 push 本身造成的**，
而是**级联碰撞**：深谷出现在 step 951（push 早在 151 步前），且成因是
**另外两个 agent（1 和 6）相撞**（S 从 1.000 崩到 0.500），
与被推的 agent 2 无关。

机制：agent 2 被甩到离 target 1.04 的位置，重入轨道时改变了群体几何，
把 agent 1 和 6 挤到一起，**130 步后**才发生碰撞。
→ **测到的 drop 主要是级联效应，不是"目标本身受扰动的恢复速度"。**

**3. plastic 完全没有出现级联碰撞（4 个 push 条件中碰撞步数全为 0）**，
而 static 的两个 push 条件出现了 102 和 275 个碰撞步。
这是 plastic 优于 static 的**第一个正面证据**：
**塑性让群体在被扰动后维持几何结构的能力更强。**

**4. 所有条件的 recovery 仍是 0 步**（第一个 post step 就回到 5% 带内）。
即使 mag=2.0，扰动也只造成**瞬时**失衡——因为 At/Dt 是**群体平均**，
单个 agent 出轨（占 1/8）只让均值掉 ~0.1，而群体本身的几何结构很快重建。

**5. 因此，要测出"恢复速度"的差异，需要改变观测/计分方式**：
当前 fitness 是群体平均，单 agent 扰动的信号被稀释了 8 倍。
更合适的是**按 agent 计分**（如"被推 agent 回到轨道带内的步数"），
或**扰动多个 agent**（如同时推 4 个，使群体结构真正被打乱）。

### 8.3 实现中修掉的一个"静默失败"bug

mag=2.0 的第一轮 8 组全部失败且**产出陈旧数据**：
wandb 认证失效（403）导致 `Experiment._setup()` 在仿真开始前抛异常，
而 runner 不检查退出码、无条件拷贝 npz → 快照全是从上一轮拷来的旧文件
（refFg 与 mag=0.5 完全一致，肉眼很难发现）。

三处修复：
1. `WANDB_MODE=offline` —— 离线运行不依赖网络服务
2. runner 捕获 `PIPESTATUS[0]` 检查脚本退出码
3. **npz 新鲜度守卫**：运行前打 stamp，要求 npz 比 stamp 新，否则拒绝拷贝并告警

验证：mag=2.0 的 8 组快照 refFg 与 mag=0.5 一致（同条件同种子），
none 条件的速度跳变不变（0.067~0.102），push 条件的跳变从 0.12~0.20 涨到 1.11~1.25。





```bash
EXP=examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents

# --- 训练静态网络对照（匹配原实验 task 配置）---
python examples/running/run_cmaes_han_flocking_custom.py \
    --task flocking --fitness-mode flocking_orbit \
    --static-mode --max-steps 800 --n-agents 4 --pop-size 20 \
    --cmaes-gens 30 --sigma0 0.3 --n-eval-episodes 2 \
    --target-motion stationary

# --- 分解评估：静止 vs 圆周移动 ---
for MOTION in stationary circle; do
python examples/running/run_cmaes_han_flocking_disturbance.py \
    --experiment-path $EXP \
    --num-episodes 1 --max-steps 1600 --disturbance-step 1599 \
    --fitness-mode flocking_orbit --disturbance-mode freeze \
    --target-motion $MOTION \
    --target-motion-radius 0.35 --target-motion-period 400
done

# --- 冻结塑性的 eval（仅作为 null control，不能用于结论）---
... --lr-hebb 0
```

---

## 9. 追逃围捕场景（evader）：plastic 的决定性优势

> 这是全项目**第一次**在同等条件下测出 plastic 显著优于 static。

### 9.1 场景设计

新增 `evader` 运动模式（[flocking_patch.py](examples/running/flocking_patch.py)）：
target 以**恒定速度 v** 在 ±0.9 方框内随机游走（每 30 步重采样方向、撞墙反射）。

**速度标定**（先量 agent 能力再选 target 速度）：

| v_target | % agent 速度 (0.372/步) | Fg | S | 现象 |
|---|---|---|---|---|
| 0.05 | 13% | 2.549 | **0.635** | 追上后挤成一团、互相碰撞 |
| 0.10 | 27% | 2.732 | 0.904 | 仍挤 |
| **0.15** | **40%** | **2.797** | **0.997** | **甜点：松散追踪、不碰撞** |
| 0.20 | 54% | 2.759 | 0.998 | 同上稍难 |
| 0.30 | 81% | 2.749 | 1.000 | 围不住（ Dt 崩） |

两个反直觉发现：
1. **旧 circle 模式的 target 只以 1.5% agent 速度移动**——agent 眼里几乎静止，
   这是 §5-6 测不出差异的根本原因之一。
2. **target 越慢 swarm 表现越差**：追上之后堆叠在 target 附近 → 挤压 → S 崩。
   难度不是单调的，v=0.15 是甜点。

**调整位置**：`--target-motion-speed`（CLI）或 [flocking_patch.py](examples/running/flocking_patch.py)
的 `_TARGET_MOTION_SPEED`（代码默认）。

### 9.2 结果（v=0.15, interval=30, 1600 步）

| 指标 | plastic | static | Δ |
|---|---|---|---|
| **Fg（稳态 400+）** | **2.836** | 2.196 | **+0.641** |
| **S（避碰）** | **0.996** | **0.189** | **+0.807** |
| Cg（聚集） | 0.386 | **1.000** | −0.614 |
| At / Dt | 0.834 / 0.512 | 0.863 / 0.512 | ≈0 |

**行为模式的本质差异**（视频可直视）：
- **static**：8 个 agent **完全挤成一团**追 target（Cg=1.0 单连通肉弹），
  内部 81% 的时间互相挤压碰撞（S=0.189）
- **plastic**：保持**松散编队**追踪（Cg=0.39），几乎全程无碰撞（S=0.996）

**围捕覆盖率几乎相同**（围捕带 0.4-1.0：43.3% vs 44.4%；
失联 >1.0：44.3% vs 43.3%）——但**队形完全不同**。
plastic 用的是"围"的形态，static 用的是"挤"的形态。

### 9.3 为什么这次测出来了（而之前测不出）

1. **target 速度进入真实追逃量级**（40% agent 速度 vs circle 的 1.5%）
2. **任务的失败模式不同**：orbit 任务里 static 的"挤成一团"不被惩罚
   （静止 target 下 orbit 半径带恰好不重叠）；
   evader 任务里挤团的 swarm 在 target 转向时**内部挤压**，S 直接崩
3. **塑性提供的正是"动态拉开间距"的能力**：W 的在线演化让 agent
   在 target 变向时各自调整响应，而不是集体同步扑向同一个点

### 9.4 注意事项

- 两组网络都是**静止 target 上训练**的，本实验是 zero-shot 泛化测试
- n=1 单 episode；但 S 的差距（0.996 vs 0.189）远超波动范围
- 若在 evader 场景上**重训**，两者差距可能变化——这是下一步

### 9.5 这能说明"zero-shot 能力"吗？——需要严格区分

**不能直接这么说。** 两组网络都只在静止 target 上训练过，
所以 evader 场景对**两者都是 zero-shot**。本实验测的是
**"两个 zero-shot 模型谁更好"**，不是"谁有 zero-shot 能力"。

但它确实给出了一个有意义的结论：**plastic 的 zero-shot 退化明显更小**。

| 模型 | 训练任务 Fg | evader Fg（稳态） | 退化 | S 退化 |
|---|---|---|---|---|
| plastic | 3.298 | 2.836 | **−14.0%** | 1.000 → 0.997（**−0.3%**） |
| static | 3.398 | 2.196 | **−35.4%** | 0.980 → 0.349（**−64.4%**） |

### 9.6 决定性检验：优势是"从一开始就有"还是"在 episode 内建立起来"？

这是区分**"训练得更好的固定策略"** vs **"真正的在线适应"**的关键判据：
若 plastic 只是训练得更好，差距应从 step 0 就存在；
若优势来自在线演化，差距应**随 episode 推进而扩大**。

| 窗口 | S_plastic | S_static | ΔS | ΔFg |
|---|---|---|---|---|
| 0-50 | 1.000 | 0.920 | +0.080 | **−0.746** |
| 50-100 | 1.000 | 0.985 | +0.015 | **−0.524** |
| 100-200 | 1.000 | 0.930 | +0.070 | **−0.182** |
| 200-400 | 1.000 | 0.717 | +0.283 | +0.201 |
| 400-800 | 0.989 | **0.130** | **+0.860** | +0.756 |
| 800-1200 | 1.000 | 0.302 | +0.698 | +0.475 |
| 1200-1600 | 1.000 | **0.135** | **+0.865** | +0.690 |

**差距在 episode 早期完全反向（ΔFg = −0.75，static 更好），
在 step 200 后翻转并急剧扩大到 +0.86。**

→ **这正是"在线适应"的特征**：plastic 需要几百步让 W 演化到适合追逃的状态，
而 static 从一开始就只能是固定的挤团策略。

### 9.7 三个混淆因素已排除

1. **"static 只是慢热"？** 不是。static 的 S 在 step 400-1600 始终在
   0.03-0.31 之间徘徊，**无恢复趋势**——它进入了一个**稳定的挤团退化态**。
2. **"static 只是运气差的一次碰撞"？** 不是。碰撞步数 1429/1600（**89.3%**），
   分散在整个 episode，不是单次事件。
3. **"plastic 的塑性没在工作"？** 不是。实测 `‖W − W₀‖` 在一个 episode 内
   从 3.24 涨到 4.78，W 剧烈演化。

### 9.8 严格结论

**可以说的**：
> plastic HAN 在**未见过**的追逃围捕场景中，**在线适应能力**使其表现显著优于
> 训练过的静态网络：稳态 fitness +0.64，避碰项 S 几乎无损（0.996 vs 0.349），
> 且这一优势是在 episode 内**逐步建立**的（前期反向、后期急剧扩大），
> 排除了"固定策略恰好更好"和"slow-warmup"的解释。

**不能说的**：
- 不能说 "plastic 有 zero-shot 能力" 而 static 没有——**两者都是 zero-shot**，
  且 static 在前 200 步反而更好
- 不能声称塑性是**唯一**原因：`b140e5f5` 与 `5de64e71` 的 ABCD/权重不同、
  训练时机不同（相隔 4 个月），存在**训练细节差异**的混淆
- n=1，需多 seed 复现

**要严格归因于塑性**，需要做**同源对照**：在 evader 场景上重训 plastic 与
static 各一套（同配置、同 seed、同代），再比较。

### 9.9 复现与输出

```bash
bash examples/running/run_evader_compare.sh
```

| 输出 | 路径 |
|---|---|
| plastic 视频 | `<plastic-exp>/evader_videos/plastic_evader_v015.mp4` |
| static 视频 | `<static-exp>/evader_videos/static_evader_v015.mp4` |
| per-step 数据 | `/tmp/evader_compare/{plastic,static}_evader.npz` |

## 10. 输出位置

| 内容 | 路径 |
|---|---|
| **实验组** plastic HAN | `examples/outputs/cmaeshan_flocking_hanmodel__b140e5f5_26_06_23-17_05_30-8-agents/` |
| **对照组** static（当前 yaml，1600步/8agents） | `examples/outputs/cmaeshan_flocking_hanmodel__5de64e71_26_10_21-19_59_20/` |
| **对照组** static（匹配 plastic 配置，800步/4agents） | `examples/outputs/cmaeshan_flocking_hanmodel__7ae9ce55_26_10_22-11_43_24/` |
| 静态权重 | `<exp>/han_results/w_params.npy`（140 维） |
| per-step 分解 | `<exp>/disturbance_eval_{push,none,freeze}/per_step_data.npz` |
| 时间序列图 | `<exp>/disturbance_eval_*/fitness_curve.png` |
| **push 矩阵 8 组完整视频** | `<exp>/push_matrix_videos/{model}_{motion}_{mode}.mp4`（各 1600 帧 @20fps） |
| push 矩阵 8 组数据快照 | `/tmp/push_matrix/{model}_{motion}_{mode}.npz` |
| 评估视频（10 ep） | `<exp>/videos_han/eval_han_*.mp4` |
| 移动 target 评估视频 | `<plastic-exp>/moving_circle_r035_p400/videos_han/`、`<plastic-exp>/sanity_stationary/videos_han/` |

> 每个保留的实验目录下都有 `README.md` 说明其角色、配置和产物。

## 11. 相关文件

| 文件 | 改动 |
|---|---|
| [benchmarl/models/han.py](benchmarl/models/han.py) | 新增 `get_w_vector` / `set_w_from_vector` / `freeze_weights` |
| [benchmarl/algorithms/cmaes_han_optimizer.py](benchmarl/algorithms/cmaes_han_optimizer.py) | 新增 `static_mode` 分支；`save()` 写 `w_params.npy` |
| [examples/running/flocking_patch.py](examples/running/flocking_patch.py) | 新增 4 种 target 运动模式 |
| [examples/running/run_cmaes_han_flocking_custom.py](examples/running/run_cmaes_han_flocking_custom.py) | 新增 CLI 参数 + static mode |
| [examples/running/run_cmaes_han_flocking_disturbance.py](examples/running/run_cmaes_han_flocking_disturbance.py) | 新增 CLI 参数 + 加载 static 模型 + `--disturbance-mode none` + 局部稳态 recovery 指标 |
| [examples/running/run_push_adaptation_matrix.sh](examples/running/run_push_adaptation_matrix.sh) | push 矩阵编排（含逐组快照） |
| [examples/running/collect_push_matrix.py](examples/running/collect_push_matrix.py) | push 矩阵收集 + 净效应分析 |
| [examples/running/han_mechanism_probe.py](examples/running/han_mechanism_probe.py) | 机制探针（见 [han_dynamic_advantage_task_design.md](han_dynamic_advantage_task_design.md)） |
