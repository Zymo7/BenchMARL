# BenchMARL 代码库使用指南

## 目录结构

```
BenchMARL/
├── benchmarl/                    # 核心源代码
│   ├── algorithms/               # MARL 算法实现
│   ├── benchmark/                # 基准测试相关
│   ├── conf/                     # Hydra 配置文件
│   │   ├── algorithm/            # 算法配置
│   │   ├── experiment/           # 实验配置
│   │   ├── model/                # 模型配置
│   │   └── task/                 # 任务环境配置
│   │       ├── vmas/             # VMAS 环境
│   │       ├── smacv2/           # SMACv2 环境
│   │       ├── pettingzoo/       # PettingZoo 环境
│   │       ├── meltingpot/       # MeltingPot 环境
│   │       └── magent/           # MAgent2 环境
│   ├── environments/             # 环境封装
│   ├── experiment/               # 实验运行逻辑
│   ├── models/                   # 神经网络模型
│   ├── run.py                    # 主入口脚本
│   ├── evaluate.py               # 评估脚本
│   ├── resume.py                 # 恢复训练脚本
│   ├── eval_results.py           # 结果评估工具
│   ├── hydra_config.py           # Hydra 配置加载
│   └── utils.py                  # 工具函数
├── examples/                     # 示例代码
├── fine_tuned/                   # 调优后的基准配置
└── outputs/                     # 实验输出目录
```

---

## 核心模块详解

### 1. algorithms/ - 算法模块

| 算法 | 配置文件 | 类型 | 适用动作空间 |
|------|----------|------|-------------|
| **MAPPO** | `mappo.yaml` | On-policy | 连续 + 离散 |
| **IPPO** | `ippo.yaml` | On-policy | 连续 + 离散 |
| **IPPO-Hebbian** | `ippo_hebbian.yaml` | On-policy | 连续 + 离散 |
| **MADDPG** | `maddpg.yaml` | Off-policy | 连续 |
| **IDDPG** | `iddpg.yaml` | Off-policy | 连续 |
| **MASAC** | `masac.yaml` | Off-policy | 连续 + 离散 |
| **ISAC** | `isac.yaml` | Off-policy | 连续 + 离散 |
| **QMIX** | `qmix.yaml` | Off-policy | 离散 |
| **VDN** | `vdn.yaml` | Off-policy | 离散 |
| **IQL** | `iql.yaml` | Off-policy | 离散 |

#### MAPPO 算法特有参数
```yaml
share_param_critic: True       # 是否共享 Critic 参数
clip_epsilon: 0.2              # PPO 裁剪 epsilon
entropy_coef: 0.0              # 熵系数
critic_coef: 1.0              # Critic 损失系数
loss_critic_type: "l2"        # Critic 损失类型
lmbda: 0.9                    # GAE lambda
use_tanh_normal: True          # 是否使用 tanh 正态分布
```

### 2. environments/ - 环境模块

#### 支持的环境

| 环境 | 安装命令 | 任务数 | 向量化 |
|------|----------|--------|--------|
| **VMAS** | `pip install vmas` | 27 | Yes |
| **SMACv2** | 见 [安装指南](README.md#smacv2) | 15 | No |
| **PettingZoo** | `pip install "pettingzoo[all]"` | 10 | No |
| **MeltingPot** | `pip install dm-meltingpot` | 49 | No |
| **MAgent2** | `pip install git+https://github.com/Farama-Foundation/MAgent2` | 1 | No |

#### VMAS 任务列表
- **平衡类**: `balance`, `wheel`
- **导航类**: `navigation`, `discovery`, `flocking`, `wind_flocking`
- **通道类**: `passage`, `joint_passage`, `joint_passage_size`, `ball_passage`
- **对抗类**: `simple_tag`, `simple_push`, `simple_adversary`
- **合作类**: `simple_spread`, `simple_reference`, `simple_speaker_listener`
- **传输类**: `transport`, `reverse_transport`, `football`
- **其他**: `buzz_wire`, `dropout`, `sampling`, `dispersion`, `give_way`, `multi_give_way`, `ball_trajectory`

### 3. models/ - 模型模块

| 模型 | 说明 | 适用场景 |
|------|------|----------|
| **MLP** | 多层感知机 | 默认选择 |
| **Hebbian** | 赫布学习动态层 | 在线自适应 |
| **GRU** | 门控循环单元 | 时序依赖 |
| **LSTM** | 长短期记忆网络 | 时序依赖 |
| **GNN** | 图神经网络 | 多智能体通信 |
| **CNN** | 卷积神经网络 | 视觉输入 |
| **Deepsets** | 深集合网络 | 置换不变性 |

#### IPPO-Hebbian 算法说明

IPPO-Hebbian 是一种两阶段训练算法，结合了 IPPO 的策略优化和赫布学习的在线适应能力。

**网络架构**：
```
输入 (obs_dim) → MLP1 → MLP2 → Hebbian输出层 → 动作分布
                                    ↑
                   ABCD 参数通过 CMA-ES 优化
                   W 在执行过程中按 ABCD 规则在线更新
```

**赫布学习层 (Hebbian Layer)**：
权重更新规则：$\Delta w_{ij} = A \cdot x_i \cdot y_j + B \cdot x_i + C \cdot y_j + D$

其中 $x_i$ 是前层神经元激活，$y_j$ 是输出神经元激活，A、B、C、D 是每个连接的四个学习参数。

**训练阶段**：
1. **Phase 1**：用 PPO 训练前两层 MLP，Hebbian 层固定（W 保持初始值）
2. **Phase 2**：冻结 MLP 层，用 CMA-ES 优化 Hebbian 层的 ABCD 参数

**运行方式**：
```python
# 示例脚本
python examples/running/run_ippo_hebbian.py
```

### 4. experiment/ - 实验模块

#### 关键实验参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_n_frames` | 3,000,000 | 最大训练帧数 |
| `max_n_iters` | null | 最大迭代次数 |
| `lr` | 0.00005 | 学习率 |
| `gamma` | 0.99 | 折扣因子 |
| `evaluation_interval` | 120,000 | 评估间隔（帧） |
| `evaluation_episodes` | 10 | 评估 episodes 数 |
| `checkpoint_interval` | 0 | 保存检查点间隔 |

#### On-policy 参数 (MAPPO, IPPO)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `on_policy_collected_frames_per_batch` | 6000 | 每次收集的帧数 |
| `on_policy_n_envs_per_worker` | 10 | 环境数量 |
| `on_policy_n_minibatch_iters` | 45 | 每个 batch 的训练轮次 |
| `on_policy_minibatch_size` | 400 | 小批量大小 |

#### Off-policy 参数 (SAC, DDPG, QMIX 等)
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `off_policy_collected_frames_per_batch` | 6000 | 每次收集的帧数 |
| `off_policy_n_envs_per_worker` | 10 | 环境数量 |
| `off_policy_n_optimizer_steps` | 1000 | 优化器步数 |
| `off_policy_train_batch_size` | 128 | 训练批量大小 |
| `off_policy_memory_size` | 1,000,000 | Replay buffer 大小 |

---

## 快速开始

### 基本运行命令

```bash
# 运行 MAPPO + Navigation
python benchmarl/run.py algorithm=mappo task=vmas/navigation

# 运行 IPPO + Navigation
python benchmarl/run.py algorithm=ippo task=vmas/navigation

# 运行 IPPO-Hebbian（两阶段训练，需要自定义网络配置）
python examples/running/run_ippo_hebbian.py

# 运行 QMIX + SMACv2 (需要离散动作)
python benchmarl/run.py algorithm=qmix task=smacv2/terran_5_vs_6

# 多任务对比
python benchmarl/run.py algorithm=mappo task=vmas/balance,vmas/navigation
```

> **注意**：IPPO-Hebbian 算法需要自定义网络配置（MLP + Hebbian 层组合），因此使用独立的示例脚本 `examples/running/run_ippo_hebbian.py` 运行，而不是通过 Hydra 命令行。

### 修改训练参数

```bash
# 修改最大训练帧数
python benchmarl/run.py algorithm=mappo task=vmas/navigation max_n_frames=5_000_000

# 修改学习率
python benchmarl/run.py algorithm=mappo task=vmas/navigation lr=0.001

# 修改评估间隔
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation_interval=100_000

# 修改设备
python benchmarl/run.py algorithm=mappo task=vmas/navigation sampling_device=cpu train_device=cuda

# 禁用视频记录（解决 torchvision 兼容性问题）
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation.save_video=False
```

### 多参数组合

```bash
python benchmarl/run.py \
    algorithm=mappo,qmix,masac \
    task=vmas/balance,vmas/navigation \
    seed=0,1,2
```

### 使用不同模型

```bash
# 使用 GNN 模型（适合多智能体通信）
python benchmarl/run.py algorithm=mappo task=vmas/navigation model=gnn

# 使用 LSTM
python benchmarl/run.py algorithm=mappo task=vmas/navigation model=lstm
```

---

## 训练后操作

### 评估训练好的模型

```bash
python benchmarl/evaluate.py <checkpoint_path>
```

### 恢复中断的训练

```bash
python benchmarl/resume.py <checkpoint_path>
```

### 查看输出结果

训练结果保存在 `outputs/` 目录下：
- `checkpoints/` - 模型检查点
- `logs/` - 日志文件
- `wandb/` - Wandb 日志（如启用）
- `{experiment_name}.json` - marl-eval 格式结果

---

## 常见问题

### 1. torchvision 版本兼容问题

如果遇到 `AttributeError: module 'torchvision.io' has no attribute 'write_video'`，禁用视频记录：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation evaluation.save_video=False
```
或者


### 2. CUDA 版本问题

如果看到 CUDA 版本警告但仍能运行，可以强制使用 CPU：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation sampling_device=cpu train_device=cpu
```

### 3. Wandb 离线模式

如果网络不稳定，使用离线模式：

```bash
python benchmarl/run.py algorithm=mappo task=vmas/navigation loggers=[csv] wandb_extra_kwargs.mode=offline
```

### 4. 生成视频过短

```bash
1. 增加环境步数（让 episode 更长）                                                                                                                             
                
python benchmarl/run.py algorithm=mappo task=vmas/navigation task.max_steps=200                                                                                
                                                                                                                                                                
max_steps 在 benchmarl/conf/task/vmas/navigation.yaml 中默认是 100，改大即可。


2. 降低视频帧率（让视频播放更慢）                                                                                                                              
                
python benchmarl/run.py algorithm=mappo task=vmas/navigation wandb_extra_kwargs.video_fps=10                                                                   
                                                                                                                                                                
默认 FPS 较高，100 步的环境以高帧率播放自然只有 1 秒。降低 FPS 会让同样步数的视频播放更久。                                                                    
                                                                                                                                                                
两者结合                                                                                                                                                       
                
python benchmarl/run.py algorithm=mappo task=vmas/navigation task.max_steps=200 wandb_extra_kwargs.video_fps=10                                                
                                                                                                                                                                
这样视频会同时更长（更多步）和更慢（更低帧率）。
```

```bash
● 从代码可以看出，navigation 任务默认没有障碍物。VMAS 的 navigation scenario 位于 pip 包中：                                                                   
                                                                                                                                                               
  /home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/navigation.py                                                         
                                                                                                                                                               
  如何添加障碍物                                                                                                                                               
                                                                                                                                                               
  方案 1：直接修改 vmas 包（简单但不推荐升级包）                                                                                                               
                                                                                                                                                               
  # 编辑 /home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/navigation.py
                                                                                                                                                               
  在 make_world 方法中，在添加 agents 之后、返回 world 之前添加障碍物：                                                                                        
                                                                                                                                                               
  # 在 self.world.add_landmark(goal) 之后添加...                                                                                                               
                  
  # 添加障碍物
  obstacle = Landmark(
      name="obstacle_1",
      collide=True,
      shape=Sphere(radius=0.2),  # 障碍物大小
      color=(0.5, 0.5, 0.5),
  )
  world.add_landmark(obstacle)
                                                                                                                                                               
  方案 2：创建自定义任务（推荐）
                                                                                                                                                               
  BenchMARL 支持扩展任务。你可以参考 examples/extending/task/ 创建自定义 navigation 任务。

  方案 3：修改 VMAS 安装目录的 scenario                                                                                                                        
  
  # 复制并修改
  cp /home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/navigation.py \
  # 复制并修改
  cp /home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/navigation.py \
     /your/path/custom_navigation.py
  # 复制并修改
  cp /home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/scenarios/navigation.py \
     /your/path/custom_navigation.py

  # 修改后在运行命令中指定
```


