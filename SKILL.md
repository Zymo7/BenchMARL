# BenchMARL 开发指南：添加新网络结构与任务环境

---

## 一、添加新的网络结构（Model）

BenchMARL 的模型系统基于 `Model` 基类和 `ModelConfig` 配置类。每个模型需要实现前向传播逻辑，并通过注册表被框架发现。

### 需要创建/修改的文件

```
benchmarl/
├── models/
│   ├── __init__.py              # 【修改】注册新模型
│   ├── common.py                # 基类（不需修改）
│   ├── [your_model].py          # 【新建】模型实现
│   └── hebbian.py               # 参考示例
├── conf/
│   └── model/
│       └── layers/
│           └── [your_model].yaml # 【新建】模型 YAML 配置
└── __init__.py                   # 不需修改（模型不由 hydra schema 管理）
```

### 步骤

#### 1. 创建模型实现文件 `benchmarl/models/[your_model].py`

```python
from dataclasses import dataclass, MISSING
from typing import Type
import torch
from torch import nn
from tensordict import TensorDictBase
from benchmarl.models.common import Model, ModelConfig


class YourModelLayer(nn.Module):
    """自定义层的核心逻辑。"""

    def __init__(self, in_features, out_features, your_param=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义参数...

    def forward(self, x):
        # 前向传播...
        return output


class YourModel(Model):
    """自定义模型，被框架通过 TensorDict 调用。"""

    def __init__(self, your_param: float, **kwargs):
        super().__init__(**kwargs)
        # 关键属性（由基类提供）：
        #   self.input_leaf_spec  → 输入形状
        #   self.output_leaf_spec → 输出形状
        #   self.n_agents         → agent 数量
        #   self.centralised      → 是否集中式
        #   self.share_params     → 是否共享参数
        #   self.output_has_agent_dim → 输出是否包含 agent 维度

        in_features = self.input_leaf_spec.shape[-1]
        out_features = self.output_leaf_spec.shape[-1]

        self.layer = YourModelLayer(in_features, out_features, your_param)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        x = tensordict.get(self.in_key)           # 读取输入
        x = self.layer(x)                          # 前向计算
        tensordict.set(self.out_key, x)            # 写入输出
        return tensordict

    def _perform_checks(self):
        super()._perform_checks()
        # 添加自定义检查（可选）


@dataclass
class YourModelConfig(ModelConfig):
    """模型的配置类。"""

    your_param: float = MISSING

    @staticmethod
    def associated_class():
        return YourModel
```

#### 2. 创建 YAML 配置 `benchmarl/conf/model/layers/[your_model].yaml`

```yaml
name: your_model    # 必须与 model_config_registry 中的 key 一致

your_param: 0.01
# 以下为通用参数（所有模型都有）：
activation_class: null
activation_kwargs: null
num_feature_dims: 1
```

#### 3. 注册模型 `benchmarl/models/__init__.py`

```python
# 添加导入
from .your_model import YourModel, YourModelConfig

# 添加到 classes 列表
classes = [
    # ...existing...
    "YourModel",
    "YourModelConfig",
]

# 添加到注册表（key 必须与 YAML 的 name 字段一致）
model_config_registry = {
    # ...existing...
    "your_model": YourModelConfig,
}
```

#### 4. 使用新模型

```python
from benchmarl.models import YourModelConfig

# 单独使用
model_config = YourModelConfig(your_param=0.01)

# 组合使用（如 MLP + YourModel）
from benchmarl.models import MlpConfig, SequenceModelConfig
model_config = SequenceModelConfig(
    model_configs=[
        MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
        YourModelConfig(your_param=0.01),
    ],
    intermediate_sizes=[64],
)
```

### 参考：现有模型对照表

| 模型 | 文件 | 配置文件 | 特点 |
|------|------|---------|------|
| MLP | `mlp.py` | `mlp.yaml` | 标准全连接网络 |
| Hebbian | `hebbian.py` | `hebbian.yaml` | 赫布学习动态层，W 在线更新 |
| GRU | `gru.py` | `gru.yaml` | 门控循环单元 |
| LSTM | `lstm.py` | `lstm.yaml` | 长短期记忆 |
| GNN | `gnn.py` | `gnn.yaml` | 图神经网络 |
| CNN | `cnn.py` | `cnn.yaml` | 卷积网络 |
| Deepsets | `deepsets.py` | `deepsets.yaml` | 深集合网络 |

### 关键注意事项

- **`ModelConfig.associated_class()`** 必须返回对应的 Model 类
- **`_forward()`** 是核心方法，通过 `tensordict.get/set` 读写数据
- **`num_feature_dims`**：模型输出中 feature 维度的数量（通常为 1）
- 组合模型通过 `SequenceModelConfig` 拼接，`intermediate_sizes` 指定中间层维度
- 如果模型需要特殊的优化方式（如 Hebbian 用 CMA-ES），需要同时在 `algorithms/` 中添加对应的算法支持

---

## 二、添加新的 VMAS 任务环境

添加新任务需要在 **两个包** 中操作：BenchMARL（任务配置层）和 VMAS（场景逻辑层）。

### 整体架构

```
VMAS 包（场景逻辑）                    BenchMARL 包（配置与集成）
├── vmas/scenarios/                   ├── benchmarl/environments/vmas/
│   └── [task_name].py  ←──场景实现    │   ├── common.py          ←──枚举注册
│                                     │   └── [task_name].py     ←──TaskConfig
└── vmas/__init__.py  ←──场景列表      ├── benchmarl/conf/task/vmas/
                                      │   └── [task_name].yaml   ←──YAML 配置
                                      └── benchmarl/__init__.py  ←──自动加载schema
```

### 需要创建/修改的文件

| # | 文件 | 操作 | 位置 |
|---|------|------|------|
| 1 | `vmas/scenarios/[task_name].py` | **新建** | VMAS 包目录 |
| 2 | `vmas/__init__.py` | **修改** | VMAS 包目录，添加场景名到 `scenarios` 列表 |
| 3 | `benchmarl/environments/vmas/common.py` | **修改** | 添加枚举值 |
| 4 | `benchmarl/environments/vmas/[task_name].py` | **新建** | TaskConfig |
| 5 | `benchmarl/conf/task/vmas/[task_name].yaml` | **新建** | YAML 配置 |

### 步骤

#### 1. 创建 VMAS 场景 `vmas/scenarios/[task_name].py`

场景必须包含一个继承 `BaseScenario` 的 `Scenario` 类，实现以下方法：

```python
import torch
from torch import Tensor
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """初始化世界（只调用一次）。"""
        # 1. 从 kwargs 读取配置参数
        self.n_agents = kwargs.pop("n_agents", 3)
        self.max_steps = kwargs.pop("max_steps", 200)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        # 2. 创建 World
        world = World(batch_dim, device, substeps=5, dt=0.1, gravity=(0, 0))

        # 3. 添加 Agent（必须有 collide=True 和 sensors）
        for i in range(self.n_agents):
            agent = Agent(name=f"agent_{i}", collide=True, ...)
            world.add_agent(agent)

        # 4. 添加 Landmark（目标、障碍物等）
        goal = Landmark(name=f"goal_{i}", collide=False, ...)
        world.add_landmark(goal)

        return world

    def reset_world_at(self, env_index: int = None):
        """重置环境（每个 episode 调用）。"""
        ScenarioUtils.spawn_entities_randomly(...)
        # 初始化状态变量（goal_dist 等）

    def pre_step(self):
        """【可选】每步物理仿真前调用，用于更新动态障碍物等。"""
        pass

    def reward(self, agent: Agent) -> Tensor:
        """计算单个 agent 的奖励。"""
        is_first = agent == self.world.agents[0]
        if is_first:
            # 全局计算（只做一次）
            ...
        return reward

    def observation(self, agent: Agent):
        """返回 agent 的观测（Tensor 或 Dict）。"""
        obs = torch.cat([...], dim=-1)
        return obs  # 或 return {"obs": obs, "pos": ..., "vel": ...}

    def done(self) -> Tensor:
        """判断 episode 是否结束。"""
        return self.all_goal_reached

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """返回额外信息。"""
        return {"pos_rew": ..., "collision_rew": ...}

    def extra_render(self, env_index: int = 0):
        """【可选】额外渲染元素。"""
        return []
```

**关键要点：**
- `make_world` 中用 `kwargs.pop()` 读取参数，最后调用 `check_kwargs_consumed(kwargs)`
- `reward()` 中用 `is_first` 模式避免重复计算全局部分
- `observation()` 返回的维度必须与 `make_world` 中的配置一致
- 需要动态更新实体时，在 `pre_step()` 中用 `entity.set_pos()` 修改位置
- Lidar 的 `entity_filter` 决定了传感器能感知哪些实体类型
- `collide=True` 的 Landmark 才能被碰撞检测和 Lidar 感知

#### 2. 注册场景到 VMAS `vmas/__init__.py`

在 `scenarios` 列表中添加场景名：

```python
scenarios = sorted([
    # ...existing...
    "your_task_name",
])
```

#### 3. 添加枚举值 `benchmarl/environments/vmas/common.py`

在 `VmasTask` 枚举中添加：

```python
class VmasTask(Task):
    # ...existing...
    YOUR_TASK_NAME = None  # 枚举名会自动转为小写匹配 VMAS 场景
```

**注意**：枚举名会被 `.lower()` 后传给 VMAS 的场景加载器，所以 `YOUR_TASK_NAME` 对应的场景文件必须叫 `your_task_name.py`。

#### 4. 创建 TaskConfig `benchmarl/environments/vmas/[task_name].py`

```python
from dataclasses import dataclass, field
from typing import List


@dataclass
class TaskConfig:
    max_steps: int = 200
    n_agents: int = 3
    # 添加场景 kwargs 中所有 pop 的参数
    # 注意：List 类型必须用 field(default_factory=...)，不能用 None
    obstacle_modes: List[str] = field(default_factory=lambda: ["linear", "circular"])
```

**注意**：
- 所有参数必须有默认值（OmegaConf 不支持 MISSING 以外的 None）
- `List` 类型必须用 `field(default_factory=...)`，不能用 `list = None`
- 参数名必须与 VMAS 场景中 `kwargs.pop()` 的 key 完全一致

#### 5. 创建 YAML 配置 `benchmarl/conf/task/vmas/[task_name].yaml`

```yaml
defaults:
  - _self_

max_steps: 200
n_agents: 3
obstacle_modes:
  - linear
  - circular
```

**注意**：如果遇到 Hydra 的 defaults 引用问题，只用 `- _self_` 即可（不引用 base config）。

### 验证清单

```bash
# 1. 验证 VMAS 场景可独立加载
conda run -n benchmarl python -c "
from vmas.make_env import make_env
env = make_env('your_task_name', num_envs=2, device='cpu')
env.reset()
env.step({agent.name: torch.zeros(2, 2) for agent in env.agents})
print('OK')
"

# 2. 验证 BenchMARL 任务注册
conda run -n benchmarl python -c "
from benchmarl.environments import VmasTask
print([t for t in VmasTask if 'YOUR' in t.name])
"

# 3. 验证配置加载
conda run -n benchmarl python -c "
from benchmarl.environments import VmasTask
task = VmasTask.YOUR_TASK_NAME.get_from_yaml()
print(task.config)
"

# 4. 验证完整实验流程（快速测试）
conda run -n benchmarl python -c "
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms.ippo import IppoConfig
from benchmarl.models import MlpConfig
import torch

experiment_config = ExperimentConfig.get_from_yaml()
experiment_config.max_n_iters = 2
experiment_config.save_folder = '/tmp/test'

experiment = Experiment(
    task=VmasTask.YOUR_TASK_NAME.get_from_yaml(),
    algorithm_config=IppoConfig.get_from_yaml(),
    model_config=MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh),
    seed=0,
    config=experiment_config,
)
experiment.run()
print('Full pipeline OK')
"
```

### VMAS 场景开发参考

| 参考场景 | 文件 | 学习要点 |
|---------|------|---------|
| `navigation` | `vmas/scenarios/navigation.py` | 基础导航、目标分配 |
| `navigation_obs` | `vmas/scenarios/navigation_obs.py` | 静态障碍物、多动力学类型 |
| `navigation_dynamic_obs` | `vmas/scenarios/navigation_dynamic_obs.py` | 动态障碍物、pre_step 更新 |
| `balance` | `vmas/scenarios/balance.py` | 合作平衡任务 |

### 常见问题

1. **`AssertionError: scenario not found`**：检查 `vmas/__init__.py` 的 `scenarios` 列表是否包含场景名
2. **`ValidationError: Non optional ListConfig cannot be constructed from None`**：TaskConfig 中的 List 必须用 `field(default_factory=...)`
3. **`Rays are only casted among collidables`**：Lidar 只能检测 `collide=True` 的实体
4. **`check_kwargs_consumed` 报错**：YAML 中的参数名必须与 `kwargs.pop()` 的 key 精确匹配

---

## 三、创建训练运行脚本

BenchMARL 提供两种运行方式：**Hydra 命令行**（适合标准算法+标准模型）和 **Python 脚本**（适合自定义模型组合、多阶段训练等）。这里重点介绍后者。

### 3.1 基本概念

一个训练脚本的核心流程：

```
配置加载 → 创建 Experiment → experiment.run() → （可选）后处理
```

`Experiment` 是核心对象，需要提供：
- `task`：任务配置（从 YAML 或代码创建）
- `algorithm_config`：算法配置
- `model_config`：Actor 网络配置
- `critic_model_config`：Critic 网络配置
- `config`：实验超参数配置
- `seed`：随机种子

### 3.2 最简脚本（标准 IPPO + MLP）

适用于：标准算法 + 标准模型，不需要自定义逻辑。

```python
# examples/running/run_ippo_navigation.py
from pathlib import Path
import torch
from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig

if __name__ == "__main__":
    # 1. 加载配置
    experiment_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.NAVIGATION.get_from_yaml()
    algorithm_config = IppoConfig.get_from_yaml()

    # 2. 配置实验参数
    experiment_config.max_n_iters = 100
    experiment_config.save_folder = str(Path(__file__).parent.parent / "outputs")

    # 3. 创建实验
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
        critic_model_config=MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
        seed=0,
        config=experiment_config,
    )

    # 4. 训练
    experiment.run()
```

运行：
```bash
conda run -n benchmarl python examples/running/run_ippo_navigation.py
```

### 3.3 多阶段训练脚本（IPPO-Hebbian 模板）

适用于：PPO 训练 + CMA-ES 优化 Hebbian 参数的两阶段训练。这是当前项目中最常用的脚本模式。

```python
# examples/running/run_ippo_hebbian_[task_name].py
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from benchmarl.algorithms.ippo_hebbian import IppoHebbianConfig
from benchmarl.algorithms.cmaes_optimizer import CmaesHebbianOptimizer
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models import MlpConfig
from benchmarl.models.hebbian import HebbianConfig
from benchmarl.models.common import SequenceModelConfig


# ========== 命令行参数 ==========
def parse_args():
    parser = argparse.ArgumentParser(description="IPPO-Hebbian Training")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="跳过训练，仅运行评估")
    parser.add_argument("--experiment-path", type=str, default=None,
                        help="已有实验目录（evaluate-only 模式必需）")
    parser.add_argument("--n-eval-episodes", type=int, default=20,
                        help="评估 episode 数量")
    parser.add_argument("--max-n-iters", type=int, default=200,
                        help="PPO 训练迭代次数")
    parser.add_argument("--cmaes-gens", type=int, default=30,
                        help="CMA-ES 优化代数")
    parser.add_argument("--pop-size", type=int, default=30,
                        help="CMA-ES 种群大小")
    return parser.parse_args()


args = parse_args()


# ========== 网络配置 ==========
# ★ 修改此处来更换网络结构
def _create_model_configs():
    model_config = SequenceModelConfig(
        model_configs=[
            MlpConfig(num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear),
            HebbianConfig(lr_hebb=0.01, weight_init=1.0),
        ],
        intermediate_sizes=[64],
    )
    critic_model_config = MlpConfig(
        num_cells=[64, 64], activation_class=torch.nn.Tanh, layer_class=torch.nn.Linear,
    )
    return model_config, critic_model_config


# ========== 实验创建 ==========
def _create_experiment(task, model_config, critic_model_config, output_dir, max_n_iters=200):
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.max_n_iters = max_n_iters
    experiment_config.save_folder = str(output_dir)
    return Experiment(
        task=task,
        algorithm_config=IppoHebbianConfig.get_from_yaml(),
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )


if __name__ == "__main__":
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    if args.evaluate_only:
        # ========== 仅评估模式 ==========
        # 从已有实验中加载训练好的模型，重新生成评估视频
        if args.experiment_path is None:
            raise ValueError("--experiment-path is required for evaluate-only mode")

        exp_path = Path(args.experiment_path)
        hebbian_dir = exp_path / "hebbian_results"

        # 加载元数据
        with open(hebbian_dir / "results.json") as f:
            metadata = json.load(f)

        # ★ 修改此处匹配目标任务
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
        model_config, critic_model_config = _create_model_configs()
        experiment = _create_experiment(task, model_config, critic_model_config, output_dir)

        # 加载训练好的策略
        policy_path = hebbian_dir / "policy_state.pt"
        if not policy_path.exists():
            raise FileNotFoundError(f"policy_state.pt not found. Re-run training first.")
        experiment.policy.load_state_dict(
            torch.load(str(policy_path), map_location=experiment.config.train_device)
        )

        # 加载 Hebbian ABCD 参数
        hebbian_layer = experiment.algorithm.get_hebbian_layer()
        abcd_path = hebbian_dir / "abcd_params.npy"
        if abcd_path.exists():
            abcd = np.load(str(abcd_path))
            hebbian_layer.set_abcd_from_vector(
                torch.tensor(abcd, device=experiment.config.train_device)
            )
            hebbian_layer.reset_weights()

        # 评估
        optimizer = CmaesHebbianOptimizer(
            experiment=experiment, hebbian_layer=hebbian_layer,
            pop_size=1, max_gens=0, n_eval_episodes=1,
            device=experiment.config.train_device,
        )
        if abcd_path.exists():
            optimizer._best_abcd_so_far = np.load(str(abcd_path))

        optimizer.evaluate(output_dir=str(exp_path), n_episodes=args.n_eval_episodes)

    else:
        # ========== 完整训练模式 ==========

        # ★ 修改此处匹配目标任务
        task = VmasTask.NAVIGATION_DYNAMIC_OBS.get_from_yaml()
        model_config, critic_model_config = _create_model_configs()
        experiment = _create_experiment(
            task, model_config, critic_model_config, output_dir, max_n_iters=args.max_n_iters
        )

        # ---- Phase 1: PPO 训练 ----
        print("Phase 1: Training MLP layers with PPO")
        experiment.run()

        # ---- Phase 2: CMA-ES 优化 Hebbian ABCD ----
        print("Phase 2: Optimizing Hebbian ABCD with CMA-ES")
        hebbian_layer = experiment.algorithm.get_hebbian_layer()
        if hebbian_layer is None:
            raise RuntimeError("Could not find Hebbian layer in the policy")

        optimizer = CmaesHebbianOptimizer(
            experiment=experiment, hebbian_layer=hebbian_layer,
            pop_size=args.pop_size, sigma0=0.5,
            max_gens=args.cmaes_gens, n_eval_episodes=6,
            device=experiment.config.train_device,
        )
        best_abcd = optimizer.run()

        # 处理中断恢复
        if optimizer._current_gen < optimizer.max_gens and optimizer._best_abcd_so_far is not None:
            optimizer.apply_best_so_far()
            best_abcd = optimizer._best_abcd_so_far

        # 保存结果
        optimizer.save(output_dir=str(experiment.folder_name))
        optimizer.plot_convergence(output_dir=str(experiment.folder_name))
        optimizer.evaluate(output_dir=str(experiment.folder_name), n_episodes=args.n_eval_episodes)
```

### 3.4 创建新脚本的检查清单

从模板复制新脚本时，需要修改以下位置（标记为 ★）：

| 修改点 | 位置 | 说明 |
|--------|------|------|
| **目标任务** | `VmasTask.YOUR_TASK.get_from_yaml()` | evaluate-only 和 full training 两处都要改 |
| **网络结构** | `_create_model_configs()` | 修改 MLP 层数/大小、Hebbian 参数、或替换为其他模型 |
| **算法配置** | `_create_experiment()` 中的 `algorithm_config` | 如不用 IPPO-Hebbian，改为对应算法的 Config |
| **默认参数** | `parse_args()` 中的 `default=` | 根据任务难度调整默认训练轮次等 |

### 3.5 实验输出目录结构

```
outputs/
└── ippohebbian_navigation_dynamic_obs_sequencemodel__[hash]_[date]/
    ├── config.pkl                                    # 任务配置
    ├── [experiment_name].json                        # 训练结果（marl-eval 格式）
    ├── hebbian_results/
    │   ├── abcd_params.npy                           # CMA-ES 优化的 ABCD 参数
    │   ├── hebbian_state.pt                          # Hebbian 层完整 state_dict
    │   ├── policy_state.pt                           # 完整策略 state_dict（用于 evaluate-only）
    │   └── results.json                              # 元数据（fitness, 世代数等）
    ├── videos_hebbian/
    │   ├── eval_hebbian_0.mp4                        # 评估视频
    │   └── ...
    ├── cmaes_convergence.png                         # CMA-ES 收敛曲线
    └── wandb/                                        # Wandb 日志
```

### 3.6 运行命令速查

```bash
# 完整训练（使用默认参数）
python examples/running/run_ippo_hebbian_[task].py

# 快速测试（少量迭代）
python examples/running/run_ippo_hebbian_[task].py --max-n-iters 5 --cmaes-gens 2 --n-eval-episodes 3

# 自定义参数训练
python examples/running/run_ippo_hebbian_[task].py --max-n-iters 300 --cmaes-gens 50 --pop-size 40

# 训练完成后重新生成评估视频（不重新训练）
python examples/running/run_ippo_hebbian_[task].py \
    --evaluate-only \
    --experiment-path outputs/[experiment_folder] \
    --n-eval-episodes 50
```

### 3.7 Hydra 命令行方式（备选）

如果只需要标准算法+标准模型（不需要自定义网络组合），可以直接用 Hydra：

```bash
# 标准格式
python benchmarl/run.py algorithm=[algo] task=vmas/[task]

# 示例
python benchmarl/run.py algorithm=mappo task=vmas/navigation
python benchmarl/run.py algorithm=ippo task=vmas/navigation_dynamic_obs

# 覆盖参数
python benchmarl/run.py algorithm=ippo task=vmas/navigation max_n_iters=100 lr=0.001
```

**限制**：Hydra 方式不支持 `SequenceModelConfig`（模型组合），因此 IPPO-Hebbian 等需要自定义模型组合的场景必须使用 Python 脚本。

---

## 四、文件路径速查

### VMAS 包路径
```
/home/zhaozeming/miniconda3/envs/benchmarl/lib/python3.10/site-packages/vmas/
├── __init__.py                          # scenarios 列表
├── scenarios/
│   ├── __init__.py                      # 场景加载器
│   ├── [task_name].py                   # 场景实现
│   └── navigation_dynamic_obs.py        # 动态障碍物示例
└── simulator/
    ├── core.py                          # Agent, Landmark, World 等
    ├── scenario.py                      # BaseScenario 基类
    ├── sensors.py                       # Lidar 等传感器
    └── utils.py                         # ScenarioUtils 等工具
```

### BenchMARL 路径
```
/home/zhaozeming/BenchMARL/
├── benchmarl/
│   ├── __init__.py                      # Hydra schema 自动加载
│   ├── models/
│   │   ├── __init__.py                  # model_config_registry
│   │   ├── common.py                    # Model, ModelConfig, SequenceModelConfig
│   │   ├── mlp.py                       # MLP 示例
│   │   └── hebbian.py                   # Hebbian 示例
│   ├── environments/
│   │   ├── __init__.py                  # task_config_registry 自动生成
│   │   ├── common.py                    # Task, TaskClass 基类
│   │   └── vmas/
│   │       ├── common.py                # VmasTask 枚举, VmasClass
│   │       └── [task_name].py           # TaskConfig
│   ├── algorithms/
│   │   ├── __init__.py                  # algorithm_config_registry
│   │   └── ippo_hebbian.py             # IPPO-Hebbian 示例
│   └── conf/
│       ├── model/layers/               # 模型 YAML
│       ├── task/vmas/                  # 任务 YAML
│       └── algorithm/                  # 算法 YAML
└── examples/
    └── running/                        # 运行脚本
```
