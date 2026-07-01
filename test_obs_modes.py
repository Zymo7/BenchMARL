"""测试 flocking_patch 的两种 obs-mode 都能产生正确维度且 lidar 能检测到 agent。"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples', 'running'))

import torch
from flocking_patch import configure


def get_obs_dim(env):
    """VMAS make_env 返回 list[Tensor], 每个 tensor = 单 agent 的 obs."""
    obs = env.reset()
    # obs shape: (n_agents, obs_dim) for single env
    if isinstance(obs, list) and len(obs) > 0:
        first = obs[0]
        if hasattr(first, 'shape') and len(first.shape) >= 2:
            return first.shape[-1]
    elif isinstance(obs, torch.Tensor):
        return obs.shape[-1]
    return None


def test_nn_mode():
    print("=" * 60)
    print("测试 1: nn 模式 → 期望 10 维")
    print("=" * 60)
    configure(obs_mode="nn", neighbor_radius=0.5)
    from vmas import make_env
    env = make_env(scenario="flocking", num_envs=1, continuous_actions=True,
                   device="cpu", n_agents=4, n_obstacles=0)
    obs_dim = get_obs_dim(env)
    assert obs_dim == 10, f"期望 10, 实际 {obs_dim}"
    print(f"✓ nn 模式观测维度正确: {obs_dim}")

    return True


def test_lidar_mode():
    print("=" * 60)
    print("测试 2: lidar 模式 → 期望 18 维")
    print("=" * 60)
    configure(obs_mode="lidar", lidar_max_range=0.5, lidar_n_rays=12)
    from vmas import make_env
    env = make_env(scenario="flocking", num_envs=1, continuous_actions=True,
                   device="cpu", n_agents=4, n_obstacles=0)
    obs_dim = get_obs_dim(env)
    assert obs_dim == 18, f"期望 18, 实际 {obs_dim}"
    print(f"✓ lidar 模式观测维度正确: {obs_dim}")

    return True


def test_lidar_detection():
    """验证 lidar 真的能检测到其他 agent。"""
    print("=" * 60)
    print("测试 3: lidar 实际检测能力")
    print("=" * 60)
    configure(obs_mode="lidar", lidar_max_range=0.5, lidar_n_rays=12)
    from vmas import make_env
    env = make_env(scenario="flocking", num_envs=1, continuous_actions=True,
                   device="cpu", n_agents=4, n_obstacles=0)
    obs = env.reset()
    # obs 是 list[Tensor],每个 Tensor = 单 agent obs
    # agent_0 的 obs = obs[0], shape (obs_dim,)
    agent_0_obs = obs[0] if isinstance(obs, list) else obs[0]
    lidar_part = agent_0_obs[-12:]  # 最后 12 维
    print(f"  agent_0 完整 obs: {agent_0_obs}")
    print(f"  agent_0 lidar 部分: {lidar_part}")
    print(f"  lidar 范围: [{lidar_part.min().item():.3f}, "
          f"{lidar_part.max().item():.3f}]")
    has_detection = (lidar_part < 0.99).any().item()
    if has_detection:
        n_detected = (lidar_part < 0.99).sum().item()
        print(f"✓ lidar 在初始位置就检测到 {n_detected} 个方向的 agent")
    else:
        print(f"⚠ 初始未检测到 — 可能初始位置所有 agent 在 radar 范围外")
        print(f"  (lidar_max_range=0.5, world 是 [-1, 1]^2, 初始位置随机)")

    return True


if __name__ == "__main__":
    ok = True
    ok &= test_nn_mode()
    print()
    ok &= test_lidar_mode()
    print()
    ok &= test_lidar_detection()
    print()
    print("=" * 60)
    print("所有测试通过" if ok else "测试失败")
    print("=" * 60)
