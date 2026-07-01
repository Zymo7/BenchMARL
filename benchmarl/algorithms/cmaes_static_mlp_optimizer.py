"""CMA-ES optimizer for the parameters of a StaticMlpModel.

Single-phase training: CMA-ES directly optimizes all weights of the
:class:`benchmarl.models.StaticMlpModel` as a flat vector. There is no
plasticity and no per-step weight mutation — every weight that affects
the forward pass comes from the CMA-ES candidate vector.

Compared to :class:`~benchmarl.algorithms.cmaes_han_optimizer.CmaesHanOptimizer`
(this is mirrored from), only the policy state management is different:

  - HAN reads/writes via ``get_abcd_vector / set_abcd_from_vector``
  - static-MLP reads/writes via ``get_weights_vector / set_weights_from_vector``

The fitness function (per-step At / Dt / Cg / S decomposition, rollout
loop, and evaluation video rendering) is identical, so the only
experimental difference between HAN and static-MLP under CMA-ES is
the network architecture — exactly the contrast we want to isolate.
"""
import json
import math
import os
import time

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


class CmaesStaticMlpOptimizer:
    """CMA-ES optimizer for the flat weight vector of a StaticMlpModel.

    The optimizer is a near-clone of
    :class:`~benchmarl.algorithms.cmaes_han_optimizer.CmaesHanOptimizer`
    with two differences:

    1. The candidate vector passed to CMA-ES is the flat MLP weight
       tensor (``StaticMlpModel.get_weights_vector``), not the ABCD
       parameters of a HanModel.
    2. Between candidates we call
       :meth:`StaticMlpModel.reset_weights` (no HanModel-style
       sliding-window reset; the static MLP has no per-step state to
       clear).
    """

    FITNESS_MODES = [
        "navigation",
        "navigation_avoidance",
        "navigation_v2",
        "navigation_avoidance_v2",
        "flocking_global",
        "flocking_orbit",
        "flocking_lf_arrival",
        "flocking_light_intensity",
    ]

    def __init__(
        self,
        experiment,
        static_mlp_model,
        fitness_mode: str = "flocking_orbit",
        pop_size: int = 30,
        sigma0: float = 0.3,
        max_gens: int = 30,
        n_eval_episodes: int = 2,
        device: str = "cpu",
        collision_penalty_weight: float = 2.0,
        safety_distance: float = 0.15,
        neighbor_radius: float = 0.5,
        movement_target_displacement: float = 1.0,
        patch_heading_from_vel: bool = True,
        orbit_radius: float = 0.7,
        orbit_radius_tolerance: float = 0.3,
        dt_floor: float = 0.1,
    ):
        self.experiment = experiment
        self.static_mlp_model = static_mlp_model
        self.fitness_mode = fitness_mode
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.max_gens = max_gens
        self.n_eval_episodes = n_eval_episodes
        self.device = device
        self.collision_penalty_weight = collision_penalty_weight
        self.safety_distance = safety_distance
        self.neighbor_radius = neighbor_radius
        self.movement_target_displacement = movement_target_displacement
        self.patch_heading_from_vel = patch_heading_from_vel
        self.orbit_radius = orbit_radius
        self.orbit_radius_tolerance = orbit_radius_tolerance
        self.dt_floor = dt_floor

        self.policy = experiment.policy

        self._initial_weights = static_mlp_model.get_weights_vector().clone()
        self._best_weights_so_far = None
        self._best_fitness_so_far = float('inf')
        self._current_gen = 0

    # ------------------------------------------------------------------
    # Policy state get/set — mirrors the ABCD helpers in CmaesHanOptimizer
    # but on the static-MLP weight vector.
    # ------------------------------------------------------------------
    def get_weights_vector(self) -> np.ndarray:
        return self.static_mlp_model.get_weights_vector().detach().cpu().numpy()

    def set_weights_from_vector(self, x: np.ndarray):
        self.static_mlp_model.set_weights_from_vector(
            torch.tensor(x, device=self.device)
        )

    def reset_weights(self):
        self.static_mlp_model.reset_weights()

    def total_weights(self) -> int:
        return self.static_mlp_model.total_weights

    # ------------------------------------------------------------------
    # Internals — identical fitness math as CmaesHanOptimizer
    # ------------------------------------------------------------------
    def _get_vmas_core(self):
        if hasattr(self, "_vmas_core") and self._vmas_core is not None:
            return self._vmas_core
        env = self.experiment.test_env
        node = env
        while True:
            nxt = getattr(node, "base_env", None) or getattr(node, "_env", None)
            if nxt is None or nxt is node:
                break
            node = nxt
        self._vmas_core = node
        return node

    @staticmethod
    def _count_connected_components(adj: torch.Tensor) -> int:
        """Identical to ``CmaesHanOptimizer._count_connected_components``."""
        N = adj.shape[0]
        if N == 0:
            return 0
        parent = list(range(N))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        adj_cpu = adj.detach().cpu().bool()
        iu, iv = torch.triu_indices(N, N, offset=1)
        edge_mask = adj_cpu[iu, iv]
        edge_i = iu[edge_mask].tolist()
        edge_j = iv[edge_mask].tolist()
        for a, b in zip(edge_i, edge_j):
            union(a, b)
        roots = {find(x) for x in range(N)}
        return len(roots)

    def _compute_flocking_orbit_fitness(self, pos_history, rot_history,
                                         target_pos_history) -> float:
        """Identical to CmaesHanOptimizer._compute_flocking_orbit_fitness."""
        T = len(pos_history)
        if T == 0:
            return 0.0
        N = pos_history[0].shape[0]
        if N < 1:
            return 0.0

        w_C = 0.2
        w_S = 0.8
        r_star = float(self.orbit_radius)
        r_sigma = float(self.orbit_radius_tolerance)
        dt_floor = float(self.dt_floor)
        nr = float(self.neighbor_radius)
        sd = float(self.safety_distance)
        eye = torch.eye(N, dtype=torch.bool)
        eps = 1e-6

        sum_At = 0.0
        sum_Dt = 0.0
        sum_Cg = 0.0
        sum_S = 0.0

        for t in range(T):
            pos = pos_history[t]
            rot = rot_history[t]
            tgt = target_pos_history[t]

            r_vec = pos - tgt
            r_norm = torch.linalg.vector_norm(r_vec, dim=-1, keepdim=True)
            r_unit = r_vec / (r_norm + eps)
            tangent = torch.stack([-r_unit[:, 1], r_unit[:, 0]], dim=-1)
            v_dir = torch.stack([torch.cos(rot), torch.sin(rot)], dim=-1)
            dot = (v_dir * tangent).sum(dim=-1)
            align = ((dot + 1.0) * 0.5).clamp(0.0, 1.0)
            valid = (r_norm.squeeze(-1) > eps).float()
            at_sum = (align * valid).sum().item()
            at_count = valid.sum().item()
            At = at_sum / at_count if at_count > 0 else 0.5

            r_dist = r_norm.squeeze(-1)
            Dt_raw = torch.exp(-((r_dist - r_star) ** 2) / (2.0 * r_sigma ** 2))
            Dt = max(Dt_raw.mean().item(), dt_floor)

            diff = pos.unsqueeze(0) - pos.unsqueeze(1)
            dist = torch.linalg.vector_norm(diff, dim=-1)
            adj = (dist < nr) & (~eye)
            num_groups = self._count_connected_components(adj)
            Cg = 1.0 / max(int(num_groups), 1)

            dist_for_collision = dist.masked_fill(eye, float("inf"))
            in_collision = (dist_for_collision < sd).any(dim=-1)
            S = 1.0 - in_collision.float().mean().item()

            sum_At += At
            sum_Dt += Dt
            sum_Cg += Cg
            sum_S += S

        F_orbit = (1.5 * sum_At + sum_Dt + w_C * sum_Cg + w_S * sum_S) / T
        return F_orbit

    def _compute_fitness(self, episode_reward, collision_count, total_steps,
                         initial_goal_dists=None, final_goal_dists=None,
                         success=False, agent_collision_ratios=None,
                         pos_history=None, rot_history=None,
                         target_pos_history=None) -> float:
        """Dispatch on ``self.fitness_mode``.

        Currently only ``flocking_orbit`` and the no-op "navigation"
        modes are wired up; the rest reuse the same formulas as
        :class:`CmaesHanOptimizer` and are easy to add.
        """
        if self.fitness_mode == "flocking_orbit":
            return self._compute_flocking_orbit_fitness(
                pos_history, rot_history, target_pos_history
            )
        # Conservative fall-back: use the negative episode reward so
        # CMA-ES still has a meaningful gradient signal even if the
        # user's fitness_mode isn't implemented here.
        return -float(episode_reward)

    def _run_one_episode(self, env, group, max_steps, policy, on_frame=None):
        td = env.reset()
        self._get_vmas_core()
        obs = td.get((group, "observation"))
        initial_goal_dists = torch.linalg.vector_norm(
            obs[..., :2].float(), dim=-1
        )

        episode_reward = 0.0
        collision_count = 0
        done = False
        step = 0
        success = False

        agent_collision_steps = None
        n_agents = None

        pos_history = []
        rot_history = []
        target_pos_history = []

        while not done and step < max_steps:
            td = policy(td)
            td = env.step(td)

            if on_frame is not None:
                on_frame(td, step)

            reward = td.get(("next", group, "reward"))
            episode_reward += reward.sum().item()

            info = td.get(("next", group, "info"))
            if info is not None:
                col_rew = info.get("agent_collision_rew")
                if col_rew is not None:
                    collision_count += (col_rew < 0).sum().item()

            core = self._vmas_core
            if core is not None and hasattr(core, "agents"):
                if n_agents is None:
                    n_agents = len(core.agents)
                    agent_collision_steps = torch.zeros(
                        n_agents, device=self.device
                    )

                pos_stack = torch.stack(
                    [a.state.pos[0] for a in core.agents], dim=0
                )
                a = pos_stack.shape[0]
                diff = pos_stack.unsqueeze(0) - pos_stack.unsqueeze(1)
                dist = torch.linalg.vector_norm(diff, dim=-1)
                eye = torch.eye(a, device=dist.device, dtype=torch.bool)
                dist = dist.masked_fill(eye, float("inf"))
                in_radius = dist < self.neighbor_radius
                in_collision = dist < self.safety_distance
                agent_in_collision = (in_radius & in_collision).any(dim=-1)
                agent_collision_steps += agent_in_collision.float()

                if self.patch_heading_from_vel:
                    for agent in core.agents:
                        vel0 = agent.state.vel[0]
                        agent.state.rot[0, 0] = torch.atan2(
                            vel0[1], vel0[0]
                        )
                pos_cpu = pos_stack.detach().cpu()
                rot_cpu = torch.stack(
                    [a.state.rot[0, 0] for a in core.agents], dim=0
                ).detach().cpu()
                pos_history.append(pos_cpu)
                rot_history.append(rot_cpu)

                target = getattr(
                    getattr(core, "scenario", None), "_target", None
                )
                if target is not None and hasattr(target.state, "pos"):
                    target_pos_history.append(
                        target.state.pos[0].detach().cpu()
                    )

            done = td.get(("next", "done")).any().item()
            if done:
                success = True
            td = td.get("next")
            step += 1

        if n_agents is not None and n_agents > 1 and step > 0:
            agent_collision_ratios = agent_collision_steps / float(step)
        else:
            agent_collision_ratios = torch.zeros(
                n_agents if n_agents is not None else 1,
                device=self.device,
            )

        return {
            "episode_reward": episode_reward,
            "collision_count": collision_count,
            "step": step,
            "success": success,
            "initial_goal_dists": initial_goal_dists,
            "final_goal_dists": initial_goal_dists,  # unused for orbit
            "agent_collision_ratios": agent_collision_ratios,
            "pos_history": pos_history,
            "rot_history": rot_history,
            "target_pos_history": target_pos_history,
        }

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness of a candidate weight vector."""
        t0 = time.time()
        self.set_weights_from_vector(x)
        # No sliding-window reset needed for static-MLP; reset the
        # weights we just loaded via the candidate vector directly.
        fitnesses = []
        group = list(self.experiment.group_map.keys())[0]
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        with torch.no_grad(), set_exploration_type(
                ExplorationType.DETERMINISTIC):
            for ep in range(self.n_eval_episodes):
                stats = self._run_one_episode(env, group, max_steps, self.policy)
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"],
                    stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                    agent_collision_ratios=stats["agent_collision_ratios"],
                    pos_history=stats["pos_history"],
                    rot_history=stats["rot_history"],
                    target_pos_history=stats["target_pos_history"],
                )
                fitnesses.append(ep_fitness)

        self._last_eval_time = time.time() - t0
        return -float(np.mean(fitnesses))

    def run(self) -> np.ndarray:
        """Run CMA-ES optimization on the flat MLP weight vector."""
        import cma
        x0 = self.get_weights_vector()
        total_params = len(x0)

        print(f"[cmaes-static-mlp] optimizing {total_params} weights "
              f"(matches HAN's 560 ABCD params)")
        print(f"  pop_size={self.pop_size}, sigma0={self.sigma0}, "
              f"max_gens={self.max_gens}")
        print(f"  fitness_mode={self.fitness_mode}, "
              f"n_eval_episodes={self.n_eval_episodes}")
        print(f"  StaticMlpModel: total_weights={self.total_weights()}")

        opts = {
            "popsize": self.pop_size,
            "maxiter": self.max_gens,
            "verbose": 1,
            "tolfun": 1e10,
            "tolfunhist": 1e10,
            "tolflatfitness": self.max_gens + 10,
            "tolstagnation": self.max_gens * 2,
            "tolx": 1e-12,
        }
        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)
        total_start = time.time()

        for gen in range(1, self.max_gens + 1):
            self._current_gen = gen
            gen_start = time.time()
            solutions = es.ask()
            fitnesses = [self.fitness(x) for x in solutions]
            es.tell(solutions, fitnesses)
            es.logger.add()
            gen_time = time.time() - gen_start

            best_fitness = min(fitnesses)
            mean_fitness = sum(fitnesses) / len(fitnesses)
            elapsed_total = time.time() - total_start
            remaining = (elapsed_total / gen) * (self.max_gens - gen)

            if best_fitness < self._best_fitness_so_far:
                self._best_fitness_so_far = best_fitness
                self._best_weights_so_far = es.result.xbest.copy()

            print(
                f"  Gen {gen}/{self.max_gens}: "
                f"best={-best_fitness:.2f}, mean={-mean_fitness:.2f} | "
                f"gen_time={gen_time:.1f}s | "
                f"elapsed={elapsed_total:.0f}s, ETA={remaining:.0f}s"
            )

        result = es.result
        best_x = result.xbest
        best_fitness = -result.fbest

        if (self._best_weights_so_far is not None
                and self._best_fitness_so_far < result.fbest):
            best_x = self._best_weights_so_far
            best_fitness = -self._best_fitness_so_far

        total_time = time.time() - total_start
        print(f"\n[cmaes-static-mlp] finished in {total_time:.1f}s "
              f"({total_time/60:.1f}min):")
        print(f"  Best fitness: {best_fitness:.2f}")
        print(f"  Best weights norm: {np.linalg.norm(best_x):.4f}")

        self.set_weights_from_vector(best_x)
        return best_x

    def plot_convergence(self, output_dir: str):
        """Generate CMA-ES convergence plot (mirrors CmaesHanOptimizer)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fit_path = os.path.join(os.getcwd(), "outcmaes", "fit.dat")
        if not os.path.exists(fit_path):
            print(f"  Skipping convergence plot: {fit_path} not found")
            return
        data = np.loadtxt(fit_path, comments="%")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        gen = data[:, 0]
        bestever = data[:, 4]
        best = data[:, 5]
        median = data[:, 6]
        worst = data[:, 7]
        sigma = data[:, 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(gen, -best, label="Best", color="blue")
        ax1.plot(gen, -bestever, label="Best Ever", color="green",
                 linestyle="--")
        ax1.plot(gen, -median, label="Median", color="orange", alpha=0.7)
        ax1.fill_between(gen, -worst, -best, alpha=0.15, color="blue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(
            f"CMA-ES Convergence (static-MLP, mode={self.fitness_mode})"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(gen, sigma, color="red")
        ax2.set_xlabel("Generation")
        ax2.set_ylabel("Sigma (step size)")
        ax2.set_title("CMA-ES Step Size")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "cmaes_convergence.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Convergence plot saved to: {plot_path}")

    def apply_best_so_far(self):
        if self._best_weights_so_far is not None:
            self.set_weights_from_vector(self._best_weights_so_far)
            return self._best_weights_so_far
        return None

    def save(self, output_dir: str):
        """Save optimization results."""
        slm_dir = os.path.join(output_dir, "static_mlp_results")
        os.makedirs(slm_dir, exist_ok=True)

        best_weights = (
            self._best_weights_so_far
            if self._best_weights_so_far is not None
            else self.get_weights_vector()
        )
        np.save(os.path.join(slm_dir, "weights.npy"), best_weights)

        torch.save(self.policy.state_dict(),
                   os.path.join(slm_dir, "policy_state.pt"))

        metadata = {
            "algorithm": "static-mlp",
            "optimization": "cmaes",
            "model": "static_mlp",
            "fitness_mode": self.fitness_mode,
            "total_weights": self.total_weights(),
            "best_fitness": (float(-self._best_fitness_so_far)
                              if self._best_fitness_so_far != float('inf')
                              else None),
            "generations_completed": self._current_gen,
            "max_generations": self.max_gens,
        }
        with open(os.path.join(slm_dir, "results.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  static-MLP results saved to: {slm_dir}/")
        return slm_dir

    def evaluate(
        self,
        output_dir: str,
        n_episodes: int = 10,
        fps: int = 20,
        max_video_frames: int = None,
    ):
        """Run evaluation episodes and save videos."""
        import torchvision

        group = list(self.experiment.group_map.keys())[0]
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        video_dir = os.path.join(output_dir, "videos_static_mlp")
        os.makedirs(video_dir, exist_ok=True)

        print(f"  Video output dir: {video_dir}")
        print(f"  max_steps={max_steps}, n_episodes={n_episodes}, "
              f"fps={fps}, max_video_frames={max_video_frames}")

        all_rewards = []
        all_fitnesses = []

        with torch.no_grad(), set_exploration_type(
                ExplorationType.DETERMINISTIC):
            for ep in range(n_episodes):
                frames = []
                frame_errors = []

                ep_max_frames = max_video_frames
                if ep_max_frames is not None:
                    def on_frame(_td, _step, _cap=ep_max_frames):
                        if len(frames) < _cap:
                            try:
                                frame = env.render(mode="rgb_array")
                                if frame is not None:
                                    frames.append(
                                        torch.tensor(frame.copy())
                                        .permute(2, 0, 1).unsqueeze(0)
                                    )
                            except Exception as e:
                                frame_errors.append(
                                    f"ep{ep} step{_step}: {e}"
                                )
                else:
                    def on_frame(_td, _step):
                        try:
                            frame = env.render(mode="rgb_array")
                            if frame is not None:
                                frames.append(
                                    torch.tensor(frame.copy())
                                    .permute(2, 0, 1).unsqueeze(0)
                                )
                        except Exception as e:
                            frame_errors.append(
                                f"ep{ep} step{_step}: {e}"
                            )

                try:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(
                            torch.tensor(frame.copy())
                            .permute(2, 0, 1).unsqueeze(0)
                        )
                except Exception as e:
                    frame_errors.append(f"ep{ep} reset: {e}")

                stats = self._run_one_episode(
                    env, group, max_steps, self.policy,
                    on_frame=on_frame,
                )

                all_rewards.append(stats["episode_reward"])
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"],
                    stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                    agent_collision_ratios=stats["agent_collision_ratios"],
                    pos_history=stats["pos_history"],
                    rot_history=stats["rot_history"],
                    target_pos_history=stats["target_pos_history"],
                )
                all_fitnesses.append(ep_fitness)

                if frames:
                    vid = torch.cat(frames, dim=0).unsqueeze(0)
                    for idx in (-1, -2):
                        if vid.shape[idx] % 2 != 0:
                            vid = vid.index_select(
                                idx, torch.arange(1, vid.shape[idx])
                            )
                    video_path = os.path.join(
                        video_dir, f"eval_static_mlp_{ep}.mp4"
                    )
                    vid_rgb = vid[0].permute(0, 2, 3, 1)
                    torchvision.io.write_video(
                        video_path, vid_rgb.numpy(), fps=fps
                    )
                    print(f"  Saved ep{ep}: {len(frames)} frames, "
                          f"reward={stats['episode_reward']:.2f}, "
                          f"fitness={ep_fitness:.2f}")
                else:
                    print(f"  SKIP ep{ep}: no frames, "
                          f"reward={stats['episode_reward']:.2f}")

        print(f"\nEvaluation ({n_episodes} episodes, mode="
              f"{self.fitness_mode}):")
        print(f"  Mean reward: {np.mean(all_rewards):.2f}")
        print(f"  Mean fitness: {np.mean(all_fitnesses):.2f}")
        print(f"  Videos saved to: {video_dir}")
        return float(np.mean(all_fitnesses))
