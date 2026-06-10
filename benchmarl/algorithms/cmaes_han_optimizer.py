import json
import os
import time

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType, set_exploration_type


class CmaesHanOptimizer:
    """CMA-ES optimizer for the ABCD parameters of a HanModel.

    Single-phase training: CMA-ES directly optimizes all ABCD Hebbian
    parameters across every :class:`HanLayer` in the network. The plastic
    weights ``W`` themselves evolve online during evaluation through the
    HAN rule (decoupled ``update_weights`` with sliding-window averaging
    and hard layer-wise max-abs normalization).

    Supports the same fitness modes as :class:`CmaesFullHebbianOptimizer`.
    """

    FITNESS_MODES = [
        "navigation",
        "navigation_avoidance",
        "navigation_v2",
    ]

    def __init__(
        self,
        experiment,
        han_model,
        fitness_mode: str = "navigation_avoidance",
        pop_size: int = 30,
        sigma0: float = 0.5,
        max_gens: int = 100,
        n_eval_episodes: int = 10,
        device: str = "cpu",
        collision_penalty_weight: float = 1.0,
    ):
        self.experiment = experiment
        self.han_model = han_model
        self.fitness_mode = fitness_mode
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.max_gens = max_gens
        self.n_eval_episodes = n_eval_episodes
        self.device = device
        self.collision_penalty_weight = collision_penalty_weight

        self.policy = experiment.policy

        self._initial_abcd = han_model.get_abcd_vector().clone()
        self._best_abcd_so_far = None
        self._best_fitness_so_far = float('inf')
        self._current_gen = 0

    def get_abcd_vector(self) -> np.ndarray:
        return self.han_model.get_abcd_vector().detach().cpu().numpy()

    def set_abcd_from_vector(self, x: np.ndarray):
        self.han_model.set_abcd_from_vector(torch.tensor(x, device=self.device))

    def _compute_fitness(self, episode_reward: float, collision_count: int, total_steps: int,
                         initial_goal_dists: torch.Tensor = None,
                         final_goal_dists: torch.Tensor = None,
                         success: bool = False,
                         ) -> float:
        """Compute fitness value from episode data based on fitness mode."""
        if self.fitness_mode == "navigation":
            return episode_reward
        elif self.fitness_mode == "navigation_avoidance":
            return episode_reward - self.collision_penalty_weight * collision_count
        elif self.fitness_mode == "navigation_v2":
            if initial_goal_dists is None or final_goal_dists is None:
                raise ValueError("navigation_v2 mode requires initial/final goal distances")
            eps = 1e-6
            progress = ((initial_goal_dists - final_goal_dists).clamp(min=0.0)
                        / (initial_goal_dists + eps))
            mask = (initial_goal_dists > eps)
            if mask.any():
                mean_progress = progress[mask].mean().item()
            else:
                mean_progress = 0.0
            success_term = 1.0 if success else 0.0
            mean_final = final_goal_dists.mean().item() if mask.any() else 0.0
            final_term = -mean_final
            w_progress = 3.0
            w_success = 5.0
            w_final = 1.0
            return w_progress * mean_progress + w_success * success_term + w_final * final_term
        else:
            raise ValueError(f"Unknown fitness mode: {self.fitness_mode}")

    def _run_one_episode(self, env, group, max_steps, policy, on_frame=None):
        """Run a single episode and return everything needed by any fitness mode."""
        td = env.reset()

        obs = td.get((group, "observation"))
        initial_goal_dists = torch.linalg.vector_norm(
            obs[..., :2].float(), dim=-1
        )

        episode_reward = 0.0
        collision_count = 0
        done = False
        step = 0
        success = False

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

            done = td.get(("next", "done")).any().item()
            if done:
                success = True
            td = td.get("next")
            step += 1

        final_obs = td.get((group, "observation"))
        final_goal_dists = torch.linalg.vector_norm(
            final_obs[..., :2].float(), dim=-1
        )

        return {
            "episode_reward": episode_reward,
            "collision_count": collision_count,
            "step": step,
            "success": success,
            "initial_goal_dists": initial_goal_dists,
            "final_goal_dists": final_goal_dists,
        }

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness of a candidate ABCD parameter vector."""
        t0 = time.time()
        self.set_abcd_from_vector(x)
        # reset_all_weights also clears the sliding windows and resets ticks.
        self.han_model.reset_all_weights()

        fitnesses = []
        group = list(self.experiment.group_map.keys())[0]
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(self.n_eval_episodes):
                self.han_model.reset_all_weights()
                stats = self._run_one_episode(env, group, max_steps, self.policy)
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"], stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                )
                fitnesses.append(ep_fitness)

        elapsed = time.time() - t0
        self._last_eval_time = elapsed
        self._last_eval_steps = 0

        return -np.mean(fitnesses)

    def run(self) -> np.ndarray:
        """Run CMA-ES optimization."""
        import cma

        x0 = self.get_abcd_vector()
        total_params = len(x0)

        layers = self.han_model.get_all_han_layers()
        layer_info = ", ".join(
            f"L{i}: {l.in_features}x{l.out_features} ({l.num_abcd_params})"
            for i, l in enumerate(layers)
        )

        print(f"CMA-ES: optimizing {total_params} ABCD parameters across {len(layers)} layers")
        print(f"  Layers: [{layer_info}]")
        print(f"  pop_size={self.pop_size}, sigma0={self.sigma0}, max_gens={self.max_gens}")
        print(f"  fitness_mode={self.fitness_mode}, n_eval_episodes={self.n_eval_episodes}")
        print(f"  HanModel: window_size={self.han_model.layers[0].window_size}, "
              f"f_nn={self.han_model.f_nn}, f_hebb={self.han_model.f_hebb}, "
              f"update_interval={self.han_model._update_interval}")

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
                self._best_abcd_so_far = es.result.xbest.copy()

            print(
                f"  Gen {gen}/{self.max_gens}: "
                f"best={-best_fitness:.2f}, mean={-mean_fitness:.2f} | "
                f"gen_time={gen_time:.1f}s | "
                f"elapsed={elapsed_total:.0f}s, ETA={remaining:.0f}s"
            )

        result = es.result
        best_x = result.xbest
        best_fitness = -result.fbest

        if self._best_abcd_so_far is not None and self._best_fitness_so_far < result.fbest:
            best_x = self._best_abcd_so_far
            best_fitness = -self._best_fitness_so_far

        total_time = time.time() - total_start
        print(f"\nCMA-ES finished in {total_time:.1f}s ({total_time/60:.1f}min):")
        print(f"  Best fitness: {best_fitness:.2f}")
        print(f"  Best ABCD norm: {np.linalg.norm(best_x):.4f}")

        self.set_abcd_from_vector(best_x)
        self.han_model.reset_all_weights()

        return best_x

    def plot_convergence(self, output_dir: str):
        """Generate CMA-ES convergence plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fit_path = os.path.join(os.getcwd(), "outcmaes", "fit.dat")
        if not os.path.exists(fit_path):
            print(f"  Skipping convergence plot: {fit_path} not found")
            return

        data = np.loadtxt(fit_path, comments="%")
        gen = data[:, 0]
        bestever = data[:, 4]
        best = data[:, 5]
        median = data[:, 6]
        worst = data[:, 7]
        sigma = data[:, 2]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(gen, -best, label="Best", color="blue")
        ax1.plot(gen, -bestever, label="Best Ever", color="green", linestyle="--")
        ax1.plot(gen, -median, label="Median", color="orange", alpha=0.7)
        ax1.fill_between(gen, -worst, -best, alpha=0.15, color="blue")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        ax1.set_title(f"CMA-ES Convergence (HAN, mode={self.fitness_mode})")
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
        if self._best_abcd_so_far is not None:
            self.set_abcd_from_vector(self._best_abcd_so_far)
            self.han_model.reset_all_weights()
            return self._best_abcd_so_far
        return None

    def save(self, output_dir: str):
        """Save optimization results."""
        abcd_dir = os.path.join(output_dir, "han_results")
        os.makedirs(abcd_dir, exist_ok=True)

        best_abcd = self._best_abcd_so_far if self._best_abcd_so_far is not None else self.get_abcd_vector()
        np.save(os.path.join(abcd_dir, "abcd_params.npy"), best_abcd)

        layers = self.han_model.get_all_han_layers()
        for i, layer in enumerate(layers):
            np.save(os.path.join(abcd_dir, f"layer{i}_abcd.npy"),
                    layer.get_abcd_vector().detach().cpu().numpy())

        torch.save(self.policy.state_dict(), os.path.join(abcd_dir, "policy_state.pt"))

        metadata = {
            "fitness_mode": self.fitness_mode,
            "total_abcd_params": self.han_model.total_abcd_params,
            "n_layers": len(layers),
            "layer_shapes": [
                {"in": l.in_features, "out": l.out_features, "n_abcd": l.num_abcd_params}
                for l in layers
            ],
            "window_size": layers[0].window_size if layers else None,
            "f_nn": self.han_model.f_nn,
            "f_hebb": self.han_model.f_hebb,
            "update_interval": self.han_model._update_interval,
            "best_fitness": float(-self._best_fitness_so_far) if self._best_fitness_so_far != float('inf') else None,
            "generations_completed": self._current_gen,
            "max_generations": self.max_gens,
        }
        with open(os.path.join(abcd_dir, "results.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  HAN results saved to: {abcd_dir}/")
        return abcd_dir

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

        video_dir = os.path.join(output_dir, "videos_han")
        os.makedirs(video_dir, exist_ok=True)

        print(f"  Video output dir: {video_dir}")
        print(f"  max_steps={max_steps}, n_episodes={n_episodes}, fps={fps}, "
              f"max_video_frames={max_video_frames}")

        all_rewards = []
        all_fitnesses = []

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(n_episodes):
                self.han_model.reset_all_weights()
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
                                        torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                    )
                            except Exception as e:
                                frame_errors.append(f"ep{ep} step{_step}: {e}")
                else:
                    def on_frame(_td, _step):
                        try:
                            frame = env.render(mode="rgb_array")
                            if frame is not None:
                                frames.append(
                                    torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0)
                                )
                        except Exception as e:
                            frame_errors.append(f"ep{ep} step{_step}: {e}")

                try:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0))
                except Exception as e:
                    frame_errors.append(f"ep{ep} reset: {e}")

                stats = self._run_one_episode(env, group, max_steps, self.policy, on_frame=on_frame)

                all_rewards.append(stats["episode_reward"])
                ep_fitness = self._compute_fitness(
                    stats["episode_reward"], stats["collision_count"], stats["step"],
                    initial_goal_dists=stats["initial_goal_dists"],
                    final_goal_dists=stats["final_goal_dists"],
                    success=stats["success"],
                )
                all_fitnesses.append(ep_fitness)

                if frames:
                    vid = torch.cat(frames, dim=0).unsqueeze(0)
                    for idx in (-1, -2):
                        if vid.shape[idx] % 2 != 0:
                            vid = vid.index_select(idx, torch.arange(1, vid.shape[idx]))
                    video_path = os.path.join(video_dir, f"eval_han_{ep}.mp4")
                    vid_rgb = vid[0].permute(0, 2, 3, 1)
                    torchvision.io.write_video(video_path, vid_rgb.numpy(), fps=fps)
                    print(f"  Saved ep{ep}: {len(frames)} frames, reward={stats['episode_reward']:.2f}, fitness={ep_fitness:.2f}")
                else:
                    print(f"  SKIP ep{ep}: no frames, reward={stats['episode_reward']:.2f}")

        print(f"\nEvaluation ({n_episodes} episodes, mode={self.fitness_mode}):")
        print(f"  Mean reward: {np.mean(all_rewards):.2f}")
        print(f"  Mean fitness: {np.mean(all_fitnesses):.2f}")
        print(f"  Videos saved to: {video_dir}")

        return np.mean(all_fitnesses)
