import json
import os
import time

import numpy as np
import torch
from tensordict.nn import TensorDictSequential
from torchrl.envs.utils import ExplorationType, set_exploration_type

from benchmarl.models.common import SequenceModel


class CmaesHebbianOptimizer:
    """CMA-ES optimizer for the ABCD parameters of the Hebbian layer.

    Phase 2 of training: freezes the MLP layers and uses CMA-ES
    to find optimal ABCD parameters that maximize episode rewards
    through online Hebbian weight adaptation.
    """

    def __init__(
        self,
        experiment,
        hebbian_layer,
        env_func=None,
        pop_size: int = 20,
        sigma0: float = 0.5,
        max_gens: int = 100,
        n_eval_episodes: int = 10,
        device: str = "cpu",
    ):
        self.experiment = experiment
        self.hebbian_layer = hebbian_layer
        self.pop_size = pop_size
        self.sigma0 = sigma0
        self.max_gens = max_gens
        self.n_eval_episodes = n_eval_episodes
        self.device = device

        # Use test_env from experiment for evaluation
        if env_func is not None:
            self.env_func = env_func
        else:
            self.env_func = None

        # Get the policy for evaluation
        self.policy = experiment.policy

        # Store initial ABCD vector
        self._initial_abcd = hebbian_layer.get_abcd_vector().clone()

        # Track best solution found so far (for mid-interrupt recovery)
        self._best_abcd_so_far = None
        self._best_fitness_so_far = float('inf')
        self._current_gen = 0

    def _get_policy_module(self):
        """Get the inner module from the policy for action extraction."""
        # policy is a TensorDictSequential of ProbabilisticActors per group
        return self.policy

    def get_abcd_vector(self) -> np.ndarray:
        return self.hebbian_layer.get_abcd_vector().detach().cpu().numpy()

    def set_abcd_from_vector(self, x: np.ndarray):
        self.hebbian_layer.set_abcd_from_vector(torch.tensor(x, device=self.device))

    def freeze_mlp_layers(self):
        """Freeze all parameters except the Hebbian layer buffers."""
        for name, param in self.policy.named_parameters():
            param.requires_grad = False

    def fitness(self, x: np.ndarray) -> float:
        """Evaluate fitness of a candidate ABCD parameter vector.

        1. Set ABCD params from x
        2. Reset Hebbian weights W to initial values
        3. Run n_eval_episodes using deterministic policy
        4. Return negative mean total reward (CMA-ES minimizes)
        """
        t0 = time.time()
        self.set_abcd_from_vector(x)
        self.hebbian_layer.reset_weights()

        total_rewards = []
        group = list(self.experiment.group_map.keys())[0]

        env = self.experiment.test_env
        total_steps = 0

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(self.n_eval_episodes):
                td = env.reset()

                episode_reward = 0.0
                done = False
                step = 0

                while not done and step < self.experiment.max_steps:
                    td = self.policy(td)
                    td = env.step(td)

                    reward = td.get(("next", group, "reward"))
                    episode_reward += reward.sum().item()

                    done = td.get(("next", "done")).any().item()
                    td = td.get("next")
                    step += 1

                total_rewards.append(episode_reward)
                total_steps += step

        elapsed = time.time() - t0
        self._last_eval_time = elapsed
        self._last_eval_steps = total_steps

        return -np.mean(total_rewards)

    def run(self) -> np.ndarray:
        """Run CMA-ES optimization.

        Returns the best ABCD parameter vector found.
        """
        import cma

        self.freeze_mlp_layers()

        x0 = self.get_abcd_vector()
        print(f"CMA-ES: optimizing {len(x0)} ABCD parameters")
        print(f"  pop_size={self.pop_size}, sigma0={self.sigma0}, max_gens={self.max_gens}")
        print(f"  n_eval_episodes={self.n_eval_episodes}")

        opts = {
            "popsize": self.pop_size,
            "maxiter": self.max_gens,
            "verbose": 1,
            # Disable early stopping: fitness is often flat at ~0 in early phases
            "tolfun": 1e10,
            "tolfunhist": 1e10,
            "tolflatfitness": self.max_gens + 10,
            "tolstagnation": self.max_gens * 2,
            "tolx": 1e-12,
        }

        es = cma.CMAEvolutionStrategy(x0, self.sigma0, opts)

        total_cmaes_start = time.time()

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
            avg_eval_time = getattr(self, '_last_eval_time', 0) / max(1, len(solutions))
            total_steps = getattr(self, '_last_eval_steps', 0)
            elapsed_total = time.time() - total_cmaes_start
            remaining = (elapsed_total / gen) * (self.max_gens - gen)

            # Track best solution for mid-interrupt recovery
            if best_fitness < self._best_fitness_so_far:
                self._best_fitness_so_far = best_fitness
                self._best_abcd_so_far = es.result.xbest.copy()

            print(
                f"  Gen {gen}/{self.max_gens}: "
                f"best={-best_fitness:.2f}, mean={-mean_fitness:.2f} | "
                f"gen_time={gen_time:.1f}s, avg_eval={avg_eval_time:.2f}s, "
                f"steps={total_steps} | "
                f"elapsed={elapsed_total:.0f}s, ETA={remaining:.0f}s"
            )

        result = es.result
        best_x = result.xbest
        best_fitness = -result.fbest

        # Use tracked best if it's better (handles mid-interrupt case)
        if self._best_abcd_so_far is not None and self._best_fitness_so_far < best_fitness:
            best_x = self._best_abcd_so_far
            best_fitness = -self._best_fitness_so_far

        total_time = time.time() - total_cmaes_start
        print(f"\nCMA-ES finished in {total_time:.1f}s ({total_time/60:.1f}min):")
        print(f"  Best fitness: {best_fitness:.2f}")
        print(f"  Best ABCD norm: {np.linalg.norm(best_x):.4f}")

        # Set the best parameters
        self.set_abcd_from_vector(best_x)
        self.hebbian_layer.reset_weights()

        return best_x

    def get_best_abcd_so_far(self):
        """Return the best ABCD vector found so far (useful after interrupt)."""
        return self._best_abcd_so_far

    def apply_best_so_far(self):
        """Apply the best ABCD found so far to the Hebbian layer for evaluation."""
        if self._best_abcd_so_far is not None:
            self.set_abcd_from_vector(self._best_abcd_so_far)
            self.hebbian_layer.reset_weights()
            return self._best_abcd_so_far
        return None

    def save(self, output_dir: str):
        """Save the Hebbian ABCD results to disk.

        Saves:
        - abcd_params.npy: The best ABCD parameter vector
        - hebbian_state.pt: The full HebbianLayer state dict
        - results.json: Metadata (fitness, generations, etc.)

        Args:
            output_dir: Directory to save results (e.g. experiment_folder)
        """
        abcd_dir = os.path.join(output_dir, "hebbian_results")
        os.makedirs(abcd_dir, exist_ok=True)

        # Save ABCD vector as numpy
        abcd_path = os.path.join(abcd_dir, "abcd_params.npy")
        best_abcd = self._best_abcd_so_far if self._best_abcd_so_far is not None else self.get_abcd_vector()
        np.save(abcd_path, best_abcd)

        # Save HebbianLayer state dict
        state_path = os.path.join(abcd_dir, "hebbian_state.pt")
        torch.save(self.hebbian_layer.state_dict(), state_path)

        # Save metadata
        metadata = {
            "in_features": self.hebbian_layer.in_features,
            "out_features": self.hebbian_layer.out_features,
            "lr_hebb": self.hebbian_layer.lr_hebb,
            "best_fitness": float(-self._best_fitness_so_far) if self._best_fitness_so_far != float('inf') else None,
            "generations_completed": self._current_gen,
            "max_generations": self.max_gens,
        }
        meta_path = os.path.join(abcd_dir, "results.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Hebbian results saved to: {abcd_dir}/")
        return abcd_dir

    @staticmethod
    def load(output_dir: str, experiment, hebbian_layer):
        """Load saved Hebbian results and apply to a HebbianLayer.

        Args:
            output_dir: Directory containing hebbian_results/ (e.g. experiment_folder)
            experiment: The experiment object
            hebbian_layer: The HebbianLayer to load parameters into

        Returns:
            CmaesHebbianOptimizer with loaded state
        """
        abcd_dir = os.path.join(output_dir, "hebbian_results")

        # Load ABCD vector
        abcd_path = os.path.join(abcd_dir, "abcd_params.npy")
        abcd = np.load(abcd_path)

        # Load metadata
        meta_path = os.path.join(abcd_dir, "results.json")
        with open(meta_path, "r") as f:
            metadata = json.load(f)

        # Apply to hebbian layer
        hebbian_layer.set_abcd_from_vector(torch.tensor(abcd, device=experiment.config.train_device))
        hebbian_layer.reset_weights()

        # Create optimizer with loaded state
        optimizer = CmaesHebbianOptimizer(
            experiment=experiment,
            hebbian_layer=hebbian_layer,
            pop_size=20,
            sigma0=0.5,
            max_gens=metadata.get("max_generations", 50),
            n_eval_episodes=10,
            device=experiment.config.train_device,
        )
        optimizer._best_abcd_so_far = abcd
        optimizer._best_fitness_so_far = -metadata["best_fitness"] if metadata["best_fitness"] else float('inf')
        optimizer._current_gen = metadata.get("generations_completed", 0)

        print(f"  Loaded Hebbian results from: {abcd_dir}/")
        print(f"    fitness={metadata['best_fitness']}, generations={optimizer._current_gen}")
        return optimizer

    def evaluate(self, output_dir: str, n_episodes: int = 10):
        """Run evaluation episodes with the trained Hebbian policy and save video.

        Args:
            output_dir: Directory to save the video (e.g. experiment_folder/videos_hebbian)
            n_episodes: Number of evaluation episodes
        """
        import os
        import traceback

        import torchvision

        group = list(self.experiment.group_map.keys())[0]
        env = self.experiment.test_env
        max_steps = self.experiment.max_steps

        # Create output directory
        video_dir = os.path.join(output_dir, "videos_hebbian")
        os.makedirs(video_dir, exist_ok=True)

        print(f"  Video output dir: {video_dir}")
        print(f"  max_steps={max_steps}, n_episodes={n_episodes}")

        all_rewards = []

        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            for ep in range(n_episodes):
                self.hebbian_layer.reset_weights()
                frames = []
                frame_errors = []

                td = env.reset()

                # Capture initial frame (before any action)
                try:
                    frame = env.render(mode="rgb_array")
                    if frame is not None:
                        frames.append(torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0))
                    else:
                        frame_errors.append(f"ep{ep} reset: frame is None")
                except Exception as e:
                    frame_errors.append(f"ep{ep} reset: {type(e).__name__}: {e}")

                episode_reward = 0.0
                done = False
                step = 0

                while not done and step < max_steps:
                    td = self.policy(td)
                    td = env.step(td)

                    # Capture frame after step
                    try:
                        frame = env.render(mode="rgb_array")
                        if frame is not None:
                            frames.append(torch.tensor(frame.copy()).permute(2, 0, 1).unsqueeze(0))
                        else:
                            frame_errors.append(f"ep{ep} step{step}: frame is None")
                    except Exception as e:
                        frame_errors.append(f"ep{ep} step{step}: {type(e).__name__}: {e}")

                    reward = td.get(("next", group, "reward"))
                    episode_reward += reward.sum().item()

                    done = td.get(("next", "done")).any().item()
                    td = td.get("next")
                    step += 1

                all_rewards.append(episode_reward)

                # Report frame errors
                if frame_errors and ep == 0:
                    print(f"  Frame errors (ep0 only): {frame_errors[:3]}")

                # Save video for this episode
                if frames:
                    vid = torch.cat(frames, dim=0).unsqueeze(0)  # (1, N, C, H, W)
                    # Ensure even dimensions (required by some codecs)
                    for idx in (-1, -2):
                        if vid.shape[idx] % 2 != 0:
                            vid = vid.index_select(idx, torch.arange(1, vid.shape[idx]))
                    video_path = os.path.join(video_dir, f"eval_hebbian_{ep}.mp4")
                    # write_video expects (N, H, W, C) format
                    vid_rgb = vid[0].permute(0, 2, 3, 1)  # (N, H, W, C)
                    torchvision.io.write_video(video_path, vid_rgb.numpy(), fps=20)
                    print(f"  Saved ep{ep}: {len(frames)} frames, reward={episode_reward:.2f}")
                else:
                    print(f"  SKIP ep{ep}: no frames captured, reward={episode_reward:.2f}")
                    if ep == 0:
                        print(f"    All errors: {frame_errors}")

        mean_reward = np.mean(all_rewards)
        print(f"\nHebbian evaluation ({n_episodes} episodes):")
        print(f"  Mean reward: {mean_reward:.2f}")
        print(f"  Min reward:  {min(all_rewards):.2f}")
        print(f"  Max reward:  {max(all_rewards):.2f}")
        print(f"  Videos saved to: {video_dir}")

        return mean_reward
