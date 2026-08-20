import os

os.environ.setdefault("MUJOCO_GL", "egl")

import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import torch
import wandb
from omegaconf import OmegaConf

from src.buffer import ReplayBuffer
from src.train import ContinualBenchVecEnv

# ---------------------------------------------------------------------------
# Constants — adjust these to trade tuning speed vs. quality
# ---------------------------------------------------------------------------
TASK = "block"          # pick-cube analog in this codebase
N_TUNE_EPISODES = 400    # episodes per trial (vs. 100 in full training)
N_EVAL_EPISODES = 20    # greedy eval episodes used to score each trial
N_TUNE_ENVS = 10         # parallel envs per trial (fewer than default 10)
N_TRIALS = 50           # total Optuna trials


def build_cfg(params: dict, n_episodes: int = N_TUNE_EPISODES) -> OmegaConf:
    """Build an OmegaConf config for a single HPO trial without touching sys.argv."""
    cfg_dir = Path(__file__).resolve().parent.parent / "cfgs"
    base = OmegaConf.load(cfg_dir / "default.yaml")
    base.merge_with(OmegaConf.load(cfg_dir / "algorithms" / "sac.yaml"))
    base.merge_with(OmegaConf.load(cfg_dir / "task.yaml"))

    overrides = OmegaConf.create(
        {
            "single_task_name": TASK,
            "benchmark_mode": "task",
            "num_envs": N_TUNE_ENVS,
            "train": {"episodes_per_task": n_episodes},
            "logging": {"enable_wandb": False},
            "autotune": True,
            **params,
        }
    )
    base.merge_with(overrides)
    return base


def train_for_hpo(cfg, trial_number: int, params: dict) -> float:
    """Run a condensed training pass on `block` and return its success rate."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    cfg.device = "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config["trial_number"] = trial_number
    wandb_config["sampled_hyperparameters"] = params
    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.wandb_entity,
        name=f"sac_{TASK}_hpo_trial_{trial_number}",
        config=wandb_config,
        group="sac_block_hpo",
        job_type="trial",
        reinit=True,
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    wandb.log(
        {
            "global_step": 0,
            "trial/number": trial_number,
            **{f"hparams/{key}": value for key, value in params.items()},
        }
    )

    bench = ContinualBenchVecEnv(cfg)
    vec_env = None
    try:
        vec_env = bench.make_task_env(TASK)

        agent = bench._build_policy(vec_env, num_envs=bench.num_envs)
        rb = ReplayBuffer(cfg=cfg, env=vec_env)

        learning_starts = int(cfg.learning_starts)
        max_episode_steps = int(vec_env.get_attr("max_path_length")[0])

        global_step = 0
        train_successes: list[float] = []
        for episode_idx in range(bench.train_episodes_per_task):
            obs = vec_env.reset()
            obs = bench._augment_obs(TASK, obs)
            ep_reward = 0.0
            ep_success = 0.0
            loss_accum = defaultdict(list)
            for timestep in range(max_episode_steps):
                if global_step < learning_starts:
                    actions = np.array(
                        [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
                    )
                else:
                    actions = agent.predict(obs, deterministic=False)

                next_obs, rewards, dones, infos = vec_env.step(actions)
                next_obs = bench._augment_obs(TASK, next_obs)
                global_step += vec_env.num_envs

                successes = np.array(
                    [float(info["success"]) for info in infos], dtype=np.float32
                )
                terminated = np.logical_and(dones, successes.astype(bool))

                real_next_obs = next_obs.copy()
                for idx, done in enumerate(dones):
                    if done:
                        real_next_obs[idx] = bench._augment_terminal_obs(
                            TASK, infos[idx]["terminal_observation"]
                        )

                rb.add(obs, actions, rewards, terminated, real_next_obs)
                obs = next_obs

                ep_reward += float(np.mean(rewards))
                ep_success = max(ep_success, float(np.mean(successes)))

                if global_step > learning_starts:
                    for _ in range(vec_env.num_envs):
                        data = rb.sample(cfg.batch_size)
                        metrics = agent.update(data)
                        if metrics:
                            for key, value in metrics.items():
                                if value is not None:
                                    loss_accum[key].append(value)

            train_successes.append(ep_success)
            wandb.log(
                {
                    "global_step": global_step,
                    "trial/number": trial_number,
                    "train/episode_idx": episode_idx,
                    "train/episode_reward": ep_reward,
                    "train/episode_success": ep_success,
                    "train/running_success_rate": float(np.mean(train_successes)),
                    **{
                        f"losses/{key}": float(np.mean(values))
                        for key, values in loss_accum.items()
                    },
                }
            )

        vec_env.close()
        train_success_rate = float(np.mean(train_successes)) if train_successes else 0.0
        eval_success_rate = bench.evaluate_task(TASK, agent, N_EVAL_EPISODES)
        # Score trials on how well the params learn the task: reward both
        # consistent success *during* training and greedy success at eval.
        combined_score = 0.5 * train_success_rate + 0.5 * eval_success_rate
        wandb.log(
            {
                "global_step": global_step,
                "trial/number": trial_number,
                "train/success_rate": train_success_rate,
                "eval/final_success_rate": eval_success_rate,
                "trial/combined_score": combined_score,
            }
        )
        run.summary["train_success_rate"] = train_success_rate
        run.summary["final_success_rate"] = eval_success_rate
        run.summary["combined_score"] = combined_score
        return combined_score
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        run.finish()


def sample_sac_params(trial: optuna.Trial) -> dict:
    return {
        "policy_lr": trial.suggest_float("policy_lr", 1e-4, 1e-3, log=True),
        "q_lr": trial.suggest_float("q_lr", 1e-4, 1e-3, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.98, 0.99, 0.995]),
        "tau": trial.suggest_categorical("tau", [0.005, 0.01]),
        "batch_size": trial.suggest_categorical("batch_size", [64,128, 256, 512]),
        "learning_starts": trial.suggest_categorical("learning_starts", [1000, 5000, 10000]),
        "policy_frequency": trial.suggest_categorical("policy_frequency", [1, 2]),
        "target_network_frequency": trial.suggest_categorical("target_network_frequency", [1, 2]),
        "hidden_size": trial.suggest_categorical("hidden_size", [256, 400, 512]),
    }


def objective(trial: optuna.Trial) -> float:
    params = sample_sac_params(trial)
    cfg = build_cfg(params)
    return train_for_hpo(cfg, trial.number, params)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="sac_block_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Combined score (0.5*train + 0.5*eval success): {best.value:.3f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    best_cfg = build_cfg(best.params)
    summary = wandb.init(
        project=best_cfg.logging.project,
        entity=best_cfg.logging.wandb_entity,
        name="sac_block_hpo_best",
        group="sac_block_hpo",
        job_type="summary",
        reinit=True,
    )
    summary.summary["best_combined_score"] = best.value
    summary.summary["best_trial_number"] = best.number
    summary.config.update(best.params)
    summary.finish()
