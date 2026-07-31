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

from src.buffer import ExpertBuffer, ReplayBuffer
from src.algorithms import RND_SAC
from src.train import ContinualBenchVecEnv

# ---------------------------------------------------------------------------
# Constants — adjust these to trade tuning speed vs. quality
# ---------------------------------------------------------------------------
TASK_SEQUENCE = ["block", "button"]  # block (hardest) first, button (easiest) second
# Per-task training budget in env steps (summed across parallel envs). Full
# runs default to 1e6 (cfgs/default.yaml train.steps_per_task); tuning trials
# use a condensed budget so multiple seeds per trial stay affordable.
STEPS_PER_TASK = {"block": 1_000_000, "button": 500_000}
# Fixed seed set shared by EVERY trial: single-seed SAC scores are dominated
# by seed luck, so each trial trains the same hyperparameters on all of these
# seeds and is scored by the mean. Using identical seeds across trials means
# trials differ only in hyperparameters, not in seed draw.
SEEDS = [0, 1]
# Score weights for the post-sequence eval of the distilled actor. block is
# weighted higher because it is both the hardest task and the one exposed to
# forgetting after button is learned.
SCORE_WEIGHTS = {"block": 0.6, "button": 0.4}
N_EVAL_EPISODES = 10        # greedy eval episodes per task used to score each trial
N_TUNE_ENVS = 10            # parallel envs per trial
N_TRIALS = 12               # total Optuna trials (each runs up to len(SEEDS) training passes)
TARGET_BUFFER_SIZE = 100_000  # |D_tau| expert rollouts per task (vs. 1e6 in full runs)
EXPERT_BUFFER_SIZE = 10_000   # |M_k| expert snapshots kept per task


def build_cfg(params: dict, seed: int) -> OmegaConf:
    """Build an OmegaConf config for a single HPO trial/seed without touching sys.argv."""
    cfg_dir = Path(__file__).resolve().parent.parent / "cfgs"
    base = OmegaConf.load(cfg_dir / "default.yaml")
    base.merge_with(OmegaConf.load(cfg_dir / "algorithms" / "sac.yaml"))
    base.merge_with(OmegaConf.load(cfg_dir / "continual.yaml"))

    overrides = OmegaConf.create(
        {
            "task_list": list(TASK_SEQUENCE),
            "benchmark_mode": "continual",
            "num_envs": N_TUNE_ENVS,
            "seed": int(seed),
            "policy": "SAC",
            "rnd": True,
            "replay_buffer_enabled": True,
            "expert_buffer_enabled": True,
            "target_replay_buffer_size": TARGET_BUFFER_SIZE,
            "expert_buffer_size": EXPERT_BUFFER_SIZE,
            "train": {"steps_per_task": max(STEPS_PER_TASK.values())},
            "autotune": True,
            "logging": {"enable_wandb": True, "project": "RND HP Tuning Block-Button"},
            **params,
        }
    )
    base.merge_with(overrides)
    return base


def _train_task_online(bench, cfg, vec_env, agent, rb, task_name, n_steps, global_step, run):
    """Online SAC training on one task; mirrors the inner loop of train.py.

    The task budget is n_steps env steps (summed across parallel envs), not an
    episode count; the loop rounds up to a full episode.
    """
    learning_starts = int(cfg.learning_starts)
    max_episode_steps = int(vec_env.get_attr("max_path_length")[0])
    train_successes: list[float] = []

    task_start_step = global_step
    episode_idx = 0
    while global_step - task_start_step < n_steps:
        obs = vec_env.reset()
        obs = bench._augment_obs(task_name, obs)
        ep_reward = 0.0
        ep_success = 0.0
        loss_accum = defaultdict(list)
        for _ in range(max_episode_steps):
            if global_step - task_start_step < learning_starts:
                actions = np.array(
                    [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
                )
            else:
                actions = agent.predict(obs, deterministic=False)

            next_obs, rewards, dones, infos = vec_env.step(actions)
            next_obs = bench._augment_obs(task_name, next_obs)
            global_step += vec_env.num_envs

            successes = np.array(
                [float(info["success"]) for info in infos], dtype=np.float32
            )
            terminated = np.logical_and(dones, successes.astype(bool))

            real_next_obs = next_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    real_next_obs[idx] = bench._augment_terminal_obs(
                        task_name, infos[idx]["terminal_observation"]
                    )

            rb.add(obs, actions, rewards, terminated, real_next_obs)
            obs = next_obs

            ep_reward += float(np.mean(rewards))
            ep_success = max(ep_success, float(np.mean(successes)))

            if global_step - task_start_step > learning_starts and hasattr(agent, "update"):
                for _ in range(vec_env.num_envs):
                    data = rb.sample(cfg.batch_size)
                    metrics = agent.update(data)
                    if metrics:
                        for key, value in metrics.items():
                            if value is not None:
                                loss_accum[key].append(value)

        train_successes.append(ep_success)
        episode_idx += 1
        run.log(
            {
                "global_step": global_step,
                f"train/{task_name}/episode_idx": episode_idx,
                f"train/{task_name}/episode_reward": ep_reward,
                f"train/{task_name}/episode_success": ep_success,
                f"train/{task_name}/running_success_rate": float(np.mean(train_successes)),
                **{
                    f"losses/{task_name}/{key}": float(np.mean(values))
                    for key, values in loss_accum.items()
                },
            }
        )

    train_success_rate = float(np.mean(train_successes)) if train_successes else 0.0
    return train_success_rate, global_step


def _collect_expert_rollouts(bench, cfg, vec_env, agent, task_name):
    """Fill a target buffer with stochastic expert rollouts for RND distillation."""
    rb_target = ReplayBuffer(cfg=cfg, env=vec_env, capacity=int(cfg.target_replay_buffer_size))
    ep_successes = 0
    ep_count = 0
    obs = vec_env.reset()
    obs = bench._augment_obs(task_name, obs)
    while rb_target.size < rb_target.replay_buffer_size:
        actions = agent.predict(obs, deterministic=False)
        next_obs, rewards, dones, infos = vec_env.step(actions)
        next_obs = bench._augment_obs(task_name, next_obs)

        real_next_obs = next_obs.copy()
        for idx, done in enumerate(dones):
            if not done:
                continue
            ep_count += 1
            ep_successes += int(bool(infos[idx].get("success", 0.0)))
            real_next_obs[idx] = bench._augment_terminal_obs(
                task_name, infos[idx]["terminal_observation"]
            )
        rb_target.add(obs, actions, rewards, dones, real_next_obs)
        obs = next_obs
    expert_success_rate = ep_successes / max(ep_count, 1)
    return rb_target, expert_success_rate


def train_for_hpo(cfg, trial: optuna.Trial, params: dict, seed: int) -> float:
    """Run one seed of a condensed RND_SAC pass over block -> button.

    Returns the weighted AP score for this seed only; the multi-seed
    aggregation (and pruning) happens in objective().
    """
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = bool(cfg.torch_deterministic)

    cfg.device = "cuda" if torch.cuda.is_available() and cfg.cuda else "cpu"

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb_config["trial_number"] = trial.number
    wandb_config["seed"] = seed
    wandb_config["sampled_hyperparameters"] = params
    run = wandb.init(
        project=cfg.logging.project,
        entity=cfg.logging.wandb_entity,
        name=f"rnd_sac_block_button_hpo_trial_{trial.number}_seed_{seed}",
        config=wandb_config,
        # Group all seeds of one trial together so wandb can plot the
        # mean +- std learning curve per hyperparameter set.
        group=f"rnd_sac_block_button_hpo_trial_{trial.number}",
        job_type="trial",
        reinit=True,
    )
    run.define_metric("global_step")
    run.define_metric("*", step_metric="global_step")
    run.log(
        {
            "global_step": 0,
            "trial/number": trial.number,
            "trial/seed": seed,
            **{f"hparams/{key}": value for key, value in params.items()},
        }
    )

    bench = ContinualBenchVecEnv(cfg)
    vec_env = None
    try:
        global_step = 0
        agent = None
        rb = None
        expert_buffer = None
        rnd_sac = None
        seen_tasks: list[str] = []
        final_per_task: dict[str, float] = {}

        for task_idx, task_name in enumerate(TASK_SEQUENCE):
            vec_env = bench.make_task_env(task_name)
            try:
                if task_idx == 0:
                    agent = bench._build_policy(vec_env, num_envs=bench.num_envs)
                    rb = ReplayBuffer(cfg=cfg, env=vec_env)
                    expert_buffer = ExpertBuffer(cfg=cfg, env=vec_env)
                else:
                    rb.reset()

                train_success_rate, global_step = _train_task_online(
                    bench,
                    cfg,
                    vec_env,
                    agent,
                    rb,
                    task_name,
                    STEPS_PER_TASK[task_name],
                    global_step,
                    run,
                )

                rb_target, expert_success_rate = _collect_expert_rollouts(
                    bench, cfg, vec_env, agent, task_name
                )
                print(
                    f"[Trial {trial.number}][Task {task_idx}] Collected {rb_target.size} "
                    f"expert transitions for {task_name} "
                    f"(episodic success {expert_success_rate:.3f})"
                )

                if rnd_sac is None:
                    rnd_sac = RND_SAC(
                        cfg,
                        vec_env,
                        rb_target,
                        agent=agent,
                        expert_buffer=expert_buffer,
                    )
                else:
                    rnd_sac.replay_buffer = rb_target
                    rnd_sac.env = vec_env
                offline_policy = rnd_sac.train(task_name, agent=agent)
                agent.actor.load_state_dict(offline_policy.state_dict())
            finally:
                vec_env.close()
                vec_env = None

            seen_tasks.append(task_name)
            # The distilled offline weights now live in agent.actor, so this
            # measures the continual (post-distillation) policy on every task
            # seen so far — including forgetting of block once button is done.
            per_task = {
                t: bench.evaluate_task(t, agent, N_EVAL_EPISODES) for t in seen_tasks
            }
            final_per_task = per_task
            run.log(
                {
                    "global_step": global_step,
                    "trial/number": trial.number,
                    f"train/{task_name}/success_rate": train_success_rate,
                    f"expert/{task_name}/rollout_success_rate": expert_success_rate,
                    **{f"eval/after_{task_name}/{t}_success": v for t, v in per_task.items()},
                    f"eval/after_{task_name}/mean_success": float(np.mean(list(per_task.values()))),
                }
            )
            print(
                f"[Trial {trial.number}][Seed {seed}][Task {task_idx}] {task_name} done. "
                f"per-task={per_task}"
            )
            # NOTE: pruning happens at the seed level in objective(), not per
            # task — Optuna's intermediate steps for this trial are seed
            # indices, and mixing task indices in would corrupt the pruner.

            # Reset & Distill: the online agent restarts from scratch on the
            # next task; knowledge is retained in the distilled offline actor.
            if task_idx < len(TASK_SEQUENCE) - 1 and hasattr(agent, "reset"):
                agent.reset()

        combined_score = float(
            sum(SCORE_WEIGHTS[t] * final_per_task[t] for t in TASK_SEQUENCE)
        )
        run.log(
            {
                "global_step": global_step,
                "trial/number": trial.number,
                **{f"eval/final_{t}_success": v for t, v in final_per_task.items()},
                "trial/combined_score": combined_score,
            }
        )
        for t, v in final_per_task.items():
            run.summary[f"final_{t}_success"] = v
        run.summary["combined_score"] = combined_score
        return combined_score
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
        run.finish()


def sample_rnd_sac_params(trial: optuna.Trial) -> dict:
    return {
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "nepochs_offline": trial.suggest_categorical("nepochs_offline", [3, 5, 10]),
    }

def objective(trial: optuna.Trial) -> float:
    """Score one hyperparameter set as the mean over a fixed set of seeds.

    SAC's seed-to-seed variance for identical hyperparameters can exceed the
    difference between good and bad configs, so a single-seed score would
    mostly rank seed luck. Every trial runs the same SEEDS, reports the
    running mean after each seed so the MedianPruner can kill configs whose
    early seeds clearly flop, and returns the mean across seeds.
    """
    params = sample_rnd_sac_params(trial)

    seed_scores: list[float] = []
    for seed_idx, seed in enumerate(SEEDS):
        cfg = build_cfg(params, seed=seed)
        score = train_for_hpo(cfg, trial, params, seed=seed)
        seed_scores.append(score)
        trial.set_user_attr("seed_scores", list(seed_scores))

        running_mean = float(np.mean(seed_scores))
        trial.report(running_mean, step=seed_idx)
        if trial.should_prune():
            trial.set_user_attr("pruned_after_seed", seed)
            raise optuna.TrialPruned()

    trial.set_user_attr("seed_score_mean", float(np.mean(seed_scores)))
    trial.set_user_attr("seed_score_std", float(np.std(seed_scores)))
    trial.set_user_attr("seed_score_median", float(np.median(seed_scores)))
    return float(np.mean(seed_scores))


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        study_name="rnd_sac_block_button_hpo",
        sampler=optuna.samplers.TPESampler(seed=42),
        # Intermediate steps are seed indices: a trial whose running mean
        # after seed k is below the median of completed trials at seed k is
        # pruned before paying for the remaining seeds.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0),
    )
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

    print("\nBest trial:")
    best = study.best_trial
    print(
        "  Combined score "
        f"({' + '.join(f'{w}*{t}' for t, w in SCORE_WEIGHTS.items())} success), "
        f"mean over seeds {SEEDS}: {best.value:.3f}"
    )
    if "seed_scores" in best.user_attrs:
        print(f"  Per-seed scores: {best.user_attrs['seed_scores']}")
        print(f"  Std across seeds: {best.user_attrs.get('seed_score_std', float('nan')):.3f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    best_cfg = build_cfg(best.params, seed=SEEDS[0])
    summary = wandb.init(
        project=best_cfg.logging.project,
        entity=best_cfg.logging.wandb_entity,
        name="rnd_sac_block_button_hpo_best",
        group="rnd_sac_block_button_hpo",
        job_type="summary",
        reinit=True,
    )
    summary.summary["best_combined_score"] = best.value
    summary.summary["best_trial_number"] = best.number
    summary.summary["seeds"] = SEEDS
    if "seed_scores" in best.user_attrs:
        summary.summary["best_seed_scores"] = best.user_attrs["seed_scores"]
        summary.summary["best_seed_score_std"] = best.user_attrs.get("seed_score_std")
    summary.config.update(best.params)
    summary.finish()
