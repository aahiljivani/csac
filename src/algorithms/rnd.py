from .sac import SAC, Actor
from src.buffer import ExpertBuffer
import torch
from copy import deepcopy
from torch.distributions import Normal, Independent, kl_divergence
import numpy as np
from tqdm import tqdm
from time import time


class RND_SAC(SAC):
    def __init__(
        self,
        cfg,
        env,
        buffer,
        agent=None,
        file_path=None,
        expert_buffer=None,
    ):
        super().__init__(cfg, env)
        self.cl_reg_coef = cfg.cl_reg_coef
        self.expert_buffer_size = cfg.expert_buffer_size
        self.nepochs_offline = cfg.nepochs_offline
        self.rnd_batch_size = cfg.rnd_batch_size
        self.batch_size = cfg.rnd_batch_size
        self.replay_buffer_size = cfg.replay_buffer_size
        self.replay_buffer = buffer
        self.file_path = file_path
        self.task_name = None
        self.cfg = cfg
        self.device = cfg.device
        self.expert_buffer = expert_buffer if expert_buffer is not None else ExpertBuffer(cfg, env)
        self.target_policy = agent.actor if agent is not None else None
        self.offline_policy = None
        self.offline_actor_optimizer = None
        self.observations = None
        self.results = {
            "Policy loss": [],
            "BC loss": [],
            "Speed (it/s)": [],
        }
        self.seen_tasks = []

    def _sync_observations(self):
        size = int(getattr(self.replay_buffer, "size", len(self.replay_buffer.observations)))
        if size <= 0:
            raise ValueError("Replay buffer is empty; cannot run RND distillation")
        self.observations = self.replay_buffer.observations[:size]

    def _init_offline_policy(self):
        # Paper initializes θ_offline randomly once; do not warm-start from the expert.
        self.offline_policy = Actor(self.env, self.hidden_size).to(self.device)
        self.offline_policy.train()
        self.offline_actor_optimizer = torch.optim.Adam(
            self.offline_policy.parameters(),
            lr=self.cfg.policy_lr,
        )

    def compute_distill_setup(self, agent=None):
        if agent is not None:
            self.target_policy = agent.actor
        if self.file_path is not None:
            self.load_target_policy(self.file_path)
        if self.target_policy is None:
            raise ValueError("RND_SAC requires an agent actor or file_path to load a target policy")

        self._sync_observations()

        self.target_policy = deepcopy(self.target_policy).to(self.device).eval()
        for param in self.target_policy.parameters():
            param.requires_grad_(False)

        if self.offline_policy is None:
            self._init_offline_policy()

    def store_expert_subset(self):
        # Algorithm 1: store M_τ ⊂ D_τ into M after distillation.
        self.expert_buffer.add_observation_batch(
            self.observations, self.task_name, self.target_policy
        )
        if self.task_name not in self.seen_tasks:
            self.seen_tasks.append(self.task_name)

    def load_target_policy(self, file_path):
        target_agent = SAC(self.cfg, self.env).reset()
        checkpoint = torch.load(file_path, map_location=self.device)
        if "actor" in checkpoint:
            target_agent.actor.load_state_dict(checkpoint["actor"])
        else:
            target_agent.actor.load_state_dict(checkpoint)
        self.target_policy = target_agent.actor
        print(f"Target policy loaded from {file_path}")

    def compute_target(self):
        observations = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        means = []
        log_stds = []
        for start in range(0, len(observations), self.rnd_batch_size):
            end = min(start + self.rnd_batch_size, len(observations))
            with torch.no_grad():
                mean, log_std = self.target_policy(observations[start:end])
                means.append(mean.detach())
                log_stds.append(log_std.detach())
        return torch.cat(means, dim=0), torch.cat(log_stds, dim=0)

    def bc_loss(self):
        # seen_tasks only contains prior tasks until store_expert_subset runs.
        if len(self.seen_tasks) == 0:
            return torch.zeros(1, device=self.device)

        bc_loss = torch.zeros(1, device=self.device)
        for task_name in self.seen_tasks:
            obs, target_means, target_log_stds = self.expert_buffer.sample_expert_batch(
                self.batch_size, task_name=task_name
            )
            means, log_stds = self.offline_policy(obs)
            bc_loss = bc_loss + self.gaussian_kl(
                (means, log_stds), (target_means, target_log_stds)
            ) * self.cl_reg_coef
        return bc_loss / len(self.seen_tasks)

    @staticmethod
    def gaussian_kl(dist1, dist2):
        mu1, log_std1 = dist1
        mu2, log_std2 = dist2
        p = Independent(Normal(mu1, log_std1.exp()), 1)
        q = Independent(Normal(mu2, log_std2.exp()), 1)
        return kl_divergence(p, q).mean()

    def train(self, task_name, agent=None):
        self.task_name = task_name
        self.compute_distill_setup(agent=agent)
        all_target_means, all_target_log_stds = self.compute_target()
        observations = torch.as_tensor(self.observations, dtype=torch.float32, device=self.device)
        global_step = 0

        for epoch in range(self.nepochs_offline):
            idx_list = list(range(len(observations)))
            np.random.shuffle(idx_list)
            pbar = tqdm(
                range(0, len(observations), self.batch_size),
                desc=f"RND {task_name} Epoch {epoch + 1}",
                ascii=" =",
                leave=True,
            )

            for i in pbar:
                start = i
                end = min(start + self.batch_size, len(observations))
                idxs = idx_list[start:end]
                obs = observations[idxs]
                batch_target_means = all_target_means[idxs]
                batch_target_log_stds = all_target_log_stds[idxs]

                means, log_stds = self.offline_policy(obs)
                distill_loss = self.gaussian_kl(
                    (batch_target_means, batch_target_log_stds), (means, log_stds)
                )
                bc_loss = self.bc_loss()
                loss = distill_loss + bc_loss

                self.offline_actor_optimizer.zero_grad()
                loss.backward()
                self.offline_actor_optimizer.step()

                end_time = time()
                global_step += 1

                if global_step % 1000 == 0:
                    logging_cfg = getattr(self.cfg, "logging", None)
                    if logging_cfg is not None and bool(getattr(logging_cfg, "enable_wandb", False)):
                        import wandb

                        wandb.log({
                            "global_step": global_step,
                            "task_name": self.task_name,
                            "RND/Policy loss": distill_loss.item(),
                            "RND/BC loss": bc_loss.item(),
                            "RND/Speed (it/s)": global_step / (end_time - self.start_time),
                        })

                    self.results["Policy loss"].append(distill_loss.item())
                    self.results["BC loss"].append(bc_loss.item())
                    self.results["Speed (it/s)"].append(
                        global_step / (end_time - self.start_time)
                    )

        #store M_τ after the distillation loop.
        self.store_expert_subset()
        return self.offline_policy

    # def fill_buffer()
