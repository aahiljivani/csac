import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .basepolicy import BasePolicy

LOG_STD_MAX = 2
LOG_STD_MIN = -5


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


class SoftQNetwork(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        observation_space = env.observation_space
        action_space = env.action_space
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class Actor(nn.Module):
    def __init__(self, env, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        observation_space = env.observation_space
        action_space = env.action_space
        obs_dim = int(np.array(observation_space.shape).prod())
        act_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, act_dim)
        self.fc_logstd = nn.Linear(hidden_size, act_dim)
        self.register_buffer(
            "action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )
        self.apply(weight_init)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

    def get_eval_action(self, x):
        mean, _ = self(x)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SAC(BasePolicy):
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.device = cfg.device
        self.hidden_size = cfg.hidden_size
        self.learning_rate = cfg.policy_lr
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.learning_starts = cfg.learning_starts
        self.policy_frequency = cfg.policy_frequency
        self.target_network_frequency = cfg.target_network_frequency
        self.alpha = cfg.alpha
        self.autotune = cfg.autotune
        self.start_time = time.time()
        self.q_lr = cfg.q_lr

    def reset(self):
        self.update_count = 0
        self.actor = Actor(self.env, self.hidden_size).to(self.device)
        self.q1 = SoftQNetwork(self.env, self.hidden_size).to(self.device)
        self.q2 = SoftQNetwork(self.env, self.hidden_size).to(self.device)
        self.q1_target = SoftQNetwork(self.env, self.hidden_size).to(self.device)
        self.q2_target = SoftQNetwork(self.env, self.hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # critic uses q_lr; actor uses learning_rate
        self.q_optimizer = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.q_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_lr)
        return self

    def collect_seed_episodes(self, env, buffer, augment_obs, augment_terminal_obs):
        """Fill `buffer` with random-action rollouts until learning_starts env steps."""
        obs = augment_obs(env.reset())
        collected = 0
        while collected < self.learning_starts:
            actions = np.array(
                [env.action_space.sample() for _ in range(env.num_envs)],
                dtype=np.float32,
            )
            next_obs, rewards, dones, infos = env.step(actions)
            next_obs = augment_obs(next_obs)
            successes = np.array(
                [float(info["success"]) for info in infos],
                dtype=np.float32,
            )
            terminated = np.logical_and(dones, successes.astype(bool))
            real_next_obs = next_obs.copy()
            for idx, done in enumerate(dones):
                if done:
                    real_next_obs[idx] = augment_terminal_obs(infos[idx]["terminal_observation"])
            buffer.add(obs, actions, rewards, terminated, real_next_obs)
            obs = next_obs
            collected += env.num_envs
        return collected

    def predict(self, obs, deterministic=False):
        with torch.no_grad():
            if deterministic:
                action = self.actor.get_eval_action(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            else:
                action, _, _ = self.actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
            return action.cpu().numpy()

    def update(self, data):  # where data is a ReplayBuffer with torch.tensor type
        self.update_count += 1
        actor_loss = None
        alpha_loss = None

        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data.next_observations)
            q1_next_target = self.q1_target(data.next_observations, next_state_actions)
            q2_next_target = self.q2_target(data.next_observations, next_state_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * self.gamma * min_q_next_target.view(-1)

        # calculate the q values for the current state
        q1_a_values = self.q1(data.observations, data.actions).view(-1)
        q2_a_values = self.q2(data.observations, data.actions).view(-1)
        # calculate the q losses
        q1_loss = F.mse_loss(q1_a_values, next_q_value)
        q2_loss = F.mse_loss(q2_a_values, next_q_value)
        # calculate the total q loss
        q_loss = q1_loss + q2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        if self.update_count % self.policy_frequency == 0:  # TD3-style delayed actor update
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.actor.get_action(data.observations)
                q1_pi = self.q1(data.observations, pi)
                q2_pi = self.q2(data.observations, pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = ((self.alpha * log_pi) - min_q_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.actor.get_action(data.observations)
                    alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

        # soft update target networks every target_network_frequency gradient steps
        if self.update_count % self.target_network_frequency == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if self.update_count % 100 != 0:
            return {}

        elapsed_time = max(time.time() - self.start_time, 1e-9)
        metrics = {
            "charts/q1_values": q1_a_values.mean().item(),
            "charts/q2_values": q2_a_values.mean().item(),
            "losses/q1_loss": q1_loss.item(),
            "losses/q2_loss": q2_loss.item(),
            "losses/actor_loss": actor_loss.item() if actor_loss is not None else None,
            "losses/alpha": self.alpha,
            "charts/SPS": int(self.update_count / elapsed_time),
        }
        if self.autotune:
            metrics["losses/alpha_loss"] = alpha_loss.item() if alpha_loss is not None else None
        return metrics

    def save(self, path):
        model_path = f"{path}/checkpoint.sac_model"
        checkpoint = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "alpha": self.alpha,
            "log_alpha": self.log_alpha if self.autotune else None,
        }
        torch.save(checkpoint, model_path)
        print(f"model saved to {model_path}")
        return model_path

    def load(self, path):
        model_path = f"{path}/checkpoint.sac_model"
        checkpoint = torch.load(model_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.alpha = checkpoint["alpha"]
        if self.autotune and checkpoint["log_alpha"] is not None:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint["log_alpha"])

        print(f"model loaded from {model_path}")
