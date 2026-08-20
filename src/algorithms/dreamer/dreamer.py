from src.buffer import ExpertBuffer
import torch
import numpy as np 
from src.algorithms.basepolicy import BasePolicy
class Dreamer(BasePolicy):
    def __init__(self, cfg, env):
        self.obs_shape = env.observation_space.shape
        self.action_size = env.action_space.shape[0]
        self.hidden_size = cfg.hidden_size
        self.learning_rate = cfg.learning_rate
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.seq_len = cfg.seq_len
        self.device = cfg.device

    def collect_seed_episodes(self, env, buffer, augment_obs, augment_terminal_obs):
        pass
    def predict(self, obs: np.ndarray, deterministic: bool)-> np.ndarray:
        pass
        
    def update(self, replay_buffer, logger, step) -> None:
        pass
        
    def save(self, path:str) -> None:
        pass
        

    def load(self, path:str) -> None:
        pass

    def reset_state(self):
        pass
    