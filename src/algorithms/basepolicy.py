from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    @abstractmethod
    def collect_seed_episodes(self, env, buffer, augment_obs, augment_terminal_obs) -> int:
        ...

    @abstractmethod
    def predict(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        ...

    @abstractmethod
    def update(self, data) -> dict:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @abstractmethod
    def reset(self):
        ...
    @abstractmethod
    def reset_state(self):
        ...
