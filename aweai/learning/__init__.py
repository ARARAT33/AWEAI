from __future__ import annotations

from aweai.learning.meta import MAML, Reptile, MetaLearner
from aweai.learning.continual import ContinualLearner, EWC, ReplayBuffer
from aweai.learning.self_supervised import SimCLR, MoCo, BYOL, DINO, MAE
from aweai.learning.reinforcement import QLearning, DQN, PPO, SAC
from aweai.learning.federated import FederatedAveraging, FedProx, SCAFFOLD

__all__ = [
    "MAML",
    "Reptile",
    "MetaLearner",
    "ContinualLearner",
    "EWC",
    "ReplayBuffer",
    "SimCLR",
    "MoCo",
    "BYOL",
    "DINO",
    "MAE",
    "QLearning",
    "DQN",
    "PPO",
    "SAC",
    "FederatedAveraging",
    "FedProx",
    "SCAFFOLD",
]
