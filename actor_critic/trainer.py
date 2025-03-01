# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# %%
class Actor2Critic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            #
            nn.Linear(128, 64),
            nn.ReLU(),
            #
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, out_channels),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

# %%
