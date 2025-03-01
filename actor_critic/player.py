import torch
import gymnasium as gym
import torch.nn as nn


env = gym.make('CartPole-v1', render_mode='human')
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n


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

    def forward(self, x):
        y = self.backbone(x)

        return self.actor_head(y)


net = Actor2Critic(n_observations, n_actions)
net.load_state_dict(torch.load('a2c.pth', weights_only=True))

while True:
    state, _ = env.reset()

    R = 0
    while True:
        state = torch.tensor(state)[None]
        action = torch.argmax(net(state)).item()
        state, reward, terminated, truncated, _ = env.step(action)

        R += reward

        if terminated or truncated:
            break

    print(R)
