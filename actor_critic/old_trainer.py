# %%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical


# %%
def calc_qvals(rewards: list[float]) -> torch.Tensor:
    qvals = torch.zeros(len(rewards))
    qval = 0

    for i in reversed(range(len(rewards))):
        qval = rewards[i] + GAMMA * qval
        qvals[i] = qval
    return qvals


# %%
env = gym.make('CartPole-v1')
writer = SummaryWriter(log_dir='runs/baseline')

# %%
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
GAMMA = 0.99
BATCH_SIZE = 8
BETA_START = 0.01  # 0.1
BETA_END = 0.0001

beta = BETA_START


# %%
class ActorCritic(nn.Module):
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

    def forward(self, x) -> list[torch.Tensor]:
        """
        returns actor, critic
        """
        x = self.backbone(x)
        actor_y = self.actor_head(x)
        critic_y = self.critic_head(x)

        return actor_y, critic_y


net = ActorCritic(n_observations, n_actions)

opt = optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
episodes = 1500

for episode in range(episodes):
    rewards = []
    action_logs = []
    entropies = []

    state, _ = env.reset()

    while True:
        state = torch.tensor(state, dtype=torch.float32)[None]
        dist = Categorical(logits=net(state))  # 1xA

        action = dist.sample().item()
        next_state, reward, terminated, truncated, _ = env.step(action)

        rewards.append(reward)
        action_logs.append(torch.log(dist.probs[0][action]))  # (1)

        entropies.append(dist.entropy())

        if terminated or truncated:
            break

        state = next_state

    action_logs = torch.stack(action_logs)  # B
    entropies = torch.stack(entropies)  # B

    qvals = calc_qvals(rewards)  # B
    advantages = qvals - qvals.mean()

    loss = -(advantages * action_logs).mean() - beta * entropies.mean()
    loss.backward()

    DECAY_FACTOR = (BETA_END / BETA_START) ** (1 / episodes)
    beta = max(beta * DECAY_FACTOR, BETA_END)

    grads = torch.cat([p.grad.flatten() for p in net.parameters() if p.grad is not None])
    grad_l2 = grads.norm(2).item()
    grad_max = grads.abs().max().item()
    grad_var = grads.var().item()

    writer.add_scalar('grads/l2', grad_l2, episode)
    writer.add_scalar('grad/max', grad_max, episode)
    writer.add_scalar('grad/var', grad_var, episode)

    if (episode + 1) % BATCH_SIZE == 0:
        opt.step()
        opt.zero_grad()
        total_loss = 0

    writer.add_scalar('rewards', sum(rewards), episode)
    print(f'episode: {episode}, reward: {sum(rewards)}, loss: {loss}')

# %%
torch.save(net.state_dict(), 'reinforce.pth')
