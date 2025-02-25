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
writer = SummaryWriter()

# %%
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n
GAMMA = 0.99
BATCH_SIZE = 16
BETA = 0.01

# %%
net = nn.Sequential(
    nn.Linear(n_observations, 128),
    nn.ReLU(),
    #
    nn.Linear(128, 64),
    nn.ReLU(),
    #
    nn.Linear(64, n_actions),
)

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
        action_logs.append(torch.log(dist.probs[0][action])) # (1)

        entropies.append(dist.entropy())

        if terminated or truncated:
            break

        state = next_state

    action_logs = torch.stack(action_logs)  # B
    entropies = torch.stack(entropies)  # B

    qvals = calc_qvals(rewards)  # B
    qvals = (qvals - qvals.mean()) / (qvals.std() + 1e-10)

    loss = -(qvals * action_logs).mean() - BETA * entropies.mean()
    loss.backward()
    
    if (episode + 1) % BATCH_SIZE == 0:
        opt.step()
        opt.zero_grad()

    writer.add_scalar('rewards', sum(rewards), episode)
    writer.add_scalar('loss', loss.item(), episode)
    print(f'episode: {episode}, reward: {sum(rewards)}, loss: {loss}')

# %%
torch.save(net.state_dict(), './reinforce4.pth')