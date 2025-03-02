# %%
import gymnasium as gym
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple
import numpy as np
from itertools import count
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# %%
np.finfo(np.float32).eps.item()

# %%
params = {
    'gamma': 0.99,
    'seed': 543,
    'lr': 3e-3,
}

# %%
env = gym.make('CartPole-v1')
env.reset(seed=params['seed'])
torch.manual_seed(params['seed'])

SavedAction = namedtuple('Savedaction', ['log_prob', 'value'])


# %%
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)

        self.saved_actions = []
        self.rewards = []

    def append_action(self, log_prob, value):
        self.saved_actions.append(SavedAction(log_prob, value))

    def append_reward(self, reward):
        self.rewards.append(reward)

    def clear_memo(self):
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        y = self.body(x)

        action = self.action_head(y)
        value = self.value_head(y)

        return action, value


# %%
model = Policy()
writer = SummaryWriter('./runs/lower_lr_big_network')
optimizer = optim.Adam(model.parameters(), lr=params['lr'])
eps = np.finfo(np.float32).eps.item()


# %%
def select_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    logits, state_value = model(state)

    dist = Categorical(logits=logits)
    action = dist.sample()

    model.append_action(dist.log_prob(action), state_value)

    return action.item()


# %%
def finish_episode():
    R = 0
    returns = []

    policy_loss = torch.tensor([0.0])
    value_loss = torch.tensor([0.0])

    for r in reversed(model.rewards):
        R = r + params['gamma'] * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(model.saved_actions, returns):
        advantage = R - value.item()  # r + gamam * V(s') - V(s)
        policy_loss += -advantage * log_prob
        value_loss += torch.nn.functional.smooth_l1_loss(value, torch.tensor([R]))

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.clear_memo()


# %%
def log_grads(step):
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    max_grad = grads.max()
    var_grad = grads.var()
    l2_grad = grads.norm()

    writer.add_scalar('grad/max', max_grad, step)
    writer.add_scalar('grad/var', var_grad, step)
    writer.add_scalar('grad/l2', l2_grad, step)


# %%
def main():
    avg_reward = 0  # 10 originally
    for i in count(1):
        state, _ = env.reset()

        episode_reward = 0

        while True:
            action = select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            model.append_reward(reward)
            episode_reward += reward
            if terminated or truncated:
                break

            state = next_state

        finish_episode()
        log_grads(i)

        avg_reward = 0.05 * episode_reward + (1 - 0.05) * avg_reward

        if i % 30 == 0:
            print(f'Episode: {i}\tLast reward: {episode_reward:.2f}\tAvg reward: {avg_reward:.2f}')
            torch.save(model.state_dict(), './dummy.pth')


# %%
if __name__ == '__main__':
    main()
