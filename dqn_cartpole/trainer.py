# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple
import gymnasium as gym
import torch.optim as optim
import random
from torchrl.data import PrioritizedReplayBuffer, ListStorage

# %%
LR = 1e-3
EPS_END = 0.05
EPS_START = 1
BATCH_SIZE = 128  # expirement with this more
EPS_DECAY = 5000  # does this cause faster or slower update?
TAU = 0.005
GAMMA = 0.99

# %%
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
buffer = PrioritizedReplayBuffer(alpha=0.7, beta=0.9, storage=ListStorage(10000))


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


# %%
env = gym.make('CartPole-v1')

n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

opt = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.HuberLoss()

# %%
steps_done = 0


def select_action(state):
    global steps_done

    eps = EPS_END + (EPS_START - EPS_END) * np.exp(-1.0 * steps_done / EPS_DECAY)  # try to look at how eps is updated
    steps_done += 1

    if random.random() < eps:
        return torch.tensor([[env.action_space.sample()]]).view(1)

    with torch.no_grad():  # attempt without this
        # why this weird shape? why not get the item directly
        return torch.argmax(policy_net(state)).view(1)


# %%
def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return

    batch, info = buffer.sample(BATCH_SIZE, return_info=True)

    state_batch = batch.state  # Bx4
    action_batch = batch.action  # Bx1
    reward_batch = batch.reward  # B
    done_batch = batch.done  # B

    not_final_mask = ~done_batch
    non_final_next_states = state_batch[not_final_mask]  # Bx4

    q_values = policy_net(state_batch).gather(1, action_batch)  # Bx1
    best_next_actions = torch.argmax(policy_net(non_final_next_states), dim=1).unsqueeze(1)  # Bx1

    y = torch.zeros(BATCH_SIZE)  # (B)
    y[not_final_mask] = GAMMA * target_net(non_final_next_states).gather(1, best_next_actions).squeeze(1).detach()
    y += reward_batch

    td_err = (y.unsqueeze(1) - q_values).abs().detach() + 1e-6  # Bx1
    buffer.update_priority(info['index'], td_err)
    loss = (info['_weight'] * criterion(q_values, y.unsqueeze(1)) * 10).mean()

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    opt.step()


# %%
num_episodes = 2000

total_rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)  # (4)

    done = False
    R = 0
    while not done:
        action = select_action(state)  # (1)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        # if terminated:
        #     reward = -10

        reward = torch.tensor(reward, dtype=torch.float32)  # (1)

        next_state = torch.tensor(next_state, dtype=torch.float32)

        buffer.add(Transition(state, action, next_state, reward, torch.tensor(terminated or truncated)))

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in target_net_state_dict:
            target_net_state_dict[key] = TAU * policy_net_state_dict[key] + (1 - TAU) * target_net_state_dict[key]
        R += reward

    total_rewards.append(R.item())
    print(f'episode: {episode}, R: {R.item()}')

plt.plot(total_rewards)
plt.show()

# %%
torch.save(policy_net.state_dict(), './cartpole.pth')
