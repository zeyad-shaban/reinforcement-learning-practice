# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import gymnasium as gym
import torch.optim as optim
from collections import namedtuple, deque
import random
import random

# %%
LR = 1e-3
EPS_END = 0.05
EPS_START = 1
BATCH_SIZE = 128  # expirement with this more
EPS_DECAY = 5000  # does this cause faster or slower update?
TAU = 0.005
GAMMA = 0.99

# %%
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def append(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, i):
        return self.memory[i]


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
env = gym.make("CartPole-v1")

n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]

policy_net = DQN(n_observations, n_actions)
target_net = DQN(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

opt = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
criterion = nn.HuberLoss()
replay_memory = ReplayMemory(2000)

# %%
steps_done = 0


def select_action(state):
    global steps_done

    eps = EPS_END + (EPS_START - EPS_END) * np.exp(
        -1.0 * steps_done / EPS_DECAY
    )  # try to look at how eps is updated
    steps_done += 1

    if random.random() < eps:
        return torch.tensor([[env.action_space.sample()]])

    with torch.no_grad():  # attempt without this
        # why this weird shape? why not get the item directly
        return torch.argmax(policy_net(state)).view(1, 1)


# %%
def optimize_model():
    if len(replay_memory) < BATCH_SIZE:
        return

    transitions = replay_memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        list(map(lambda state: state is not None, batch.next_state))
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)  # Bx4
    action_batch = torch.cat(batch.action)  # Bx1
    reward_batch = torch.cat(batch.reward)  # B

    q_values = policy_net(state_batch).gather(1, action_batch)  # 128x1

    next_q_values = torch.zeros(BATCH_SIZE)  # 128
    next_q_values[non_final_mask] = (
        target_net(non_final_next_states).max(1).values.detach()
    )  # 128

    expected_q_values = reward_batch + GAMMA * next_q_values  # 128

    loss = criterion(q_values, expected_q_values)

    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    opt.step()


# %%
num_episodes = 2000

total_rewards = []
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 4)

    done = False
    R = 0
    while not done:
        action = select_action(state)  # (1, 1)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        if terminated:
            reward = -10

        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)  # (1, 1)

        next_state = (
            None if done else torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        )

        replay_memory.append(state, action, next_state, reward)

        state = next_state

        optimize_model()

        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()

        for key in target_net_state_dict:
            target_net_state_dict[key] = (
                TAU * policy_net_state_dict[key]
                + (1 - TAU) * target_net_state_dict[key]
            )
        R += reward

    total_rewards.append(R.item())
    print(f"episode: {episode}, R: {R.item()}")

plt.plot(total_rewards)
plt.show()
