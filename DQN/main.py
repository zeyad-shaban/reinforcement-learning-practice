import gymnasium as gym
import time
import torch
import torch.nn as nn


env = gym.make("FrozenLake-v1", render_mode="human")
n_states = env.observation_space.n
n_actions = env.action_space.n


class DQN(nn.Module):
    def __init__(self, inp_size, out_size):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(inp_size, 128),
            nn.ReLU(),
            #
            nn.Linear(128, 64),
            nn.ReLU(),
            #
            nn.Linear(64, 32),
            nn.ReLU(),
            #
            nn.Linear(32, 4),
        )

    def forward(self, x):
        return self.layer(x)


dqn = DQN(n_states, n_actions)
dqn.load_state_dict(
    torch.load(
        "/home/zeyadcode/workspace/Sandbox/reinforcement/DQN/dqn.pth", weights_only=True
    )
)

win, loss = 0, 0
for i in range(1000):
    obs, _ = env.reset()
    while True:
        obs = nn.functional.one_hot(torch.tensor(obs), n_states).to(torch.float32)
        action = torch.argmax(dqn(obs)).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            # time.sleep(1)
            if reward > 0:
                win += 1
            else:
                loss += 1
            break

        obs = next_obs

        # time.sleep(0.05)
        # env.render()

env.close()

print(f"Wins: {win}, Losses: {loss}")
