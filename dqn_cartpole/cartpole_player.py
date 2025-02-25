import gymnasium as gym
import time
import torch
import torch.nn as nn


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
    
net = DQN(4, 2)

net.load_state_dict(torch.load('./cartpole.pth', weights_only=True))

env = gym.make('CartPole-v1', render_mode='human')

while True:
    obs, _ = env.reset()
    R = 0

    while True:
        state = torch.tensor(obs, dtype=torch.float32)[None]
        action = torch.argmax(net(state)).item()

        
        next_obs, reward, terminated, truncated, _ = env.step(action)
        R += reward

        if terminated or truncated:
            time.sleep(1)
            break
        

        env.render()
        time.sleep(0.01)
        obs = next_obs
    print(R)
