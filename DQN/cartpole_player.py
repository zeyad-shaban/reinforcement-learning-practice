import gymnasium as gym
import time
import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    #
    nn.Linear(128, 64),
    nn.ReLU(),
    #
    nn.Linear(64, 32),
    nn.ReLU(),
    #
    nn.Linear(32, 16),
    nn.ReLU(),
    #
    nn.Linear(16, 2)
)

net.load_state_dict(torch.load('./cartpole_dqn.pth', weights_only=True))

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
