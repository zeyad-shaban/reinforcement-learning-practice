import torch
import gymnasium as gym
import torch.nn as nn


env = gym.make('CartPole-v1', render_mode='human')
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

net = nn.Sequential(
    nn.Linear(n_observations, 128),
    nn.ReLU(),
    #
    nn.Linear(128, 64),
    nn.ReLU(),
    #
    nn.Linear(64, n_actions),
)

net.load_state_dict(torch.load('reinforce4.pth', weights_only=True))


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