import gymnasium as gym
import torch
import torch.nn as nn
import time

model = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),

    nn.Linear(100, 50),
    nn.ReLU(),
    
    nn.Linear(50, 25),
    nn.ReLU(),
    
    nn.Linear(25, 4),
)
model.load_state_dict(torch.load('/home/zeyadcode/workspace/Sandbox/reinforcement/frozen_lake_killer/weights.pth', weights_only=True))

if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='human')

    obs, _ = env.reset()
    while True:
        env.render()
        obs_t = torch.tensor([obs], dtype=torch.float32) # Shape (1)
        action = torch.argmax(model(obs_t)).item()
        
        next_obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            break

        obs = next_obs
        time.sleep(0.1)