import torch
import torch.nn as nn
import gymnasium as gym
import time

model = nn.Sequential(
    nn.Linear(4, 100),
    nn.ReLU(),

    nn.Linear(100, 50),
    nn.ReLU(),

    nn.Linear(50, 25),
    nn.ReLU(),

    nn.Linear(25, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load('/home/zeyadcode/workspace/Sandbox/reinforcement/cartpole_beater/model.pth', weights_only=True))
model.eval()

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')

    while True:
        obs, _ = env.reset()
        R = 0

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32).view(1, -1)
            action = int(torch.round(model(obs_t)).item())
            obs, reward, terminated, truncated, info = env.step(action)

            R += reward
            env.render()
            time.sleep(0.01)

            if terminated or truncated:
                print(f'truncated: {truncated}, termianted: {terminated}')
                break

        print(R)
