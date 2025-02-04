import numpy as np
import gymnasium as gym
import ale_py
import cv2
from torchvision import transforms
import torch
import torch.nn as nn

# Discriminator (sigmoid output)



# Generator (fully convolution layer)




class InputWrapper(gym.ObservationWrapper):
    IMG_SIZE = 150

    def __init__(self, env):
        super().__init__(env)
        self.transform = transforms.ToTensor()

        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(InputWrapper.IMG_SIZE, InputWrapper.IMG_SIZE), dtype=np.float32
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation, (InputWrapper.IMG_SIZE, InputWrapper.IMG_SIZE)
        )
        observation = observation.transpose(1, 2, 0)
        observation = self.transform(observation)

        return observation # 3 x 150 x 150


if __name__ == '__main__':
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = InputWrapper(env)
    
    observation, info = env.reset()
    total_reward = 0

    while True:
        env.render()

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        if terminated or truncated:
            observation, info = env.reset()
            break

    print(total_reward)
