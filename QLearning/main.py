import gymnasium as gym
import numpy as np
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt

ALPHA = 0.2
GAMMA = 0.98
EPSILON = 0.2
LIVING_PENALITY = 1000
EPISODES = 10000

videos_dir = '/home/zeyadcode/workspace/Sandbox/reinforcement/QLearning/videos'
q_table_path = '/home/zeyadcode/workspace/Sandbox/reinforcement/QLearning/q_table.npy'


def get_action(Q_table: np.ndarray, obs, epsilon=EPSILON) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(Q_table.shape[1])

    return np.argmax(Q_table[obs])


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', render_mode='rgb_array')
    env = gym.wrappers.RecordVideo(env, videos_dir)

    Q_table = np.zeros((env.observation_space.n, env.action_space.n))

    for episode in range(EPISODES):
        obs, _ = env.reset()
        while True:
            action = get_action(Q_table, obs)
            new_obs, reward, terminated, truncated, _ = env.step(action)

            reward *= 50
            reward -= LIVING_PENALITY

            Q_table[obs, action] = Q_table[obs, action] + ALPHA * (
                reward + GAMMA * np.max(Q_table[new_obs]) - Q_table[obs, action]
            )

            obs = new_obs

            if terminated or truncated:
                break

    env.close()

    np.save(q_table_path, Q_table)

    # render for human
    env = gym.make('FrozenLake-v1', render_mode='human')
    Q_table = np.load('/home/zeyadcode/workspace/Sandbox/reinforcement/QLearning/q_table.npy')
    while True:
        obs, _ = env.reset()
        while True:
            action = get_action(Q_table, obs, epsilon=0)
            new_obs, reward, terminated, truncated, _ = env.step(action)

            obs = new_obs
            env.render()
            time.sleep(0.001)

            if terminated or truncated:
                time.sleep(0.5)
                break