import torch
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F


env = gym.make('CartPole-v1', render_mode='human')
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n


# %%

# class Policy(nn.Module):
#     """
#     implements both actor and critic in one model
#     """

#     def __init__(self):
#         super(Policy, self).__init__()
#         self.affine1 = nn.Linear(4, 128)
#         self.action_head = nn.Linear(128, 2)
#         self.value_head = nn.Linear(128, 1)
#         self.saved_actions = []
#         self.rewards = []

#     def forward(self, x):
#         x = F.relu(self.affine1(x))
#         action_prob = F.softmax(self.action_head(x), dim=-1)
#         state_values = self.value_head(x)
#         return action_prob, state_values



class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(64, 2)
        self.value_head = nn.Linear(64, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        y = self.body(x)

        action = self.action_head(y)
        value = self.value_head(y)

        return action, value


net = Policy()
net.load_state_dict(torch.load('dummy.pth', weights_only=True))

while True:
    state, _ = env.reset()

    R = 0
    while True:
        state = torch.tensor(state)[None]
        action = torch.argmax(net(state)[0]).item()
        state, reward, terminated, truncated, _ = env.step(action)

        R += reward

        if terminated or truncated:
            break

    print(R)
