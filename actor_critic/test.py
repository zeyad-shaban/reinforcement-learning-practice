import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.actor = nn.Linear(64, out_channels)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.backbone(x)
        return self.actor(x), self.critic(x)


# Hyperparameters
LR = 0.001
BATCH_SIZE = 32
GAMMA = 0.99
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

env = gym.make('CartPole-v1')
n_obs = env.observation_space.shape[0]
n_acts = env.action_space.n

model = ActorCritic(n_obs, n_acts)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter('./runs')
mse_criterion = nn.MSELoss()

for epoch in range(1500):
    total_value_loss = 0.0
    total_actor_loss = 0.0
    total_entropy = 0.0
    episode_rewards = []

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    episode_reward = 0
    done = False

    for _ in range(BATCH_SIZE):
        logits, value = model(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward

        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward_t = torch.tensor(reward, dtype=torch.float32)

        if done:
            next_value = torch.tensor(0.0)
            # Reset environment but don't use this state in current batch
            next_state, _ = env.reset()
            next_state = torch.tensor(next_state, dtype=torch.float32)
            episode_rewards.append(episode_reward)
            episode_reward = 0
        else:
            with torch.no_grad():
                _, next_value = model(next_state)

        # Calculate TD target and advantage
        td_target = reward_t + GAMMA * next_value
        advantage = td_target - value

        # Calculate losses
        value_loss = mse_criterion(value, td_target.detach())
        actor_loss = -(advantage.detach() * dist.log_prob(action))
        entropy_loss = -ENTROPY_COEF * dist.entropy()

        total_value_loss += value_loss
        total_actor_loss += actor_loss
        total_entropy += entropy_loss

        state = next_state

    # Average losses and optimize
    avg_value_loss = total_value_loss / BATCH_SIZE
    avg_actor_loss = total_actor_loss / BATCH_SIZE
    avg_entropy = total_entropy / BATCH_SIZE
    total_loss = avg_value_loss + avg_actor_loss + avg_entropy

    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    optimizer.step()

    # Logging
    if episode_rewards:
        mean_reward = np.mean(episode_rewards)
        writer.add_scalar('reward/mean', mean_reward, epoch)
        writer.add_scalar('loss/value', avg_value_loss.item(), epoch)
        writer.add_scalar('loss/actor', avg_actor_loss.mean().item(), epoch)
        writer.add_scalar('loss/entropy', avg_entropy.mean().item(), epoch)

    print(f'Epoch {epoch}, Mean Reward: {mean_reward if episode_rewards else 0}')

torch.save(model.state_dict(), './a2c_fixed.pth')