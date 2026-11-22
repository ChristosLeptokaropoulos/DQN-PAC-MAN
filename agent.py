"""Replay buffer, ε-greedy επιλογή ενέργειας, βήμα εκπαίδευσης."""
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from dqn import DQN


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        n_actions,
        lr=1e-3,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=20000,
        batch_size=64,
        buffer_size=50000,
        target_update=500,
        device="cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)

        self.policy = DQN(obs_dim, n_actions).to(self.device)
        self.target = DQN(obs_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.steps = 0
        self.learn_steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, greedy=False):
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy(s).argmax(dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, s2, d = self.memory.sample(self.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        d = torch.as_tensor(d, device=self.device)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target(s2).max(dim=1).values
            target = r + self.gamma * (1.0 - d) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
        self.target.load_state_dict(self.policy.state_dict())
