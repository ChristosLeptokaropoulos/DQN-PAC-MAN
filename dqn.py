# dqn.py — Ο "εγκέφαλος" του agent: παίρνει το state (15 αριθμοί)
# και βγάζει ένα Q-value για κάθε κίνηση (4 αριθμοί).

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_inputs=15, n_actions=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),   # hidden layer 1: 15 → 64
            nn.ReLU(),
            nn.Linear(hidden, hidden),     # hidden layer 2: 64 → 64
            nn.ReLU(),
            nn.Linear(hidden, n_actions),  # output layer: 64 → 4, χωρίς ενεργοποίηση (linear)
        )

    def forward(self, x):
        # x: το state (ή πολλά states μαζί). Επιστρέφει τα Q-values.
        return self.net(x)