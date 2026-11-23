# agent.py — Ο agent: έχει τον εγκέφαλο (DQN) και αποφασίζει ποια κίνηση θα κάνει.

import random
import torch
from dqn import DQN


class Agent:
    def __init__(self, n_inputs=15, n_actions=4,
                 eps_start=1.0, eps_end=0.1, eps_steps=20000):
        self.n_actions = n_actions
        self.model = DQN(n_inputs, n_actions)   # ο εγκέφαλος

        # ε-greedy: από 100% τυχαίες κινήσεις σε 10%, σε ευθεία γραμμή
        self.eps_start = eps_start     # ε στην αρχή (1.0 = 100%)
        self.eps_end = eps_end         # ε στο τέλος (0.1 = 10%)
        self.eps_steps = eps_steps     # σε πόσα βήματα φτάνει από το start στο end
        self.steps = 0                 # πόσες κινήσεις έχει κάνει μέχρι τώρα

    def epsilon(self):
        # Πόσο έχει προχωρήσει η πτώση: 0 = μόλις ξεκίνησε, 1 = τελείωσε
        progress = min(self.steps / self.eps_steps, 1.0)
        # Ευθεία γραμμή από eps_start σε eps_end
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def select_action(self, state):
        eps = self.epsilon()
        self.steps += 1

        # "Ζάρι": με πιθανότητα ε → τυχαία κίνηση (τρόπος Β, εξερεύνηση)
        if random.random() < eps:
            return random.randrange(self.n_actions)

        # Αλλιώς → ρωτάμε τον εγκέφαλο, παίρνουμε το μεγαλύτερο Q (τρόπος Α)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32)
            q = self.model(s)
            return q.argmax().item()