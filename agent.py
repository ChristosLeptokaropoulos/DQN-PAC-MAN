# agent.py — Ο agent: έχει τον εγκέφαλο (DQN) και αποφασίζει ποια κίνηση θα κάνει.

import random
import torch
from dqn import DQN
from collections import deque

class ReplayBuffer:
    # Η αποθήκη: κρατάει τις τελευταίες κινήσεις του agent.
    # Κάθε κίνηση = (state, action, reward, next_state, done)

    def __init__(self, capacity=50000):
        # Λίστα με όριο: όταν γεμίσει, η πιο παλιά κίνηση φεύγει αυτόματα
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Βάζει μία νέα κίνηση στο τέλος της αποθήκης
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        # Τραβάει batch_size τυχαίες, διαφορετικές κινήσεις
        batch = random.sample(self.buffer, batch_size)

        # Από 32 πεντάδες → 5 ομάδες των 32 (όλα τα states μαζί, όλα τα actions μαζί, ...)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Μετατροπή σε tensors για να τα καταλαβαίνει το PyTorch
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        # Πόσες κινήσεις έχει η αποθήκη. Μας χρειάζεται για να ξέρουμε
        # αν έχουμε μαζέψει αρκετές για ένα batch (η απάντηση β που έδωσες)
        return len(self.buffer)

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