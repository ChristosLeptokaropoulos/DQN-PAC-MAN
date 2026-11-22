# pacman_env.py — Το παιχνίδι: χάρτης, κινήσεις, φάντασμα, rewards
from collections import deque
import random
# ---------- Ο χάρτης ----------
LAYOUT = [
    "###################",
    "#........#........#",
    "#.######.#.######.#",
    "#.................#",
    "#.#.###########.#.#",
    "#.#.....P..G....#.#",
    "#.#.###########.#.#",
    "#.#...............#",
    "#.######.#.######.#",
    "#........#........#",
    "###################",
]

# ---------- Οι κινήσεις ----------
# Κάθε κίνηση είναι ένας αριθμός (0-3) που αντιστοιχεί σε (αλλαγή γραμμής, αλλαγή στήλης)
ACTIONS = {
    0: (-1, 0),   # πάνω
    1: (1, 0),    # κάτω
    2: (0, -1),   # αριστερά
    3: (0, 1),    # δεξιά
}

# ---------- Τα rewards ----------
REWARD_PELLET = 10     # έφαγε κουκκίδα
REWARD_GHOST = -100    # τον έπιασε το φάντασμα
REWARD_WIN = 100       # έφαγε όλες τις κουκκίδες
REWARD_STEP = -1       # κάθε βήμα κοστίζει λίγο

GHOST_RANDOM = 0.2   # στο 20% των γύρων κάνει τυχαία κίνηση αντί να σε κυνηγάει
GHOST_DELAY = 5      # πόσους γύρους περιμένει το φάντασμα στην αρχή

class PacmanEnv:
    def __init__(self, layout=LAYOUT):
        self.layout = layout
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.reset()

    def reset(self):
        # Διαβάζουμε τον χάρτη και βρίσκουμε πού είναι το καθετί
        self.walls = set()
        self.pellets = set()
        for r, line in enumerate(self.layout):
            for c, cell in enumerate(line):
                if cell == "#":
                    self.walls.add((r, c))
                elif cell == ".":
                    self.pellets.add((r, c))
                elif cell == "P":
                    self.pacman = (r, c)
                elif cell == "G":
                    self.ghost = (r, c)
        self.score = 0
        self.done = False
        self.done = False
        self.ghost_dir = (0,0)
        self.turn = 0             # μετρητής γύρων, ξεκινάει από το 0 σε κάθε νέο παιχνίδι

    
    def step(self, action):
            self.turn += 1            # ξεκίνησε νέος γύρος
            reward = REWARD_STEP

            # Κίνηση του Pacman
            dr, dc = ACTIONS[action]
            r, c = self.pacman
            new_pos = (r + dr, c + dc)
            if new_pos not in self.walls:
                self.pacman = new_pos

            # Έλεγχος 1: μπήκε ο Pacman στο κελί του φαντάσματος;
            if self.pacman == self.ghost:
                return self.lose()

            # Κουκκίδα
            if self.pacman in self.pellets:
                self.pellets.remove(self.pacman)
                reward += REWARD_PELLET
                self.score += 10

            # Νίκη
            if not self.pellets:
                self.done = True
                self.won = True
                return reward + REWARD_WIN, self.done

            # Κίνηση του φαντάσματος (σημείο 2: αμέσως μετά τον Pacman)
            if self.turn > GHOST_DELAY:
                self.move_ghost()

            # Έλεγχος 2: μπήκε το φάντασμα στο κελί του Pacman;
            if self.pacman == self.ghost:
                return self.lose()

            return reward, self.done

    def lose(self):
        self.done = True
        self.won = False
        return REWARD_GHOST, self.done

    def move_ghost(self):
        r, c = self.ghost
        back = (-self.ghost_dir[0], -self.ghost_dir[1])   # η κατεύθυνση «πίσω»

        # Όλες οι κινήσεις που επιτρέπονται: όχι τοίχος, όχι πίσω (σημεία 3 και 4)
        options = []
        for direction in ACTIONS.values():
            new_pos = (r + direction[0], c + direction[1])
            if new_pos in self.walls:
                continue
            if direction == back:
                continue
            options.append((direction, new_pos))

        # Αδιέξοδο: επιτρέπεται να γυρίσει πίσω
        if not options:
            options = [(back, (r + back[0], c + back[1]))]

        if random.random() < GHOST_RANDOM:
            choice = random.choice(options)            # τυχαία κίνηση
        else:
            pr, pc = self.pacman                        # κυνηγητό: το πιο κοντινό κελί
            choice = None
            best_dist = None
            for option in options:
                pos = option[1]
                dist = abs(pos[0] - pr) + abs(pos[1] - pc)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    choice = option

        self.ghost_dir, self.ghost = choice              # θυμάται την κατεύθυνση για τον επόμενο γύρο

    def nearest_pellet_dir(self):
        start = self.pacman
        queue = deque()
        seen = {start}

        # Πρώτο κύμα: τα 4 κουτάκια δίπλα του (απόσταση 1)
        for a in range(4):
            dr, dc = ACTIONS[a]
            nxt = (start[0] + dr, start[1] + dc)
            if nxt not in self.walls:
                seen.add(nxt)
                queue.append((nxt, a, 1))

        while queue:
            pos, first, dist = queue.popleft()
            if pos in self.pellets:
                return first, dist
            for dr, dc in ACTIONS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt not in self.walls and nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, first, dist + 1))

        return None, None   # δεν έμεινε καμία κουκίδα

    def get_state(self):
        r, c = self.pacman
        state = []

        # 1) Τοίχοι γύρω του (4 κουτάκια)
        for a in range(4):
            dr, dc = ACTIONS[a]
            state.append(1 if (r + dr, c + dc) in self.walls else 0)

        # 2) Προς τα πού είναι το φάντασμα (4 κουτάκια)
        gr, gc = self.ghost
        state.append(1 if gr < r else 0)   # πάνω μου
        state.append(1 if gr > r else 0)   # κάτω μου
        state.append(1 if gc < c else 0)   # αριστερά μου
        state.append(1 if gc > c else 0)   # δεξιά μου

        # 3) Πόσο κοντά είναι το φάντασμα (1 κουτάκι)
        dist = abs(gr - r) + abs(gc - c)
        state.append(1 - min(dist, 10) / 10)

        # 4) Έρχεται το φάντασμα προς το μέρος μου; (1 κουτάκι)
        dr, dc = self.ghost_dir
        next_ghost = (gr + dr, gc + dc)
        next_dist = abs(next_ghost[0] - r) + abs(next_ghost[1] - c)
        state.append(1 if next_dist < dist else 0)

        # 5) Προς τα πού είναι η πιο κοντινή κουκίδα (4 κουτάκια)
        first, pdist = self.nearest_pellet_dir()
        for a in range(4):
            state.append(1 if a == first else 0)

        # 6) Πόσο κοντά είναι η πιο κοντινή κουκίδα (1 κουτάκι)
        if pdist is None:
            state.append(0)
        else:
            state.append(1 - min(pdist, 10) / 10)

        return state