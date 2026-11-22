# render.py — Εμφάνιση του παιχνιδιού στην οθόνη
import pygame
from pacman_env import LAYOUT, PacmanEnv, ACTIONS as ACTION_DIRS

TILE = 40  # πόσα pixels είναι κάθε κελί του χάρτη

# Χρώματα σε (Κόκκινο, Πράσινο, Μπλε), από 0 έως 255
BLACK = (0, 0, 0)
BLUE = (33, 33, 255)
YELLOW = (255, 255, 0)
PINK = (255, 184, 255)
WHITE = (255, 255, 255)


def is_wall(row, col):
    # Ό,τι είναι έξω από τον χάρτη το μετράμε σαν τοίχο
    if row < 0 or row >= len(LAYOUT) or col < 0 or col >= len(LAYOUT[0]):
        return True
    return LAYOUT[row][col] == "#"


def draw_wall(screen, row, col):
    x, y = col * TILE, row * TILE
    # Ζωγραφίζουμε γραμμή μόνο στις πλευρές που ακουμπάνε διάδρομο
    if not is_wall(row - 1, col):   # από πάνω διάδρομος
        pygame.draw.line(screen, BLUE, (x, y), (x + TILE, y), 3)
    if not is_wall(row + 1, col):   # από κάτω διάδρομος
        pygame.draw.line(screen, BLUE, (x, y + TILE - 1), (x + TILE, y + TILE - 1), 3)
    if not is_wall(row, col - 1):   # αριστερά διάδρομος
        pygame.draw.line(screen, BLUE, (x, y), (x, y + TILE), 3)
    if not is_wall(row, col + 1):   # δεξιά διάδρομος
        pygame.draw.line(screen, BLUE, (x + TILE - 1, y), (x + TILE - 1, y + TILE), 3)


def draw_pacman(screen, cx, cy, action):
    r = TILE // 2 - 4
    pygame.draw.circle(screen, YELLOW, (cx, cy), r)
    dr, dc = ACTION_DIRS[action]        # προς τα πού κοιτάει
    px, py = -dr, dc                    # η κάθετη κατεύθυνση (για το άνοιγμα του στόματος)
    tip_x, tip_y = cx + dc * (r + 1), cy + dr * (r + 1)
    mouth = [
        (cx, cy),
        (tip_x + px * (r // 2), tip_y + py * (r // 2)),
        (tip_x - px * (r // 2), tip_y - py * (r // 2)),
    ]
    pygame.draw.polygon(screen, BLACK, mouth)


def draw_ghost(screen, cx, cy):
    r = TILE // 2 - 4
    pygame.draw.circle(screen, PINK, (cx, cy - 2), r)            # κεφάλι
    pygame.draw.rect(screen, PINK, (cx - r, cy - 2, 2 * r, r))   # σώμα
    for dx in (-6, 6):                                           # δύο μάτια
        pygame.draw.circle(screen, WHITE, (cx + dx, cy - 4), 5)
        pygame.draw.circle(screen, BLUE, (cx + dx + 2, cy - 4), 2)


def draw_board(screen, env, action):
    # Τοίχοι: δεν αλλάζουν ποτέ, τους παίρνουμε από τον χάρτη
    for row, line in enumerate(LAYOUT):
        for col, cell in enumerate(line):
            if cell == "#":
                draw_wall(screen, row, col)

    # Κουκκίδες: μόνο όσες έχουν μείνει
    for pos in env.pellets:
        pygame.draw.circle(screen, YELLOW, center(pos), 4)

    cx, cy = center(env.pacman)
    draw_pacman(screen, cx, cy, action)
    gx, gy = center(env.ghost)
    draw_ghost(screen, gx, gy)

def center(pos):
    # Από θέση (γραμμή, στήλη) στο κέντρο του κελιού σε pixels (x, y)
    row, col = pos
    return col * TILE + TILE // 2, row * TILE + TILE // 2

def main():
    pygame.init()
    env = PacmanEnv()
    width = env.cols * TILE
    height = env.rows * TILE + 40          # +40 pixels από κάτω για το σκορ
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("DQN Pac-Man")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 24, bold=True)

    KEYS = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
    action = 3            # ξεκινάει κοιτώντας δεξιά
    MOVE_EVERY = 200      # κάθε πόσα milliseconds κάνει ένα βήμα
    last_move = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in KEYS:
                    action = KEYS[event.key]
                elif event.key == pygame.K_r:
                    env.reset()
                    action = 3

        now = pygame.time.get_ticks()
        if not env.done and now - last_move >= MOVE_EVERY:
            env.step(action)
            last_move = now

        screen.fill(BLACK)
        draw_board(screen, env, action)
        text = font.render(f"SCORE {env.score}", True, WHITE)
        screen.blit(text, (10, env.rows * TILE + 8))
        if env.done:
            if env.won:
                msg = font.render("YOU WIN!  (R = restart)", True, YELLOW)
            else:
                msg = font.render("GAME OVER  (R = restart)", True, PINK)
            screen.blit(msg, (width - msg.get_width() - 10, env.rows * TILE + 8))
        
        pygame.display.flip()     # ← λείπει: δείξε στην οθόνη ό,τι ζωγραφίσαμε
        clock.tick(60)  

    pygame.quit()
if __name__ == "__main__":
    main()