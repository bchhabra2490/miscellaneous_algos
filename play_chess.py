import pygame
import chess
import pickle
import numpy as np
from chess_example import NeuralNet, choose_move_fast, extract_features, HIDDEN_LAYER, INPUT_FEATURES, OUTPUT_SIZE

# ----------------------------
# 1️⃣ Load Best NN
# ----------------------------
with open("top3_weights.pkl", "rb") as f:
    top3_weights = pickle.load(f)

best_nn = NeuralNet(INPUT_FEATURES, HIDDEN_LAYER, OUTPUT_SIZE, weights=top3_weights[0])

# ----------------------------
# 2️⃣ Pygame Setup
# ----------------------------
pygame.init()
WIDTH, HEIGHT = 512, 512
SQUARE_SIZE = WIDTH // 8
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Play vs GA NN Chess")

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (100, 200, 100)

# Load piece images
PIECES = {}
# Map chess symbols to actual filenames in images/
SYMBOL_TO_FILE = {
    "P": "wp.png",
    "N": "wn.png",
    "B": "wb.png",
    "R": "wr.png",
    "Q": "wq.png",
    "K": "wk.png",
    "p": "p.png",
    "n": "n.png",
    "b": "b.png",
    "r": "r.png",
    "q": "q.png",
    "k": "k.png",
}

for symbol, filename in SYMBOL_TO_FILE.items():
    PIECES[symbol] = pygame.transform.scale(pygame.image.load(f"images/{filename}"), (SQUARE_SIZE, SQUARE_SIZE))


# ----------------------------
# 3️⃣ Draw Board
# ----------------------------
def draw_board(board, selected_square=None):
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(SCREEN, color, rect)
            if selected_square == (row, col):
                pygame.draw.rect(SCREEN, HIGHLIGHT, rect, 4)

            # Draw pieces
            square = chess.square(col, 7 - row)  # chess board is 0 bottom-left
            piece = board.piece_at(square)
            if piece:
                SCREEN.blit(PIECES[piece.symbol()], rect)


# ----------------------------
# 4️⃣ Random Player
# ----------------------------
def get_random_move(board):
    """Get a random legal move from the current position"""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None
    return random.choice(legal_moves)


# ----------------------------
# 5️⃣ Main Game Loop
# ----------------------------
def play_ai_vs_random_gui(nn, ai_color=chess.WHITE):
    board = chess.Board()
    running = True
    clock = pygame.time.Clock()

    # Add a small delay to make moves visible
    move_delay = 1000  # milliseconds
    last_move_time = 0

    while running:
        clock.tick(30)
        current_time = pygame.time.get_ticks()

        # Handle events (only quit for now)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw board
        draw_board(board)
        pygame.display.flip()

        # Make moves with delay
        if current_time - last_move_time > move_delay and not board.is_game_over():
            if board.turn == ai_color:
                # AI move
                move = choose_move_fast(board, nn)
                if move:
                    board.push(move)
                    print(f"AI ({'White' if ai_color else 'Black'}): {move}")
            else:
                # Random player move
                move = get_random_move(board)
                if move:
                    board.push(move)
                    print(f"Random ({'White' if not ai_color else 'Black'}): {move}")

            last_move_time = current_time

        if board.is_game_over():
            result = board.result()
            print(f"Game Over! Result: {result}")
            if result == "1-0":
                winner = "White"
            elif result == "0-1":
                winner = "Black"
            else:
                winner = "Draw"
            print(f"Winner: {winner}")
            running = False


# ----------------------------
# 6️⃣ Run the GUI
# ----------------------------
import random

play_ai_vs_random_gui(best_nn, ai_color=chess.WHITE)
pygame.quit()
