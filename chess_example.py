import numpy as np
import random
import chess
import chess.engine
import pickle
from multiprocessing import Pool, cpu_count


## Parameters

POPULATION_SIZE = 100
ELITE_COUNT = 10
GENERATIONS = 100
MATCHES_PER_INDIVIDUAL = 5
INPUT_FEATURES = 20
HIDDEN_LAYER = 64
OUTPUT_SIZE = 1

MUTATION_RATE = 0.1
MUTATION_STD = 0.02
ROLLOUT_DEPTH = 1

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


class NeuralNet:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        if weights is None:
            self.W1 = np.random.randn(hidden_size, input_size) * 0.1
            self.b1 = np.zeros(hidden_size)
            self.W2 = np.random.randn(hidden_size, hidden_size) * 0.1
            self.b2 = np.zeros(hidden_size)
            self.W3 = np.random.randn(output_size, hidden_size) * 0.1
            self.b3 = np.zeros(output_size)
        else:
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights

    def forward(self, x):
        h1 = np.maximum(0, self.W1.dot(x) + self.b1)
        h2 = np.maximum(0, self.W2.dot(h1) + self.b2)
        out = self.W3.dot(h2) + self.b3
        return out[0]

    def get_weights(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), self.W3.copy(), self.b3.copy())

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = weights


## GA Helpers


def crossover(parent1, parent2):
    child_weights = []
    for w1, w2 in zip(parent1.get_weights(), parent2.get_weights()):
        alpha = np.random.rand(*w1.shape)
        child_weights.append(alpha * w1 + (1 - alpha) * w2)
    return NeuralNet(INPUT_FEATURES, HIDDEN_LAYER, OUTPUT_SIZE, weights=tuple(child_weights))


def mutate(individual):
    new_weights = []
    for w in individual.get_weights():
        mask = np.random.rand(*w.shape) < MUTATION_RATE
        w_new = w + mask * np.random.randn(*w.shape) * MUTATION_STD
        new_weights.append(w_new)

    individual.set_weights(tuple(new_weights))
    return individual


## Feature extraction
def extract_features(board):
    def count_pieces(board, color):
        counts = []
        for piece_type, max_count in zip(
            [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING], [8, 2, 2, 2, 1, 1]
        ):
            counts.append(len(board.pieces(piece_type, color)) / max_count)
        return counts

    player = board.turn
    opponent = not board.turn
    features = count_pieces(board, player) + count_pieces(board, opponent)

    # Mobility
    features.append(len(list(board.legal_moves)) / 50)
    board.push(chess.Move.null())
    features.append(len(list(board.legal_moves)) / 50)
    board.pop()

    # Castling rights
    features.append(int(board.has_kingside_castling_rights(player)))
    features.append(int(board.has_queenside_castling_rights(player)))

    # Check status
    features.append(int(board.is_check()))
    board.push(chess.Move.null())
    features.append(int(board.is_check()))
    board.pop()

    # Squares attacked
    def attacked_squares(board, color):
        attacked = chess.SquareSet()
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                attacked |= board.attacks(square)
        return len(attacked) / 64

    features.append(attacked_squares(board, player))
    features.append(attacked_squares(board, opponent))

    return np.array(features, dtype=float)


def choose_move_fast(board, nn, rollout_depth=ROLLOUT_DEPTH):
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    features_batch = []
    for move in legal_moves:
        board.push(move)
        features_batch.append(extract_features(board))
        board.pop()

    scores = np.array([nn.forward(f) for f in features_batch])
    best_idx = np.argmax(scores)
    return legal_moves[best_idx]


def play_game(individual, population, generation, engine_path=None):
    board = chess.Board()
    if False and generation < 10 and engine_path:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        while not board.is_game_over():
            move = choose_move_fast(board, individual)
            if move is None:
                break
            board.push(move)
            if board.is_game_over():
                break
            result = engine.play(board, chess.engine.Limit(time=0.01))
            board.push(result.move)
        engine.quit()
    else:
        opponent = random.choice(population)
        while opponent == individual:
            opponent = random.choice(population)
        current_player = individual
        while not board.is_game_over():
            move = choose_move_fast(board, current_player)
            if move is None:
                break
            board.push(move)
            current_player = opponent if current_player == individual else individual
    # Evaluate game outcome
    result = board.result()
    if result == "1-0":
        return 1 if generation < 10 else (1 if current_player == individual else 0)
    elif result == "0-1":
        return 0 if generation < 10 else (1 if current_player == opponent else 0)
    else:
        return 0.5


def fitness_worker(args):
    individual, population, generation, engine_path = args
    total = 0
    for _ in range(MATCHES_PER_INDIVIDUAL):
        total += play_game(individual, population, generation, engine_path)
    return total


def main():
    population = [NeuralNet(INPUT_FEATURES, HIDDEN_LAYER, OUTPUT_SIZE) for _ in range(POPULATION_SIZE)]
    history = []

    for gen in range(GENERATIONS):
        args_list = [(ind, population, gen, STOCKFISH_PATH) for ind in population]
        with Pool(cpu_count()) as pool:
            fitness_scores = pool.map(fitness_worker, args_list)

        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        history.append((avg_fitness, max_fitness))
        print(f"Gen {gen+1}: Avg Fitness={avg_fitness:.2f}, Max Fitness={max_fitness:.2f}")

        # Elitism + reproduction
        sorted_indices = np.argsort(fitness_scores)[::-1]
        elites = [population[i] for i in sorted_indices[:ELITE_COUNT]]
        new_population = elites.copy()
        while len(new_population) < POPULATION_SIZE:
            p1, p2 = random.sample(elites, 2)
            child = crossover(p1, p2)
            child = mutate(child)
            new_population.append(child)
        population = new_population

    fitness_scores = [fitness_worker((ind, population, GENERATIONS - 1, None)) for ind in population]
    top3_idx = np.argsort(fitness_scores)[-3:][::-1]
    top3_weights = [population[i].get_weights() for i in top3_idx]
    with open("top3_weights.pkl", "wb") as f:
        pickle.dump(top3_weights, f)
    print("\nSaved weights of top 3 individuals to 'top3_weights.pkl'")

    # ----------------------------
    # 10️⃣ Fitness History Logging
    # ----------------------------
    print("\nFitness History:")
    for gen, (avg, max_) in enumerate(history, 1):
        improvement = max_ - history[gen - 2][1] if gen > 1 else 0
        print(f"Gen {gen}: Avg={avg:.2f}, Max={max_:.2f}, Improvement={improvement:.2f}")


if __name__ == "__main__":
    try:
        from multiprocessing import set_start_method

        set_start_method("spawn", force=True)
    except Exception:
        pass
    main()
