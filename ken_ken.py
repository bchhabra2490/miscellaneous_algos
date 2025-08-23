import random


class KenKen:
    def __init__(self, size, cages):
        self.size = size
        self.cages = cages

    def valid(self, grid, r, c, val):
        if val in grid[r]:
            return False
        if val in [grid[i][c] for i in range(self.size)]:
            return False

        for cells, op, target in self.cages:
            if (r, c) in cells:
                vals = [grid[x][y] for x, y in cells if grid[x][y] != 0]
                vals.append(val)
                if len(vals) == len(cells):
                    if not self.check_cage(vals, op, target):
                        return False
        return True

    def check_cage(self, vals, op, target):
        if op == "+":
            return sum(vals) == target
        elif op == "*":
            prod = 1
            for v in vals:
                prod *= v
            return prod == target
        elif op == "-":
            return abs(vals[0] - vals[1]) == target
        elif op == "/":
            a, b = vals
            return max(a, b) // min(a, b) == target
        else:
            return vals[0] == target

    def solve(self, limit=2):
        grid = [[0] * self.size for _ in range(self.size)]
        solutions = []

        def backtrack(r=0, c=0):
            if r == self.size:
                solutions.append([row[:] for row in grid])
                return len(solutions) >= limit

            next_r, next_c = (r, c + 1) if c + 1 < self.size else (r + 1, 0)

            for val in range(1, self.size + 1):
                if self.valid(grid, r, c, val):
                    grid[r][c] = val
                    if backtrack(next_r, next_c):
                        return True
                grid[r][c] = 0
            return False

        backtrack()
        return solutions


def generate_latin_square(n):
    base = list(range(1, n + 1))
    square = []
    for i in range(n):
        row = base[i:] + base[:i]
        square.append(row)
    ## Shuffle the columns
    count = 0
    while count < 5:
        col_1 = random.randint(0, n - 1)
        col_2 = random.randint(0, n - 1)
        if col_1 != col_2:
            for i in range(n):
                square[i][col_1], square[i][col_2] = square[i][col_2], square[i][col_1]
            count += 1
    return square


def random_cages(solution):
    n = len(solution)
    cells = [(r, c) for r in range(n) for c in range(n)]
    random.shuffle(cells)

    cages = []
    used = set()

    while cells:
        cell = cells.pop()
        if cell in used:
            continue

        cage = [cell]
        used.add(cell)

        size = random.choice([1, 2, 3])

        for _ in range(size - 1):
            neighbors = [
                (r + dr, c + dc)
                for r, c in cage
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 < r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in used
            ]

            if not neighbors:
                break

            new_cell = random.choice(neighbors)
            cage.append(new_cell)
            used.add(new_cell)

        values = [solution[r][c] for r, c in cage]

        if len(values) == 1:
            op, target = None, values[0]
        else:
            op = random.choice(["+", "*"]) if len(values) > 2 else random.choice(["+", "*", "-", "/"])

            if op == "+":
                target = sum(values)
            elif op == "*":
                prod = 1
                for v in values:
                    prod *= v
                target = prod
            elif op == "-":
                target = abs(values[0] - values[1])
            elif op == "/":
                a, b = values
                target = max(a, b) // min(a, b)

        cages.append((cage, op, target))
    return cages


def generate_kenken(n):
    solution = generate_latin_square(n)
    print(solution)
    cages = random_cages(solution)
    return solution, cages


solution, cages = generate_kenken(5)
kenken = KenKen(5, cages)
solutions = kenken.solve(100)

for i, solution in enumerate(solutions):
    print(f"Solution {i+1}:")
    for row in solution:
        print(" ".join(f"{num:2d}" for num in row))
    print()
