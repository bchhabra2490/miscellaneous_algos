import random


def monte_carlo_pi(num_samples=1_000_000):
    inside_circle = 0
    for _ in range(num_samples):
        x, y = random.random(), random.random()  # random point in [0,1] x [0,1]
        if x * x + y * y <= 1:  # inside quarter circle
            inside_circle += 1
    return 4 * inside_circle / num_samples


print(monte_carlo_pi())
