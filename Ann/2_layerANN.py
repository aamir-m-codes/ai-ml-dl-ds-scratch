import numpy as np
import math
import random


def two_hidden_layer():
    X = np.array([random.random() for _ in range(3)])
    W1 = np.array(
        [
            [random.random() for _ in range(3)],
            [random.random() for _ in range(3)],
            [random.random() for _ in range(3)],
        ]
    )

    B1 = np.array([random.random() for _ in range(3)])
    X_W1 = np.dot(X, W1.T)

    H1 = X_W1 + B1
    W2 = np.array(
        [
            [random.random() for _ in range(3)],
            [random.random() for _ in range(3)],
        ]
    )

    B2 = np.array([random.random() for _ in range(2)])
    H1_W2 = np.dot(H1, W2.T)

    H2 = H1_W2 + B2
    W3 = np.array([random.random() for _ in range(2)])

    b3 = random.random()
    H2_W3 = np.dot(H2, W3.T)

    y = H2_W3 + b3

    output = 1 / (1 + math.exp(-y))
    print(f"Final Output: {output}")


if __name__ == "__main__":
    two_hidden_layer()
