import numpy as np
import random
import math


def one_hidden_layer():
    """"""
    # generate 3 inputs
    X = np.array([random.random() for _ in range(3)])
    # generate weights (each input node has 2 weights for pointing to 2 nodes of hidden layer) so 3 nodes (starting) * 2 nodes(hidden layer) = 6 (lines with weights)
    W1 = np.array(
        [
            [random.random() for _ in range(3)],
            [random.random() for _ in range(3)],
        ]
    )

    # generate 2 biases for each node of hidden layer
    B1 = np.array([random.random() for _ in range(2)])

    # matrix multiplication for multplying every node input with its correspond weights of lines
    X_W1 = np.dot(X, W1.T)

    # add both biases with output of first 3 nodes to make input of hidden layer's nodes
    H = X_W1 + B1

    # generate the weights of hidden layer nodes lines that pointing to final node
    W2 = np.array([random.random() for _ in range(2)])
    # generate the bias for final node
    b2 = random.random()

    # matrix multiplication (for multiply every input with its corresponding weight)
    H_W2 = np.dot(H, W2.T)

    # adding the bias to final node

    y = H_W2 + b2
    # apply activation function on y -> sigmoid formula: 1/(1 + e^-x)
    output = 1 / (1 + math.exp(-y))
    print(f"Final Output: {output}")


if __name__ == "__main__":
    one_hidden_layer()
