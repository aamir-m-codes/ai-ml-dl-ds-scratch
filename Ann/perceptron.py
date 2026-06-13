import numpy as np
import random
import math


def perceptron():
    # generate inputs
    X = np.array([random.random() for _ in range(3)])
    # generate weights
    W = np.array([random.random() for _ in range(3)])
    # generate bias
    b = random.random()
    # matrix multiplication (for multiply every input with its corresponding weight)
    y = np.dot(X, W.T)
    # add bias
    y += b

    # apply activation function on y -> sigmoid formula: 1/(1 + e^-x)
    output = 1 / (1 + math.exp(-y))
    print(f"Final Output: {output}")


"""
  It's Optional
  this line of code ensure that the lines of code under this condition is execute
  only if this file is directly execute not by any other file
"""
if __name__ == "__main__":
    perceptron()
