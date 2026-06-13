import numpy as np
import random
import math

def perceptron():
    X = np.array([random.random() for _ in range(3)])
    W = np.array([random.random() for _ in range(3)])
    b = random.random()
    y = np.dot(X, W.T)
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
