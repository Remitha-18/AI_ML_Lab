# Experiment 8
# Implementation of Perceptron and Backpropagation Neural Network

import numpy as np


# ==========================================
# PART 1: PERCEPTRON
# ==========================================

print("----- PERCEPTRON -----")

# AND gate input
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# AND gate output
y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

learning_rate = 0.1

# Training
for epoch in range(10):

    for i in range(len(X)):

        weighted_sum = np.dot(X[i], weights) + bias

        # Step activation function
        if weighted_sum >= 0:
            prediction = 1
        else:
            prediction = 0

        error = y[i] - prediction

        # Update weights and bias
        weights = weights + learning_rate * error * X[i]
        bias = bias + learning_rate * error


print("Final Weights:", weights)
print("Final Bias:", bias)

print("\nPerceptron Output:")

for i in range(len(X)):

    weighted_sum = np.dot(X[i], weights) + bias

    if weighted_sum >= 0:
        prediction = 1
    else:
        prediction = 0

    print(X[i], "->", prediction)


# ==========================================
# PART 2: BACKPROPAGATION NEURAL NETWORK
# ==========================================

print("\n----- BACKPROPAGATION NEURAL NETWORK -----")

# XOR input
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# XOR output
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Initialize weights
np.random.seed(42)

W1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.zeros((1, 2))

W2 = np.random.uniform(-1, 1, (2, 1))
b2 = np.zeros((1, 1))

learning_rate = 0.5


# ==========================================
# TRAINING
# ==========================================

for epoch in range(10000):

    # Forward propagation

    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    # Calculate error
    error = y - output

    # Backpropagation

    output_delta = error * sigmoid_derivative(output)

    hidden_error = np.dot(output_delta, W2.T)

    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    # Update weights
    W2 += np.dot(hidden_output.T, output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

    W1 += np.dot(X.T, hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


# ==========================================
# FINAL XOR OUTPUT
# ==========================================

print("\nXOR Output:")

for i in range(len(X)):

    hidden_input = np.dot(X[i], W1) + b1
    hidden_output = sigmoid(hidden_input)

    output_input = np.dot(hidden_output, W2) + b2
    output = sigmoid(output_input)

    # Convert NumPy array to a single number
    value = output.item()

    print(
        X[i],
        "->",
        round(value, 3),
        "Class:",
        int(value >= 0.5)
    )


print("\nTraining Completed Successfully!")