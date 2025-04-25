# This updated version adds a bias term to the model.
# The prediction function becomes: prediction = weight + bias
# Both parameters (weight, bias) require gradients and get updated via gradient descent.

import numpy as np

# Set the real value
target_value = 0.81

# Initialize parameters (weight and bias) with random guesses
weight = np.random.rand()
bias = np.random.rand()

# Learning rate
lr = 0.1

# Number of iterations
epochs = 25

# Training loop
for epoch in range(epochs):
    # Compute prediction with weight and bias
    prediction = weight + bias

    # Compute the error
    error = prediction - target_value

    # Compute MSE loss
    loss = error ** 2

    # Compute gradients
    dL_dw = 2 * error      # ∂L/∂w
    dL_db = 2 * error      # ∂L/∂b (same because both w and b affect output linearly)

    # Update parameters
    weight -= lr * dL_dw
    bias   -= lr * dL_db

    # Print progress
    print(f"Epoch {epoch+1}: Weight={weight:.5f}, Bias={bias:.5f}, Loss={loss:.5f}")

# Final output
print(f"\nFinal estimated weight: {weight:.5f}")
print(f"Final estimated bias: {bias:.5f}")
print(f"Target value: {target_value:.5f}")
print(f"Final prediction: {weight + bias:.5f}")
print(f"Difference: {target_value - (weight + bias):.5f}")
