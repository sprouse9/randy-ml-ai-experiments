# This updated version adds an input term to the model.
# The prediction function becomes: prediction = weight*input + bias
# Both parameters (weight, bias) require gradients and get updated via gradient descent.

import numpy as np

# Set the real value
target_value = 0.81

# Initialize parameters (weight and bias) with random guesses
weight = np.random.rand()
bias = np.random.rand()
input = np.random.rand()

# Learning rate
lr = 0.001

# Number of iterations
epochs = 500

# Training loop
for epoch in range(epochs):
    # Compute prediction with weight and bias
    prediction = weight*input + bias
    
    # Compute the error
    error = prediction - target_value

    # Compute MSE loss
    loss = error ** 2

    # Compute gradients
    dL_dw = 2 * error * input      # ∂L/∂w
    dL_db = 2 * error              # ∂L/∂b (same because both w and b affect output linearly)
    # dL_dx = 2 * error * weight   # ∂L/∂x (not used)

    # Update parameters
    weight -= lr * dL_dw
    # input  -= lr * dL_dx    # this was an error. we don't update the input value
    bias   -= lr * dL_db
    

    # Print progress
    # print(f"Epoch {epoch+1}: Weight={weight:.5f}, Input={input:.5f}, Bias={bias:.5f}, Loss={loss:.5f}")

# Final output
print(f"\nFinal estimated weight: {weight:.5f}")
print(f"\nFinal estimated input(x): {input:.5f}")
print(f"Final estimated bias: {bias:.5f}")
print(f"Target value: {target_value:.5f}")
print(f"Final prediction: {weight + bias:.5f}")
print(f"Difference: {target_value - (weight + bias):.5f}")
