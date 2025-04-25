import numpy as np

# Set the real value
target_value = 0.81

# Initialize the weight with a random guess
weight = np.random.rand()  # Random value between 0 and 1

# Learning rate
lr = 0.1

# Number of iterations
epochs = 25

# Training loop
for epoch in range(epochs):
    # Compute prediction (in this case, the weight itself)
    prediction = weight

    # Compute the error
    error = prediction - target_value

    # Compute MSE loss
    loss = error ** 2

    # Compute gradient of loss w.r.t. weight (derivative of MSE)
    gradient = 2 * error

    # Update weight using gradient descent
    weight -= lr * gradient

    # Print progress
    print(f"Epoch {epoch+1}: Weight={weight:.5f}, Loss={loss:.5f}")

print(f"\nFinal estimated weight: {weight:.5f}")
print(f"Target value: {target_value:.5f}\n")

print(f"Difference: {target_value-weight:.5f}\n\n")
