# This version displays the Gradient Descent Convergence graph

import numpy as np
import matplotlib.pyplot as plt

# Set the real value
target_value = 0.81

# Initialize the weight with a random guess
weight = np.random.rand()  # Random value between 0 and 1

# Learning rate
lr = 0.01

# Number of iterations
epochs = 300

# Store weight updates for plotting
weights_history = []

# Training loop
for epoch in range(epochs):

    weights_history.append(weight)  # Store current weight

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


# Plot the convergence
plt.plot(weights_history, label="Estimated Weight")
plt.axhline(y=target_value, color='r', linestyle='--', label="Target Value")
plt.xlabel("Epochs")
plt.ylabel("Weight Value")
plt.legend()
plt.title("Gradient Descent Convergence")
plt.show()



