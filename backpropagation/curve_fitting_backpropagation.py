import numpy as np
import matplotlib.pyplot as plt

# Generate x values (from -10 to 10)
x = np.linspace(-10, 10, 100)

# True function: y = 2x^2 + 3x + 5 + noise
true_a, true_b, true_c = 2, 3, 5  # The real coefficients
noise = np.random.normal(0, 3, size=x.shape)  # Add some noise
y = true_a * x**2 + true_b * x + true_c + noise  # NumPy applies element-wise operations (vectorization)

# Plot the generated data
plt.scatter(x, y, label="Noisy Data", color='blue', alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
#plt.show()

def quadratic_model(x, a, b, c):
    return a * x**2 + b * x + c


# We need a way to measure how well our model fits the data.
# A common choice is Mean Squared Error (MSE).
# 
# y_true: The actual noisy data points.
# y_pred: The model’s predictions.
# Returns: The average squared difference (lower is better).
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)



test_a, test_b, test_c = 1.5, 2.5, 4.5  # Random coefficients
y_pred = quadratic_model(x, test_a, test_b, test_c)
loss = mean_squared_error(y, y_pred)
print(f"Test Loss: {loss:.4f}")

# Now let’s move on to Gradient Descent. This is where we will manually compute 
# the gradients and update the coefficients to minimize the loss.

# Initialize coefficients
a, b, c = 0.0, 0.0, 0.0  # Start with random values
learning_rate = 0.00034  # Small learning rate
epochs = 9000  # Number of iterations

# Gradient Descent Loop
for epoch in range(epochs):
    # Compute predictions
    y_pred = quadratic_model(x, a, b, c)

    # Compute the loss
    loss = mean_squared_error(y, y_pred)

    # Compute gradients
    grad_a = (2 / len(x)) * np.sum((y_pred - y) * x**2)
    grad_b = (2 / len(x)) * np.sum((y_pred - y) * x)
    grad_c = (2 / len(x)) * np.sum(y_pred - y)

    # Update parameters
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c

    # Print loss every 100 iterations
    if epoch < 10 or epoch > epochs-10:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, a: {a:.4f}, b: {b:.4f}, c: {c:.4f}")

# Final coefficients
print(f"\nFinal parameters: a: {a:.4f}, b: {b:.4f}, c: {c:.4f}")
print(f"Epochs:{epochs}, lr:{learning_rate}")


# Define the true function
def true_function(x, a=2, b=3, c=5):
    return a * x**2 + b * x + c

# Create x values and the corresponding y values for the true function
x_values = np.linspace(-10, 10, 100)
y_true = true_function(x_values)

# Generate predicted y values using your best-fit parameters
y_pred = 1.9980 * x_values**2 + 2.9708 * x_values + 5.1380

# Plot the results
plt.plot(x_values, y_true, label="True Function", color="orange", linestyle='--')
plt.scatter(x_values, y_pred, color="blue", label="Predicted Points")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('True Function vs Predicted Curve')
plt.grid(True)
plt.show()