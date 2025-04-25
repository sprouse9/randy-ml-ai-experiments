# %%
'''
Now we move on from One-shot learning to a Mini dataset.
We ask the model to find a single pair of weight and bias values that
work well across all examples, not just one.
'''

import numpy as np
import matplotlib.pyplot as plt

# 1. Training data
x_data = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
y_data = np.array([0.2, 0.4, 0.5, 0.8, 1.0])
# %%
# 2. Initialize parameters
w = np.random.rand()
b = np.random.rand()
lr = 0.1
epochs = 200
# %%
# 4. Training loop
for epoch in range(epochs):
    y_pred = w * x_data + b
    error = y_pred - y_data
    loss = np.mean(error ** 2)

    # Gradients
    dL_dw = 2 * np.mean(error * x_data)
    dL_db = 2 * np.mean(error)

    # Update parameters
    w -= lr * dL_dw
    b -= lr * dL_db
# %%
# 5. Final model predictions
final_preds = w * x_data + b
final_errors = final_preds - y_data
final_loss = np.mean(final_errors ** 2)
# %%
# 6. Print summary
print("\n=== Training Summary ===")
print(f"Final weight (w): {w:.5f}")
print(f"Final bias   (b): {b:.5f}")
print(f"Final MSE loss: {final_loss:.6f}")
print("\nPredictions vs Targets:")
for x, pred, target in zip(x_data, final_preds, y_data):
    print(f"x={x:.1f} → Predicted: {pred:.4f}, Target: {target:.4f}, Error: {pred - target:.4f}")
