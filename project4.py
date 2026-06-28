import numpy as np
import matplotlib.pyplot as plt
screen_time = np.array([
    3.6, 7.7, 6.1, 5.2, 2.1, 2.1, 1.4, 7.1, 5.2, 6.0,
    1.1, 7.8, 6.8, 2.5, 2.3, 2.3, 3.1, 4.7, 4.0, 3.0
])

battery_used = np.array([
    43.0, 94.0, 72.0, 61.0, 34.0, 28.0, 21.0, 82.0, 63.0, 74.0,
    14.0, 96.0, 81.0, 33.0, 30.0, 37.0, 41.0, 56.0, 53.0, 36.0
])

w = 0 
b = 0 
n = len(screen_time)
learning_rate = 0.005
epochs = 2000
mse_history = []

for epoch in range(epochs):
    Y_pred = w * screen_time + b
    mse = np.mean((battery_used - Y_pred) ** 2)
    mse_history.append(mse)

    dw = (-2/n) * np.sum(screen_time * (battery_used - Y_pred))
    db = (-2/n) * np.sum(battery_used - Y_pred)

    w -= learning_rate * dw
    b -= learning_rate * db

    if(epoch  +1) % 10 == 0:
        print(
            f"Epoch: {epoch + 1} | "
            f"MSE: {mse:.4f} | "
            f"w: {w:.4f} | "
            f"b: {b:.4f} | "
            f"Predicted value: {Y_pred}"
        )
Ypred = w * 3.5 + b
ss_res = np.sum((battery_used - Y_pred) ** 2)
ss_total = np.sum((battery_used - np.mean(battery_used)) ** 2)
r2 = 1 - (ss_res / ss_total)

plt.plot(mse_history)
plt.title("Error minimization")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
print(Ypred)
print(r2 * 100)
plt.show()