import numpy as np
import matplotlib.pyplot as plt
X = np.array([1,2,3,4,5])#hours studied
Y = np.array([60,62,62,64,65])#exam score

w = 0
b = 0
learning_rate = 0.01
epochs = 10000
n = len(X)
mse_history = []

for epoch in range(epochs): 
    Ypred = w * X + b
    mse = np.mean((Y - Ypred) ** 2)
    mse_history.append(mse)

    dw  = (-2 / n) * np.sum(X * (Y - Ypred))
    db = ( -2 / n) * np.sum(Y - Ypred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    #optional, too look at what is happening every iterations
    print(
        f"Epoch : {epoch+1} | "
        f"MSE : {mse:.4f} | "
        f"w : {w:.4f} | "
        f"b : {b:.4f} | "
        f"Predicted Value : {Ypred} | "
    )

pred = w * 6 + b
ss_res = np.sum((Y - Ypred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r2 = 1 - (ss_res / ss_tot) 
print(f"R2 : {r2:.4f}")

# plt.plot(mse_history)
# plt.title("MSE drops over epochs")
# plt.xlabel("Epochs")
# plt.ylabel("LOSS (MSE)")

fig ,(ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

#plot for MSE 
ax1.plot(mse_history, color='magenta', label='MSE VALUE')
ax1.set_title("Error Minimization History")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.grid(True)

#plot for regression model
ax2.scatter(X , Y, color='blue', label='Data Coordinates')
X_range = np.linspace(min(X) -1, max(X)+2, 100)
Y_range = w * X_range +  b
ax2.plot(X_range, Y_range,color='green',label='Best fit line')
ax2.scatter([6], [w * 6 + b], color='red', marker='*',s =150, label='6hf Target')
ax2.set_title("Regression fir line")
ax2.set_xlabel("Hours")
ax2.set_ylabel("Score")
ax2.legend()
plt.tight_layout()
print(pred)
plt.show()
