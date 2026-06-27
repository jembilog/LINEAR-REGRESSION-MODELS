import numpy as np
import matplotlib.pyplot as plt

X = np.array([30,50,70,90,120])
Y = np.array([50000,70000,90000,120000,150000])

#i need to scale this cuz it is too large or you can just use smaller values cuz it affects mse when iterating large data
X_max = np.max(X)
Y_max = np.max(Y)

X_scaled = X / X_max
Y_scaled = Y / Y_max

w = 0
b = 0
n = len(X)
mse_history = []
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    Y_pred = w * X_scaled + b
    mse = np.mean((Y_scaled - Y_pred) ** 2)
    dw = (-2 / n) * np.sum(X_scaled * ( Y_scaled - Y_pred))
    db = (-2 / n) * np.sum(Y_scaled - Y_pred)
    w -= learning_rate * dw
    b -= learning_rate * db


    print(
        f"Epoch : {epoch+1} | "
        f"MSE : {mse:.4f} | "
        f"w : {w:.4f} | "
        f"b : {b:.4f} | "
        f"Predicted Value : {Y_scaled} | "
    )
    mse_history.append(mse)

print("\nTraining complete")

final_predictions_orig = (w * X_scaled + b) * Y_max
y_mean = np.mean(Y)

ss_res = np.sum((Y - final_predictions_orig) ** 2) 
ss_tot = np.sum((Y -y_mean)  ** 2)
r2 = 1 - (ss_res / ss_tot)

new_x = 150
new_x_scaled = new_x / X_max
prediction_scaled = w * new_x_scaled + b
final_prediction = prediction_scaled * Y_max

print(mse)
print(f"R2 : {r2:.4f}")
print(final_prediction)

#plotting

plt.figure(figsize=(8,6))
plt.scatter(X, Y, color='blue',s=80,zorder=5,label='actual data')
X_line = np.linspace(min(X), max(new_x, max(X)), 100)
X_line_scaled = X_line / X_max
Y_line = ((w * X_line_scaled) + b) * Y_max
plt.plot(X_line, Y_line, color='green',linewidth=2.5, label='Regression line')
plt.scatter(new_x, final_prediction,color='red',marker='*',s=200,zorder=6,label=f'Prediction for X={new_x} ({final_prediction:.2f})')

plt.title("Linear Regressin Fit with Next Feature Prediction")
plt.xlabel("House size")
plt.ylabel("House price")
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
