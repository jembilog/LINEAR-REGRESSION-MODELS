import numpy as np
import matplotlib.pyplot as plt
#car fuel consumption
#pounds in thousands ->  fuel used(miles per gallon)

X = np.array([3.5,3.69,3.44,3.43,4.34,4.42,2.37])
Y = np.array([18,15,18,16,15,14,24])

w = 0
b = 0
n = len(X)
epochs = 10000
learning_rate = 0.05
mse_his =[]

for epoch in range(epochs):
    Y_pred = w * X + b
    mse = np.mean((Y - Y_pred)**2)
    mse_his.append(mse)
    
    dw = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if(epoch + 1) % 10 == 0:
        print(
            f"Epoch: {epoch + 1} | "
            f"MSE: {mse:.4f} | "
            f"w: {w:.4f} | "
            f"b: {b:.4f} | "
            f"Predicted value: {Y_pred}"
        )

X_new = 0.2
Y_pred_new = w * X_new + b
ss_res = np.sum((Y - Y_pred) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r2 = 1 - (ss_res / ss_tot) 
print(f"R2 : {r2:.4f}")
print(Y_pred_new)
print(mse)
# plt.plot(mse_his)
# plt.title("Error minimization history")
# plt.grid(True) 

plt.scatter(X, Y)
start_point= min(min(X), X_new)
X_range = np.linspace(start_point,max(X) + 1, 100)
# X_range = np.linspace(min(X) - 1, max(X) + 1, 100)
Y_range = w * X_range + b
plt.plot(X_range,Y_range, color='green',label='Best fit line')
plt.scatter([X_new],[w * X_new + b], color='red',marker='*',label='Target')
plt.title("Car Fuel Consumption")
plt.xlabel("Pounds in 1000s")
plt.ylabel("Miles per gallon")
plt.legend()
plt.show()