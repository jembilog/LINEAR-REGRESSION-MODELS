import numpy as np
X = np.array([1,2,3,4,5,6,7])
Y = np.array([10,11,12,13,14,15,16])

w = 0
b = 0

lr = 0.01
for epoch in range(1000):
    Y_pred = w * X + b
    mse = np.mean((Y - Y_pred) ** 2)
    dw = (-2 / len(X)) * np.sum(X * (Y - Y_pred))
    db = (-2 / len(X)) * np.sum(Y - Y_pred)

    w = w - lr * dw
    b = b - lr * db

    print(
        f"Epoch : {epoch+1} | "
        f"MSE : {mse:.4f} | "
        f"w : {w:.4f} | "
        f"b : {b:.4f} | "
        f"Predicted Value : {Y_pred} | "
    )

prediction = w * 23 + b
print(prediction)
