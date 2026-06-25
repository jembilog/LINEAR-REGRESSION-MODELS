import numpy as np

X = np.array([1,2,3,4,5])#features
Y = np.array([28,29,30,31,32])#output

w = 0.0
b = 0.0

alpha = 0.01
epochs = 1000
n = len(X)

#gradient descent
for epoch in range(epochs):
    Y_predict = w * X + b

    #partial derivative of weight
    dw = (-2 / n) * np.sum(X * (Y - Y_predict))
    #partial derivative of bias
    db = (-2 /n) * np.sum(Y - Y_predict)

    #update rule 
    w -= alpha * dw
    b -= alpha * db

Y_predict = w * X + b#need ulit ng variable ng ypred for mse,rmse,mae kasi local var lang yung kanina na na sa for loop

#checking loss and accuracy
mse = np.mean((Y - Y_predict)**2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(Y - Y_predict))

ss_res = np.sum((Y - Y_predict) ** 2)
ss_tot = np.sum((Y - np.mean(Y)) ** 2)
r2 = 1 - (ss_res / ss_tot)

nextfeature = 7
prediction = w * nextfeature + b
#updated bias and weight na yung nakalagay rito since nakuha na ni gradient descent yung best weight ans bias

print("MODEL PARAMETERS")
print(f"Weight/Slope (w): {w:.4f}")
print(f"Bias/Intercept (b): {b:.4f}")

print("\nTRAINING RESULTS")
for x, actual, pred in zip(X, Y, Y_predict):
    print(f"Iteration {int(x)} | Actual: {actual:.2f} | Predicted:{pred:.2f}")

print("\nEVALUATION")
print(f"MSE : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2 : {r2:.4f}")

print("\nPREDICTION")
print(f"Predicted number at index{nextfeature} : {prediction}")
