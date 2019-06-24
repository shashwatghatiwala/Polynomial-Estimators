from random import random
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

x = []
y = []
with open('hw1data.txt', 'r') as data:
    for line in data:
        x.append(float(line.split()[0]))
        y.append(float(line.split()[1]))

x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
# Splitting the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, shuffle=True)

i = 0
mse_train = [0, 0, 0, 0]
mse_test = [0,0,0,0]

sum_train = np.zeros(38)
sum_test = np.zeros(13)

y_pred = np.zeros(len(y))
mse_main = [0,0,0,0]
variance_pred = [0,0,0,0]
bias_main = [0,0,0,0]

for deg in (1, 5, 10, 50):
    # Polynomial Estimators
    poly = PolynomialFeatures(degree= deg)
    x_train_poly = poly.fit_transform(x_train[: np.newaxis])
    x_test_poly = poly.fit_transform(x_test[: np.newaxis])
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    y_train_pred = model.predict(x_train_poly[: np.newaxis])
    y_test_pred = model.predict(x_test_poly[: np.newaxis])

    # Making one array for all the predicted values
    for j in range(len(y_train_pred)):
        sum_train[j] += y_train_pred[j]
    for j in range(len(y_test_pred)):
        sum_test[j] += y_test_pred[j]

    a = len(y_train_pred)

    # Computing Variance of Predictions
    for j in range(len(y_train_pred)):
        y_pred[j] = y_train_pred[j]
    for j in range(len(y_test_pred)):
        y_pred[(j + a)] = y_test_pred[j]

    # Computing Variance
    variance_pred[i] = np.var(y_pred)

    # Computing MSE of all Predictions (Training and Testing)
    mse_main[i] = mean_squared_error(y, y_pred)

    # Computing Bias of all Predictions (Training and Testing)
    m = np.mean(y_pred)
    for j in range(len(y)):
        bias_main[i] += math.pow((y[j] - m), 2)

    # Computing MSE for average of four estimators
    mse_train[i] = mean_squared_error(y_train, y_train_pred)
    mse_test[i] = mean_squared_error(y_test, y_test_pred)
    i = i + 1

    # Plotting graph for each estimator
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x_train,y_train_pred), key=sort_axis)
    x_train, y_train_pred = zip(*sorted_zip)
    plt.scatter(x_train, y_train, s = 10, color = 'k')
    plt.plot(x_train, y_train_pred)
    plt.gca().legend(('Degree 1', 'Degree 5', 'Degree 10', 'Degree 50'))
plt.xlabel('Degree of Polynomial')
plt.ylabel('Prediction Curve')
plt.show()

degree_ = [1, 5, 10, 50]
# print(mse_main)
# print("Bias for All Estimators is: ", bias_main)

# Computing Average of Four Estimators
avg_train = np.zeros(38)
avg_test = np.zeros(13)
for j in range(len(sum_train)):
    avg_train[j] = (sum_train[j] / 4)
for j in range(len(sum_test)):
    avg_test[j] = (sum_test[j] / 4)

mse_avg_train = mean_squared_error(y_train, avg_train)
mse_avg_test = mean_squared_error(y_test, avg_test)

print("MSE of training data of each estimator: ", mse_train)
print("MSE of average on training data: ", mse_avg_train)

print("MSE of testing data of each estimator: ", mse_test)
print("MSE of average on testing data: ", mse_avg_test)

# Plotting Traning and Testing MSE
plt.plot(degree_, mse_train, label = "Training MSE Error")
plt.plot(degree_, mse_test, label = 'Testing MSE Error')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Mean Squared Error of Training and Testing Set')
plt.legend()
plt.show()

# Plotting average of all estimators on training data
plt.scatter(x_train, y_train, s = 10, color = 'k')
plt.plot(x_train, avg_train)
plt.show()

# Plotting Total MSE, Squared Bias, and Variance
plt.plot(degree_, mse_main, label = "Total MSE Error")
plt.plot(degree_, bias_main, label = 'Total Bias')
plt.plot(degree_, variance_pred, label = 'Total Variance')
plt.xlabel('Degree of Polynomial')
plt.ylabel('MSE, Squared Bias and Variance')
plt.legend()
plt.show()



