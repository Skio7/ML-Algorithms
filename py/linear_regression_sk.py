import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generating random data
np.random.seed(0)
x = np.random.randint(0, 100, 50).reshape(-1, 1)
y = 2 * x.ravel() + np.random.normal(0, 10, 50)

# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(x, y)

# Getting the slope (m) and intercept (b)
m = model.coef_[0]
b = model.intercept_

print("Slope (m):", m)
print("Y-intercept (b):", b)

# Predicting for a new value of x
new_x = np.array([[6]])
y_pred = model.predict(new_x)
print("Predicted y for x =", new_x, ":", y_pred)

# Plotting
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, model.predict(x), color='red', label='Linear regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with scikit-learn')
plt.legend()
plt.grid(True)
plt.show()
