import matplotlib.pyplot as plt
import numpy as np

def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum([xi * yi for xi, yi in zip(x, y)])
    sum_x_sqr = sum([xi ** 2 for xi in x])

    # Calculate slope (m) and y-intercept (b)
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_sqr - sum_x ** 2)
    b = (sum_y - m * sum_x) / n

    return m, b

def predict(m, b, x):
    y = m * x + b
    return y

np.random.seed(0)
x = np.random.randint(0, 100, 50)
y = 2 * x + np.random.normal(0, 10, 50)

m, b = linear_regression(x, y)
print("Slope (m):", m)
print("Y-intercept (b):", b)


# Predict for a new value of x
new_x = 6
y_pred = predict(m, b, new_x)
print("Predicted y for x =", new_x, ":", y_pred)

# Plotting
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, [predict(m, b, i) for i in x], color='red', label='Linear regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.grid(True)
plt.legend()
plt.show()
