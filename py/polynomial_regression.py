import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for polynomial regression
np.random.seed(0)
X = np.random.uniform(-3, 3, 100)
y = 2*X - 3*X**2 + np.random.normal(0, 3, 100)  # Polynomial with noise

# Perform polynomial regression without scikit-learn
degree = 2  # Degree of the polynomial
coefficients = np.polyfit(X, y, degree)
p = np.poly1d(coefficients)

# Sort the data points for plotting
sorted_indices = np.argsort(X)
X_sorted = X[sorted_indices]
y_sorted = y[sorted_indices]

# Plot the data and polynomial fit
plt.scatter(X_sorted, y_sorted, color='blue', label='Data points')
plt.plot(X_sorted, p(X_sorted), color='red', label='Polynomial Fit (Without scikit-learn)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
