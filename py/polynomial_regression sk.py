import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate some random data for polynomial regression
np.random.seed(0)
X = np.random.uniform(-3, 3, 100)
y = 2*X - 3*X**2 + np.random.normal(0, 3, 100)  # Polynomial with noise

# Perform polynomial regression with scikit-learn
degree = 2  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X.reshape(-1, 1))
model = LinearRegression()
model.fit(X_poly, y)

# Sort the data points for plotting
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)  # Generate evenly spaced X values for plotting
X_plot_poly = poly_features.transform(X_plot)
y_pred = model.predict(X_plot_poly)
sorted_indices = np.argsort(X_plot.flatten())
X_plot_sorted = X_plot.flatten()[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Plot the data and polynomial fit
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X_plot_sorted, y_pred_sorted, color='red', label='Polynomial Fit (With scikit-learn)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Polynomial Regression')
plt.legend()
plt.grid(True)
plt.show()
