import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate some random data for multiple linear regression
np.random.seed(0)
X = np.random.randn(100, 3)  # Features (3 variables)
y = 2 * X[:, 0] + 3 * X[:, 1] - 5 * X[:, 2] + np.random.normal(0, 1, 100)  # Target variable

# Fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Make predictions
predictions = model.predict(X)

# Plot the actual vs. predicted values
plt.scatter(y, predictions)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted values')
plt.grid(True)
plt.show()
