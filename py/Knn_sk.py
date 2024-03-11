import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Generate some random data for classification
np.random.seed(0)
X_train = np.array([[1, 2], [2, 3], [3, 2], [2, 1], [5, 6], [6, 5], [5, 5], [6, 6]])  # Training features
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # Training labels (binary)

X_test = np.random.randn(20, 2)  # Test features

# Fit the k-Nearest Neighbors classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Perform predictions on test data
predictions = knn.predict(X_test)

# Plotting the training and test data with predicted labels
plt.figure(figsize=(8, 6))

# Plot training data
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='blue', marker='o', label='Class 0 (Training)')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='red', marker='o', label='Class 1 (Training)')

# Plot test data with predicted labels
for i, prediction in enumerate(predictions):
    color = 'blue' if prediction == 0 else 'red'
    plt.scatter(X_test[i, 0], X_test[i, 1], c=color, marker='x', label=f'Class {prediction} (Test)' if i == 0 else None)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-Nearest Neighbors Classification (Using scikit-learn)')
plt.legend()
plt.show()

print("Predictions for test data:", predictions)
