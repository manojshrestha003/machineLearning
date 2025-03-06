import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate Sample Data (X: Independent Variable, y: Dependent Variable)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # Random values between 0 and 10
y = 2.5 * X + np.random.randn(100, 1) * 2  # y = 2.5X + noise

# Convert to Pandas DataFrame (Optional)
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Split Data into Training and Testing Sets (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)
print(y_pred)
# Print Model Coefficients
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficient: {model.coef_[0][0]}")

# Visualizing the Regression Line
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("X (Independent Variable)")
plt.ylabel("y (Dependent Variable)")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
