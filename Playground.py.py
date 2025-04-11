import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Height': [150, 160, 165, 170, 175, 180, 185, 190],
    'Weight': [50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)

# Step 2: Define features (X) and target (y)
X = df[['Height']]  # Features (independent variables)
y = df['Weight']    # Target (dependent variable)

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Print model coefficients (slope and intercept)
print(f"Slope (m): {model.coef_[0]}")
print(f"Intercept (b): {model.intercept_}")

# Step 7: Make a prediction for a specific height
height = 172  # Example: predict weight for a height of 172 cm
predicted_weight = model.predict([[height]])
print(f"Predicted weight for height {height} cm: {predicted_weight[0]:.2f} kg")

# Step 8: Plot the data and the regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs. Weight - Linear Regression')
plt.legend()
plt.show()
