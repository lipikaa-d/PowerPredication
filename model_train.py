import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_excel('T1.xlsx')

# Select relevant columns
df = df[['Wind Speed (m/s)', 'Wind Direction (°)', 'LV ActivePower (kW)', 'Theoretical_Power_Curve (KWh)']]
df.dropna(inplace=True)

# Features and targets
X = df[['Wind Speed (m/s)', 'Wind Direction (°)']]
y = df[['LV ActivePower (kW)', 'Theoretical_Power_Curve (KWh)']]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Calculate new error: absolute difference between predicted Theoretical and LV values
errors = abs(y_pred[:, 1] - y_pred[:, 0])

# Average of these absolute differences
avg_error = errors.mean()

print(f"📊 Average Absolute Error (Theoretical - LV): {avg_error:.2f} kW")

# Save model
joblib.dump(model, 'model.pkl')
print("✅ Model saved as model.pkl")
