import joblib
import pandas as pd

# Load the trained model
model = joblib.load('cnf_status_model.pkl')

# Example input: [Train No, WL Position, Quota, Class, Days to Journey]
new_data = pd.DataFrame({
    'Train No': [54321],
    'WL Position': [100],
    'Quota': [2],  # Replace with encoded value (General=0, Tatkal=1, etc.)
    'Class': [3],  # Replace with encoded value (Sleeper=0, AC 3-Tier=1, etc.)
    'Days to Journey': [10]
})

# Make a prediction
prediction = model.predict(new_data)

# Output the prediction
if prediction == 1:
    print("Prediction: Confirmed")
else:
    print("Prediction: Waitlist")
