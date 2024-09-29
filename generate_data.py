import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Define parameters
num_records = 1000
train_numbers = [12345, 54321, 67890, 98765, 11223]
quotas = ['General', 'Tatkal', 'Ladies', 'Senior Citizen']
classes = ['Sleeper', 'AC 3-Tier', 'AC 2-Tier', 'General']

# Generate random data
booking_ids = np.arange(1, num_records + 1)
train_nos = np.random.choice(train_numbers, num_records)
booking_dates = [datetime.today() - timedelta(days=np.random.randint(1, 365)) for _ in range(num_records)]
wl_positions = np.random.randint(0, 100, num_records)
quotas_selected = np.random.choice(quotas, num_records)
classes_selected = np.random.choice(classes, num_records)
days_to_journey = np.random.randint(1, 180, num_records)

# Simulate Status based on WL Position and Quota
status = []
for wl, quota in zip(wl_positions, quotas_selected):
    if quota == 'Tatkal':
        prob = 0.7 if wl < 10 else 0.3
    elif quota == 'General':
        prob = 0.6 if wl < 20 else 0.2
    elif quota == 'Ladies' or quota == 'Senior Citizen':
        prob = 0.8 if wl < 15 else 0.4
    else:
        prob = 0.5
    status.append(np.random.choice(['Confirmed', 'Waitlist'], p=[prob, 1 - prob]))

# Create DataFrame
data = pd.DataFrame({
    'Booking ID': booking_ids,
    'Train No': train_nos,
    'Booking Date': booking_dates,
    'WL Position': wl_positions,
    'Quota': quotas_selected,
    'Class': classes_selected,
    'Days to Journey': days_to_journey,
    'Status': status
})

# Save to CSV
data.to_csv('train_data.csv', index=False)
print("Dataset created successfully!")
