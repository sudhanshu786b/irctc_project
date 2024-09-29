import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('train_data.csv')

# Handle missing values if needed (assuming there are no missing values in this dataset)

# Convert categorical columns (Quota, Class, Status) into numerical values using LabelEncoder
le_quota = LabelEncoder()
le_class = LabelEncoder()
le_status = LabelEncoder()

data['Quota'] = le_quota.fit_transform(data['Quota'])
data['Class'] = le_class.fit_transform(data['Class'])
data['Status'] = le_status.fit_transform(data['Status'])  # 'Confirmed' -> 1, 'Waitlist' -> 0

# Separate features (X) and target (y)
X = data[['Train No', 'WL Position', 'Quota', 'Class', 'Days to Journey']]
y = data['Status']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data to CSV
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Data preprocessing complete!")
