import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv('your_dataset.csv')

# Preprocess data (example)
X = data.drop(columns=['Class'])
y = data['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
with open('models/logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
