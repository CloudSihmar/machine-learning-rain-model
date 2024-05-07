import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the CSV file as a DataFrame
df = pd.read_csv('/root/machine-learning-rain-model/weather.csv')

# Handle missing values
df = df.fillna(df.mode().iloc[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'])

# Check if 'RainTomorrow' is in the DataFrame before dropping it
if 'RainTomorrow' in df.columns:
    # Split the data into features (X) and target (y)
    X = df.drop(['Date', 'Location', 'RainTomorrow'], axis=1)
    y = df['RainTomorrow']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]  # Probability of rain

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("\nConfusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_rep)

    # Predict the probability of rain for the next day
    new_data = X.tail(1)  # Assuming the last row in the dataset is for the next day
    rain_probability = model.predict_proba(new_data)[:,1][0]
    print("\nProbability of rain for the next day:", rain_probability)
else:
    print("'RainTomorrow' column not found in DataFrame.")
