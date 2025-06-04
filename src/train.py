import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from src.data_preprocessing import load_data, preprocess_data

def train_model():
    # Load and preprocess the data
    data = load_data('data/raw/data.csv')
    X, y = preprocess_data(data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, 'models/random_forest_model.pkl')

if __name__ == "__main__":
    train_model()