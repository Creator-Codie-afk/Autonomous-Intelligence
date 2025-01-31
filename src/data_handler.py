import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocess data by normalizing the values."""
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def split_data(data, labels, test_size=0.2):
    """Split data into training and testing sets."""
    return train_test_split(data, labels, test_size=test_size, random_state=42)
pass
