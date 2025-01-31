import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
  """Load data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    return pd.read_csv(filepath)

def preprocess_data(data):
    """Preprocess data by normalizing the values."""

    Parameters:
    data (pd.DataFrame): The data to preprocess.

    Returns:
    pd.DataFrame: The normalized data.
    """
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def split_data(data, labels, test_size=0.2):
   """Split data into training and testing sets.

    Parameters:
    data (pd.DataFrame): The data to split.
    labels (pd.Series): The corresponding labels.
    test_size (float): The proportion of the dataset to include in the test split.

    Returns:
    tuple: Returns (X_train, X_test, y_train, y_test).
    """""
    return train_test_split(data, labels, test_size=test_size, random_state=42)
pass
