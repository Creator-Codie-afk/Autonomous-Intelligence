from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_model(data, labels):
   "

    Parameters:
    data (pd.DataFrame): The training data.
    labels (pd.Series): The training labels.

    Returns:
    RandomForestClassifier: The trained model.
    """"""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data, labels)
    return model

def save_model(model, filepath):
   

    Parameters:
    model: The trained machine learning model.
    filepath (str): The path to the file where the model will be saved.
    """
    dump(model, filepath)
