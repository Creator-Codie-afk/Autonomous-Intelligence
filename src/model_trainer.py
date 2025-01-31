from sklearn.ensemble import RandomForestClassifier
from joblib import dump

def train_model(data, labels):
    """Train a machine learning model using RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(data, labels)
    return model

def save_model(model, filepath):
    """Save the trained model to a file using joblib."""
    dump(model, filepath)
