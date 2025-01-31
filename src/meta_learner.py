from sklearn.base import BaseEstimator, ClassifierMixin

class MetaLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model):
        self.base_model = base_model

    def fit(self, X, y):
        """Train the meta-learning model."""
        self.base_model.fit(X, y)
        return self

    def predict(self, X):
        """Predict using the meta-learning model."""
        return self.base_model.predict(X)

    def adapt(self, new_data, new_labels):
        """Adapt the meta-learning model to new data."""
        self.base_model.fit(new_data, new_labels)
