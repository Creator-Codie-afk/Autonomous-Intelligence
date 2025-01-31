import unittest
from sklearn.datasets import make_classification
from src.model_trainer import train_model, save_model

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        self.data, self.labels = make_classification(n_samples=100, n_features=4, random_state=42)

    def test_train_model(self):
        # Test if train_model function trains a model correctly
        model = train_model(self.data, self.labels)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

    def test_save_model(self):
        # Test if save_model function saves a model correctly
        model = train_model(self.data, self.labels)
        save_model(model, 'test_model.joblib')
        # Check if the file exists
        import os
        self.assertTrue(os.path.exists('test_model.joblib'))
        # Clean up the test file
        os.remove('test_model.joblib')

if __name__ == '__main__':
    unittest.main()
