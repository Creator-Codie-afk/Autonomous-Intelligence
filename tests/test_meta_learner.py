import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.meta_learner import MetaLearner

class TestMetaLearner(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        data, labels = make_classification(n_samples=100, n_features=4, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.base_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.meta_learner = MetaLearner(self.base_model)
        self.meta_learner.fit(self.X_train, self.y_train)

    def test_fit(self):
        # Test if MetaLearner's fit method works correctly
        self.assertIsInstance(self.meta_learner, MetaLearner)
        self.assertTrue(hasattr(self.meta_learner.base_model, 'predict'))

    def test_predict(self):
        # Test if MetaLearner's predict method works correctly
        predictions = self.meta_learner.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))

    def test_adapt(self):
        # Test if MetaLearner's adapt method works correctly
        new_data, new_labels = make_classification(n_samples=10, n_features=4, random_state=42)
        self.meta_learner.adapt(new_data, new_labels)
        # This test ensures the adapt method runs without error

if __name__ == '__main__':
    unittest.main()
