import unittest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.evaluator import evaluate_model, report_performance

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data for testing
        data, labels = make_classification(n_samples=100, n_features=4, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_evaluate_model(self):
        # Test if evaluate_model function evaluates the model correctly
        accuracy, report = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertIsNotNone(accuracy)
        self.assertIsNotNone(report)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(report, str)

    def test_report_performance(self):
        # Test if report_performance function prints the performance metrics
        accuracy, report = evaluate_model(self.model, self.X_test, self.y_test)
        report_performance(accuracy, report)
        # This test simply ensures the function runs without error

if __name__ == '__main__':
    unittest.main()
