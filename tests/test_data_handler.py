import unittest
import pandas as pd
from src.data_handler import load_data, preprocess_data, split_data

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8]
        })
        self.labels = pd.Series([0, 1, 0, 1])

    def test_load_data(self):
        # Test if load_data function works correctly
        df = load_data('path/to/sample.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_preprocess_data(self):
        # Test if preprocess_data function normalizes data correctly
        normalized_data = preprocess_data(self.data)
        self.assertEqual(normalized_data.shape, self.data.shape)

    def test_split_data(self):
        # Test if split_data function splits data correctly
        X_train, X_test, y_train, y_test = split_data(self.data, self.labels)
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_train), 3)
        self.assertEqual(len(y_test), 1)

if __name__ == '__main__':
    unittest.main()
