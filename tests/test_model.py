"""
Unit tests for the heart disease prediction model.
"""
import unittest
import numpy as np
from src.model import create_model, train_model, predict
from src.data_preprocessing import preprocess_data

class TestModel(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create simple test data
        self.X_train = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        self.y_train = np.array([1, 0, 1])
        self.X_test = np.array([[1, 0, 0]])
        
    def test_model_creation(self):
        """Test model creation."""
        model = create_model()
        self.assertIsNotNone(model)
        
    def test_model_training(self):
        """Test model training."""
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        
    def test_prediction(self):
        """Test model prediction."""
        model = train_model(self.X_train, self.y_train)
        predictions = predict(model, self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        
if __name__ == '__main__':
    unittest.main()