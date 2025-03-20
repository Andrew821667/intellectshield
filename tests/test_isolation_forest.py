
import unittest
import numpy as np
from intellectshield.detectors.isolation_forest import IsolationForestDetector

class TestIsolationForestDetector(unittest.TestCase):
    def setUp(self):
        # Создаем синтетические данные с выбросами
        np.random.seed(42)
        self.X_normal = np.random.normal(0, 1, (100, 2))
        self.X_outliers = np.random.uniform(-5, 5, (10, 2))
        self.X = np.vstack([self.X_normal, self.X_outliers])
        
        # Создаем детектор
        self.detector = IsolationForestDetector(contamination=0.1)
        
    def test_initialization(self):
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.contamination, 0.1)
        
    def test_fit_predict(self):
        # Обучаем детектор
        self.detector.fit(self.X)
        
        # Получаем предсказания
        predictions = self.detector.predict(self.X)
        
        # Проверяем, что предсказания имеют правильный формат
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))
        
        # Проверяем, что количество обнаруженных аномалий примерно соответствует ожидаемому
        anomalies_count = sum(1 for pred in predictions if pred == -1)
        expected_count = int(len(self.X) * 0.1)
        self.assertLessEqual(abs(anomalies_count - expected_count), 5)
        
if __name__ == "__main__":
    unittest.main()
