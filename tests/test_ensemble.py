
import unittest
import numpy as np
from intellectshield.detectors.ensemble import EnsembleAnomalyDetector
from intellectshield.detectors.isolation_forest import IsolationForestDetector
from intellectshield.detectors.lof import LOFDetector

class TestEnsembleAnomalyDetector(unittest.TestCase):
    def setUp(self):
        # Создаем синтетические данные с выбросами
        np.random.seed(42)
        self.X_normal = np.random.normal(0, 1, (100, 2))
        self.X_outliers = np.random.uniform(-5, 5, (10, 2))
        self.X = np.vstack([self.X_normal, self.X_outliers])
        
        # Создаем базовые детекторы
        self.iso_forest = IsolationForestDetector(contamination=0.1)
        self.lof = LOFDetector(contamination=0.1)
        
        # Создаем ансамблевый детектор
        self.ensemble = EnsembleAnomalyDetector(
            detectors=[self.iso_forest, self.lof],
            voting='majority'
        )
        
    def test_fit_predict(self):
        # Обучаем детектор
        self.ensemble.fit(self.X)
        
        # Получаем предсказания
        predictions = self.ensemble.predict(self.X)
        
        # Проверяем, что предсказания имеют правильный формат
        self.assertEqual(len(predictions), len(self.X))
        self.assertTrue(all(pred in [-1, 1] for pred in predictions))
        
if __name__ == "__main__":
    unittest.main()
