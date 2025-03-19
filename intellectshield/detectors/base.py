import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import os
import time
import datetime

class BaseAnomalyDetector:
    """
    Базовый класс для всех детекторов аномалий.
    Определяет общий интерфейс для различных типов моделей.
    """
    
    def __init__(self, model_dir="models"):
        """
        Инициализация базового детектора.
        """
        self.model = None
        self.scaler = None
        self.model_dir = model_dir
        self.features = None
        self.training_summary = {}
        
        # Создание директории для моделей, если не существует
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def preprocess_data(self, data, train=False):
        """
        Предобработка данных.
        """
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")
    
    def train(self, data, **kwargs):
        """
        Обучение модели.
        """
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")
    
    def predict(self, data):
        """
        Обнаружение аномалий в данных.
        """
        raise NotImplementedError("Метод должен быть переопределен в дочернем классе")
    
    def evaluate(self, data):
        """
        Оценка производительности модели.
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Для оценки необходимы данные с колонкой 'is_anomaly'")
        
        # Проверяем наличие колонки predicted_anomaly
        if 'predicted_anomaly' not in data.columns:
            # Если нет, выполняем предсказание
            result_df = self.predict(data)
        else:
            # Используем существующие предсказания
            result_df = data.copy()
        
        # Вычисляем метрики
        cm = confusion_matrix(result_df['is_anomaly'], result_df['predicted_anomaly'])
        tn, fp, fn, tp = cm.ravel()
        
        # Расчет метрик
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Полный отчет
        report = classification_report(
            result_df['is_anomaly'],
            result_df['predicted_anomaly'],
            output_dict=True
        )
        
        # Результаты
        evaluation = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'detector_type': self.__class__.__name__
        }
        
        return evaluation
