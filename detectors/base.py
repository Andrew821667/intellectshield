import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import joblib
import os
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

# Создание директории для моделей
os.makedirs("models", exist_ok=True)

# Базовый класс детектора аномалий
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
    
    def save_model(self, filename=None):
        """
        Сохранение модели детектора в файл.
        """
        if self.model is None:
            raise ValueError("Модель не обучена и не может быть сохранена.")
        
        # Если имя файла не указано, генерируем его из типа модели и времени
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_type = self.__class__.__name__
            filename = f"{model_type}_{timestamp}.joblib"
        
        # Полный путь к файлу
        filepath = os.path.join(self.model_dir, filename)
        
        # Создаем словарь с моделью и метаданными
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features,
            'training_summary': self.training_summary,
            'model_type': self.__class__.__name__,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Сохраняем в файл
        joblib.dump(model_data, filepath)
        print(f"Модель сохранена в {filepath}")
        
        return filepath
    
    def load_model(self, filepath):
        """
        Загрузка модели детектора из файла.
        """
        # Загружаем данные из файла
        model_data = joblib.load(filepath)
        
        # Проверяем, что тип модели совпадает
        if model_data['model_type'] != self.__class__.__name__:
            print(f"Предупреждение: загружаемая модель типа {model_data['model_type']}, "
                  f"но текущий детектор типа {self.__class__.__name__}")
        
        # Загружаем компоненты
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.features = model_data['features']
        self.training_summary = model_data['training_summary']
        
        print(f"Модель загружена из {filepath}")
        print(f"Тип модели: {model_data['model_type']}")
        print(f"Дата создания: {model_data['timestamp']}")
        
        return self
