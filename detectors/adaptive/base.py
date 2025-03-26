from intellectshield.detectors.base import BaseAnomalyDetector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

# Импортируем компоненты нашего улучшенного детектора
from intellectshield.detectors.adaptive.features import FeatureExtractor
from intellectshield.detectors.adaptive.profiles import ProfileManager
from intellectshield.detectors.adaptive.detection import AnomalyDetection

class EnhancedAdaptiveDetector(BaseAnomalyDetector):
    """
    Улучшенный детектор с адаптивными параметрами в зависимости от текущих характеристик трафика.
    
    Ключевые улучшения:
    1. Многоуровневый профиль нормального поведения
    2. Динамическая адаптация порогов
    3. Обнаружение контекстуальных и коллективных аномалий
    4. Анализ временных паттернов и сезонности
    5. Интеграция машинного обучения и статистических методов
    """
    
    def __init__(self, model_dir="models"):
        """
        Инициализация улучшенного адаптивного детектора.
        
        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        """
        super().__init__(model_dir)
        
        # Компоненты детектора
        self.feature_extractor = FeatureExtractor()
        self.profile_manager = ProfileManager()
        self.anomaly_detection = AnomalyDetection()
        
        # Параметры адаптивных порогов
        self.threshold_multipliers = {
            'low': 2.0,     # Низкий порог (высокая чувствительность)
            'medium': 3.0,  # Средний порог
            'high': 4.0     # Высокий порог (низкая чувствительность)
        }
        self.current_sensitivity = 'medium'  # Текущий уровень чувствительности
        
        # Флаг, указывающий, был ли детектор обучен
        self.is_trained = False
    
    def preprocess_data(self, data, train=False):
        """
        Комплексная предобработка данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для предобработки
        train : bool
            Флаг режима обучения (True) или предсказания (False)
            
        Returns:
        --------
        pandas.DataFrame
            Предобработанные данные
        """
        return self.feature_extractor.process(data, train)
    
    def train(self, data, **kwargs):
        """
        Обучение улучшенного адаптивного детектора.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения детектора
        **kwargs : dict
            Дополнительные параметры
            
        Returns:
        --------
        self
            Обученный детектор
        """
        print("Начало обучения улучшенного адаптивного детектора...")
        start_time = time.time()
        
        # Разделяем данные на нормальные и аномальные
        if 'is_anomaly' in data.columns:
            normal_data = data[data['is_anomaly'] == 0]
            anomaly_data = data[data['is_anomaly'] == 1]
            print(f"Данные содержат {len(normal_data)} нормальных и {len(anomaly_data)} аномальных образцов")
        else:
            normal_data = data  # Предполагаем, что все данные нормальные, если нет меток
            anomaly_data = pd.DataFrame()
            print(f"Данные не содержат меток аномалий. Предполагаем, что все {len(data)} образцов нормальные.")
        
        # Предобработка данных
        print("Предобработка данных...")
        preprocessed_data = self.preprocess_data(normal_data, train=True)
        
        # Обновление профилей
        print("Обновление профилей...")
        self.profile_manager.update_profiles(preprocessed_data, self.feature_extractor.feature_groups)
        
        # Инициализация модуля обнаружения аномалий
        print("Инициализация модуля обнаружения аномалий...")
        self.anomaly_detection.initialize(
            preprocessed_data, 
            self.profile_manager.profiles,
            self.feature_extractor.feature_groups,
            self.threshold_multipliers
        )
        
        # Обучение ML-моделей
        print("Обучение ML-моделей...")
        self.anomaly_detection.train_ml_models(preprocessed_data, self.feature_extractor.feature_groups)
        
        # Отмечаем, что детектор обучен
        self.is_trained = True
        
        # Завершение обучения
        training_time = time.time() - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return self
    
    def predict(self, data):
        """
        Обнаружение аномалий в данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Результаты анализа с оценками аномальности и предсказаниями
        """
        if not self.is_trained:
            raise ValueError("Детектор не обучен. Сначала вызовите метод train().")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=False)
        
        # Обнаружение аномалий
        return self.anomaly_detection.detect_anomalies(
            preprocessed_data, 
            data,
            self.current_sensitivity
        )
        
    def visualize_anomalies(self, result_df, max_samples=1000):
        """
        Визуализация обнаруженных аномалий.
        
        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты обнаружения аномалий (выход метода predict)
        max_samples : int
            Максимальное количество образцов для визуализации
        """
        # Проверка наличия аномалий
        if 'predicted_anomaly' not in result_df.columns or 'anomaly_score' not in result_df.columns:
            print("Ошибка: входные данные не содержат результатов обнаружения аномалий")
            return
        
        anomaly_count = result_df['predicted_anomaly'].sum()
        if anomaly_count == 0:
            print("Аномалии не обнаружены. Визуализация невозможна.")
            return
        
        # Ограничиваем количество образцов для визуализации
        if len(result_df) > max_samples:
            # Стратифицированная выборка для сохранения пропорции аномалий
            normal_samples = min(int(max_samples * 0.7), (result_df['predicted_anomaly'] == 0).sum())
            anomaly_samples = min(int(max_samples * 0.3), anomaly_count)
            
            normal_df = result_df[result_df['predicted_anomaly'] == 0].sample(normal_samples, random_state=42)
            anomaly_df = result_df[result_df['predicted_anomaly'] == 1].sample(anomaly_samples, random_state=42)
            
            vis_df = pd.concat([normal_df, anomaly_df])
        else:
            vis_df = result_df.copy()
        
        # Создаем фигуру для визуализации
        plt.figure(figsize=(16, 12))
        
        # 1. Распределение аномальных оценок
        plt.subplot(2, 2, 1)
        plt.hist([
            vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
            vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score']
        ], bins=50, label=['Нормальные', 'Аномальные'])
        plt.xlabel('Оценка аномальности')
        plt.ylabel('Количество')
        plt.title('Распределение оценок аномальности')
        plt.legend()
        plt.grid(True)
        
        # 2. Типы аномалий (если есть)
        plt.subplot(2, 2, 2)
        if 'anomaly_type' in vis_df.columns:
            anomaly_types = vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
            anomaly_types.plot(kind='bar', color='coral')
            plt.title('Типы обнаруженных аномалий')
            plt.xlabel('Тип аномалии')
            plt.ylabel('Количество')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
        else:
            plt.text(0.5, 0.5, 'Информация о типах аномалий отсутствует',
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # 3. Временной ряд аномальных оценок (если есть временная метка)
        plt.subplot(2, 2, 3)
        if 'timestamp' in vis_df.columns:
            plt.scatter(
                vis_df[vis_df['predicted_anomaly'] == 0]['timestamp'], 
                vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
                alpha=0.5, label='Нормальные', s=20, color='blue'
            )
            plt.scatter(
                vis_df[vis_df['predicted_anomaly'] == 1]['timestamp'], 
                vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score'],
                alpha=0.7, label='Аномальные', s=30, color='red'
            )
            plt.title('Временной ряд аномальных оценок')
            plt.xlabel('Время')
            plt.ylabel('Оценка аномальности')
            plt.legend()
            plt.grid(True)
        else:
            # Если нет временной метки, визуализируем по индексу
            plt.scatter(
                range(len(vis_df[vis_df['predicted_anomaly'] == 0])),
                vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
                alpha=0.5, label='Нормальные', s=20, color='blue'
            )
            plt.scatter(
                range(len(vis_df[vis_df['predicted_anomaly'] == 0]), len(vis_df)),
                vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score'],
                alpha=0.7, label='Аномальные', s=30, color='red'
            )
            plt.title('Оценки аномальности по индексу')
            plt.xlabel('Индекс')
            plt.ylabel('Оценка аномальности')
            plt.legend()
            plt.grid(True)
        
        # 4. Матрица ошибок (если есть истинные метки)
        plt.subplot(2, 2, 4)
        if 'is_anomaly' in vis_df.columns:
            from sklearn.metrics import confusion_matrix
            
            cm = confusion_matrix(vis_df['is_anomaly'], vis_df['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Нормальные', 'Аномальные'],
                        yticklabels=['Нормальные', 'Аномальные'])
            plt.title('Матрица ошибок')
            plt.xlabel('Предсказание')
            plt.ylabel('Истинное значение')
            
            # Рассчитываем метрики качества
            tp = cm[1, 1]
            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = (
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}"
            )
            
            plt.text(1.5, 0.5, metrics_text, fontsize=11, 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            plt.text(0.5, 0.5, 'Истинные метки аномалий отсутствуют',
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим сводную информацию
        print(f"Всего записей: {len(result_df)}")
        print(f"Обнаруженных аномалий: {anomaly_count} ({anomaly_count/len(result_df)*100:.2f}%)")
        
        if 'anomaly_type' in result_df.columns:
            print("\nРаспределение типов аномалий:")
            type_counts = result_df[result_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
            for anomaly_type, count in type_counts.items():
                print(f"  {anomaly_type}: {count} ({count/anomaly_count*100:.2f}%)")
        
        if 'is_anomaly' in result_df.columns:
            # Вычисляем метрики
            from sklearn.metrics import classification_report
            print("\nМетрики качества обнаружения:")
            print(classification_report(result_df['is_anomaly'], result_df['predicted_anomaly']))
