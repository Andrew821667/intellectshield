"""
Основной класс расширенного адаптивного детектора аномалий.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple

from intellectshield.detectors.base import BaseAnomalyDetector
from .anomaly_score_calculator import AnomalyScoreCalculator
from .threshold_manager import ThresholdManager
from .profile_manager import ProfileManager
from .ml_models_manager import MLModelsManager
from .anomaly_type_determiner import AnomalyTypeDeterminer


class EnhancedAdaptiveDetector(BaseAnomalyDetector):
    """
    Расширенный адаптивный детектор аномалий.
    
    Этот детектор объединяет несколько подходов к обнаружению аномалий:
    1. Статистический анализ на основе профилей
    2. Контекстуальный анализ (временной и другие контексты)
    3. Машинное обучение (Isolation Forest, LOF, DBSCAN)
    4. Анализ коллективных аномалий (аномалии в последовательностях)
    """
    
    def __init__(self, model_dir: str = "models", **kwargs):
        """
        Инициализация расширенного адаптивного детектора.
        
        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        **kwargs : dict
            Дополнительные параметры
        """
        super().__init__(model_dir=model_dir)
        
        # Инициализируем компоненты
        self.profile_manager = ProfileManager()
        self.threshold_manager = ThresholdManager()
        self.ml_models_manager = MLModelsManager()
        self.anomaly_score_calculator = AnomalyScoreCalculator(
            profile_manager=self.profile_manager,
            threshold_manager=self.threshold_manager,
            ml_models_manager=self.ml_models_manager
        )
        self.anomaly_type_determiner = AnomalyTypeDeterminer(
            profile_manager=self.profile_manager
        )
        
        # Параметры детектора
        self.feature_groups = None
        self.threshold_multipliers = None
        self.default_sensitivity = kwargs.get('default_sensitivity', 'medium')
        
        # Сохраняем информацию о последнем обучении
        self.last_training_data_shape = None
        self.is_initialized = False
    
    def initialize(self, data: pd.DataFrame, profiles: Dict, feature_groups: Dict, 
                 threshold_multipliers: Dict) -> None:
        """
        Инициализирует детектор с профилями и параметрами.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для инициализации
        profiles : dict
            Профили нормального поведения
        feature_groups : dict
            Словарь с группами признаков
        threshold_multipliers : dict
            Множители порогов для разных уровней чувствительности
        """
        # Сохраняем параметры
        self.feature_groups = feature_groups
        self.threshold_multipliers = threshold_multipliers
        
        # Инициализируем компоненты
        self.profile_manager.set_profiles(profiles)
        self.threshold_manager.calculate_thresholds(
            data=data,
            profiles=profiles,
            feature_groups=feature_groups
        )
        
        self.is_initialized = True
    
    def preprocess_data(self, data: pd.DataFrame, train: bool = False) -> pd.DataFrame:
        """
        Предобработка данных для детектора.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Исходные данные
        train : bool
            Флаг режима обучения
            
        Returns:
        --------
        pandas.DataFrame
            Предобработанные данные
        """
        # Базовая предобработка
        preprocessed_data = data.copy()
        
        # Заполняем пропуски в числовых признаках
        if self.feature_groups and 'numeric' in self.feature_groups:
            numeric_features = [f for f in self.feature_groups['numeric'] if f in preprocessed_data.columns]
            for feature in numeric_features:
                preprocessed_data[feature] = preprocessed_data[feature].fillna(0)
        
        return preprocessed_data
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict:
        """
        Обучение детектора.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
        **kwargs : dict
            Дополнительные параметры
            
        Returns:
        --------
        dict
            Результаты обучения
        """
        if not self.is_initialized:
            raise ValueError("Детектор не инициализирован. Сначала вызовите метод initialize()")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=True)
        
        # Обучаем ML модели
        self.ml_models_manager.train_models(
            data=preprocessed_data,
            feature_groups=self.feature_groups
        )
        
        # Сохраняем информацию об обучении
        self.last_training_data_shape = data.shape
        self.training_summary = {
            'data_shape': data.shape,
            'feature_count': len(preprocessed_data.columns),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return self.training_summary
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Результаты обнаружения аномалий
        """
        if not self.is_initialized:
            raise ValueError("Детектор не инициализирован. Сначала вызовите метод initialize()")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data)
        
        # Создаем копию для результатов
        result_df = data.copy()
        
        # Определяем чувствительность
        sensitivity = self.default_sensitivity
        threshold_multiplier = self.threshold_multipliers.get(sensitivity, 3.0)
        
        # Вычисляем оценки аномальности для различных типов аномалий
        scores = self.anomaly_score_calculator.calculate_scores(
            data=preprocessed_data,
            threshold_multiplier=threshold_multiplier
        )
        
        # Комбинируем оценки с весами
        combined_scores = self.anomaly_score_calculator.combine_scores(scores)
        
        # Нормализуем оценки
        normalized_scores = self._normalize_scores(combined_scores)
        
        # Добавляем оценку аномальности
        result_df['anomaly_score'] = normalized_scores
        
        # Определяем аномалии на основе порога
        anomaly_threshold = self._determine_anomaly_threshold(normalized_scores)
        result_df['predicted_anomaly'] = (normalized_scores >= anomaly_threshold).astype(int)
        
        # Определяем типы аномалий
        anomaly_indices = result_df[result_df['predicted_anomaly'] == 1].index
        if len(anomaly_indices) > 0:
            result_df = self.anomaly_type_determiner.determine_anomaly_types(
                result_df=result_df,
                data=preprocessed_data,
                scores=scores
            )
        
        return result_df
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Нормализует оценки аномальности.
        
        Parameters:
        -----------
        scores : numpy.ndarray
            Оценки аномальности
            
        Returns:
        --------
        numpy.ndarray
            Нормализованные оценки
        """
        if np.max(scores) > np.min(scores):
            return (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            return np.zeros_like(scores)
    
    def _determine_anomaly_threshold(self, normalized_scores: np.ndarray) -> float:
        """
        Определяет порог для выявления аномалий.
        
        Parameters:
        -----------
        normalized_scores : numpy.ndarray
            Нормализованные оценки аномальности
            
        Returns:
        --------
        float
            Порог для выявления аномалий
        """
        # Адаптивный порог: верхние 5% рассматриваются как аномалии
        percentile_threshold = np.percentile(normalized_scores, 95)
        
        # Абсолютный порог
        absolute_threshold = 0.7
        
        # Выбираем более консервативный порог
        return min(percentile_threshold, absolute_threshold)
