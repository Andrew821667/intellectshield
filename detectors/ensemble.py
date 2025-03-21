"""
Модуль ensemble.py - реализация ансамблевого детектора аномалий.

Этот модуль предоставляет класс EnsembleAnomalyDetector, который объединяет
результаты нескольких базовых детекторов для повышения точности обнаружения аномалий.
"""

import numpy as np
import pandas as pd
import os
import time
import datetime
import joblib
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Any, Tuple
from intellectshield.detectors.base import BaseAnomalyDetector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


class EnsembleStrategy(ABC):
    """
    Абстрактный класс для стратегий ансамблирования.
    
    Этот класс определяет интерфейс для различных стратегий ансамблирования,
    которые могут быть использованы в EnsembleAnomalyDetector.
    """
    
    @abstractmethod
    def combine(self, predictions: pd.DataFrame, weights: List[float]) -> Dict[str, np.ndarray]:
        """
        Объединение предсказаний от нескольких детекторов.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов
        weights : List[float]
            Веса детекторов
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь с итоговыми предсказаниями и оценками
        """
        pass


class MajorityVotingStrategy(EnsembleStrategy):
    """
    Стратегия голосования большинством (hard voting).
    """
    
    def combine(self, predictions: pd.DataFrame, weights: List[float]) -> Dict[str, np.ndarray]:
        """
        Применение метода голосования большинством.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов
        weights : List[float]
            Веса детекторов
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь с итоговыми предсказаниями и оценками
        """
        n_detectors = len(weights)
        
        # Получаем предсказания от всех детекторов
        detector_predictions = np.array([predictions[f'predicted_anomaly_{i}'].values
                                      for i in range(n_detectors)]).T
        
        # Веса детекторов для взвешенного голосования
        weights_array = np.array(weights)
        
        # Считаем взвешенную сумму голосов
        weighted_votes = np.sum(detector_predictions * weights_array, axis=1)
        
        # Определяем порог для положительного предсказания (больше половины от суммы весов)
        threshold = 0.5  # Можно настроить
        
        # Итоговые предсказания
        final_predictions = (weighted_votes > threshold).astype(int)
        
        # Оценки аномальности как взвешенное среднее оценок всех детекторов
        detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                  for i in range(n_detectors)]).T
        
        final_scores = np.sum(detector_scores * weights_array, axis=1)
        
        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }


class WeightedAverageStrategy(EnsembleStrategy):
    """
    Стратегия взвешенного среднего (soft voting).
    """
    
    def combine(self, predictions: pd.DataFrame, weights: List[float]) -> Dict[str, np.ndarray]:
        """
        Применение метода взвешенного среднего.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов
        weights : List[float]
            Веса детекторов
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь с итоговыми предсказаниями и оценками
        """
        n_detectors = len(weights)
        
        # Получаем оценки аномальности от всех детекторов
        detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                  for i in range(n_detectors)]).T
        
        # Нормализуем оценки перед взвешиванием
        normalized_scores = self._normalize_scores(detector_scores)
        
        # Веса детекторов
        weights_array = np.array(weights)
        
        # Взвешенная сумма нормализованных оценок
        final_scores = np.sum(normalized_scores * weights_array, axis=1)
        
        # Определяем аномалии на основе оценок
        # Используем адаптивный порог: верхние X% рассматриваются как аномалии
        anomaly_threshold = np.percentile(final_scores, 95)  # Верхние 5%
        final_predictions = (final_scores > anomaly_threshold).astype(int)
        
        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Нормализация оценок аномальности.
        
        Parameters:
        -----------
        scores : np.ndarray
            Матрица оценок аномальности
            
        Returns:
        --------
        np.ndarray
            Нормализованные оценки аномальности
        """
        normalized = np.zeros_like(scores)
        
        for i in range(scores.shape[1]):
            # Преобразуем оценки в диапазон [0, 1]
            min_score = np.min(scores[:, i])
            max_score = np.max(scores[:, i])
            
            if max_score > min_score:
                normalized[:, i] = (scores[:, i] - min_score) / (max_score - min_score)
            else:
                normalized[:, i] = 0
                
        return normalized


class RankFusionStrategy(EnsembleStrategy):
    """
    Стратегия слияния на основе ранжирования.
    """
    
    def combine(self, predictions: pd.DataFrame, weights: List[float]) -> Dict[str, np.ndarray]:
        """
        Применение метода слияния на основе ранжирования.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов
        weights : List[float]
            Веса детекторов
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь с итоговыми предсказаниями и оценками
        """
        n_samples = len(predictions)
        n_detectors = len(weights)
        
        # Создаем матрицу рангов
        ranks = np.zeros((n_samples, n_detectors))
        
        for i in range(n_detectors):
            # Получаем оценки аномальности от текущего детектора
            scores = predictions[f'anomaly_score_{i}'].values
            
            # Ранжируем образцы (большие оценки -> меньшие ранги)
            # argsort возвращает индексы, которые бы отсортировали массив
            # argsort(argsort) дает ранги
            ranks[:, i] = n_samples - np.argsort(np.argsort(scores))
        
        # Веса детекторов
        weights_array = np.array(weights).reshape(1, -1)
        
        # Взвешенная сумма рангов
        weighted_ranks = np.sum(ranks * weights_array, axis=1)
        
        # Нормализуем ранги для получения оценок аномальности
        min_rank = np.min(weighted_ranks)
        max_rank = np.max(weighted_ranks)
        
        if max_rank > min_rank:
            final_scores = (weighted_ranks - min_rank) / (max_rank - min_rank)
        else:
            final_scores = np.zeros(n_samples)
        
        # Определяем аномалии на основе рангов
        # Используем адаптивный порог: верхние 5% рассматриваются как аномалии
        rank_threshold = np.percentile(weighted_ranks, 95)
        final_predictions = (weighted_ranks > rank_threshold).astype(int)
        
        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }


class StackingStrategy(EnsembleStrategy):
    """
    Стратегия стекинга (meta-learning).
    """
    
    def __init__(self, metamodel: Optional[Any] = None):
        """
        Инициализация стратегии стекинга.
        
        Parameters:
        -----------
        metamodel : Optional[Any]
            Мета-модель для стекинга
        """
        self.metamodel = metamodel
        self.fallback_strategy = WeightedAverageStrategy()
    
    def combine(self, predictions: pd.DataFrame, weights: List[float]) -> Dict[str, np.ndarray]:
        """
        Применение метода стекинга.
        
        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов
        weights : List[float]
            Веса детекторов
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Словарь с итоговыми предсказаниями и оценками
        """
        # Проверяем, была ли обучена метамодель
        if self.metamodel is None:
            print("Предупреждение: метамодель не обучена. Используется метод взвешенного среднего.")
            return self.fallback_strategy.combine(predictions, weights)
        
        # Подготавливаем данные для метамодели
        feature_cols = [f'anomaly_score_{i}' for i in range(len(weights))]
        X_meta = predictions[feature_cols]
        
        # Проверяем на NaN
        if X_meta.isna().any().any():
            print("Предупреждение: обнаружены NaN значения в данных для stacking.")
            print("Используем метод взвешенного среднего вместо стекинга.")
            return self.fallback_strategy.combine(predictions, weights)
        
        try:
            # Получаем предсказания от метамодели
            final_predictions = self.metamodel.predict(X_meta)
            
            # Получаем вероятности (оценки аномальности)
            if hasattr(self.metamodel, 'predict_proba'):
                probas = self.metamodel.predict_proba(X_meta)
                final_scores = probas[:, 1]  # Вероятность аномального класса
            else:
                # Если метамодель не поддерживает predict_proba, используем взвешенное среднее
                detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                          for i in range(len(weights))]).T
                final_scores = np.sum(detector_scores * np.array(weights), axis=1)
        except Exception as e:
            print(f"Ошибка при применении стекинга: {e}")
            print("Используем метод взвешенного среднего вместо стекинга.")
            return self.fallback_strategy.combine(predictions, weights)
        
        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }
    
    def fit(self, data: pd.DataFrame, y: np.ndarray) -> None:
        """
        Обучение мета-модели для стекинга.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Данные для обучения
        y : np.ndarray
            Метки классов
        """
        if self.metamodel is None:
            self.metamodel = LogisticRegression(class_weight='balanced')
        
        try:
            self.metamodel.fit(data, y)
            print("Метамодель обучена.")
        except Exception as e:
            print(f"Ошибка при обучении метамодели: {e}")
            self.metamodel = None


class AnomalyTypeClassifier:
    """
    Классификатор типов аномалий на основе характеристик данных и детекторов.
    """
    
    def classify(self, result_df: pd.DataFrame, detectors: List[BaseAnomalyDetector]) -> pd.DataFrame:
        """
        Определение типов аномалий.
        
        Parameters:
        -----------
        result_df : pd.DataFrame
            Результаты обнаружения аномалий
        detectors : List[BaseAnomalyDetector]
            Список детекторов аномалий
            
        Returns:
        --------
        pd.DataFrame
            Результаты с добавленной информацией о типах аномалий
        """
        # Создаем копию датафрейма для изменений
        df = result_df.copy()
        
        # Инициализируем колонку для типов аномалий
        df['anomaly_type'] = 'Normal'
        
        # Фильтруем только аномалии
        anomalies = df[df['predicted_anomaly'] == 1]
        
        if len(anomalies) == 0:
            return df
        
        # Применяем различные классификаторы
        df = self._classify_dos_attacks(df)
        df = self._classify_volume_anomalies(df)
        df = self._classify_port_anomalies(df)
        df = self._classify_time_anomalies(df)
        df = self._classify_detector_specific_anomalies(df, detectors)
        
        # Если не удалось определить тип аномалии, помечаем как "Unknown"
        unknown_condition = (df['predicted_anomaly'] == 1) & (df['anomaly_type'] == 'Normal')
        df.loc[unknown_condition, 'anomaly_type'] = 'Unknown Anomaly'
        
        return df
    
    def _classify_dos_attacks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация DoS атак."""
        if 'dos_attack_type' in df.columns:
            dos_attacks = df[(df['predicted_anomaly'] == 1) & (df['dos_attack_type'] != 'Normal')]
            df.loc[dos_attacks.index, 'anomaly_type'] = dos_attacks['dos_attack_type']
        
        return df
    
    def _classify_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация объемных аномалий."""
        if all(col in df.columns for col in ['bytes', 'duration']):
            # Большой объем данных за короткое время
            volume_ratio = df['bytes'] / (df['duration'] + 0.1)  # +0.1 чтобы избежать деления на 0
            threshold = np.percentile(volume_ratio, 99)
            
            volume_anomalies = (df['predicted_anomaly'] == 1) & (volume_ratio > threshold)
            df.loc[volume_anomalies & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Volume Anomaly'
        
        return df
    
    def _classify_port_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация аномалий портов."""
        # Аномалии портов
        if 'dst_port' in df.columns:
            unusual_ports = [6667, 31337, 4444, 9001, 1337, 8080]  # Порты, часто используемые для атак
            port_anomalies = (df['predicted_anomaly'] == 1) & (df['dst_port'].isin(unusual_ports))
            df.loc[port_anomalies & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Unusual Port'
        
        # Сканирование портов
        if all(col in df.columns for col in ['packets', 'duration']):
            scan_condition = (df['predicted_anomaly'] == 1) & (df['packets'] <= 3) & (df['duration'] < 0.5)
            df.loc[scan_condition & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Port Scan'
        
        return df
    
    def _classify_time_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Классификация временных аномалий."""
        if 'timestamp' in df.columns:
            try:
                if df['timestamp'].dtype != 'datetime64[ns]':
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Необычное время активности (нерабочее время)
                work_hours = (df['timestamp'].dt.hour >= 8) & (df['timestamp'].dt.hour <= 18)
                work_days = (df['timestamp'].dt.dayofweek < 5)  # Пн-Пт
                
                time_anomalies = (df['predicted_anomaly'] == 1) & (~(work_hours & work_days))
                df.loc[time_anomalies & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Unusual Time'
            except Exception as e:
                print(f"Ошибка при анализе временных аномалий: {e}")
        
        return df
    
    def _classify_detector_specific_anomalies(self, df: pd.DataFrame, 
                                              detectors: List[BaseAnomalyDetector]) -> pd.DataFrame:
        """Классификация специфичных аномалий от разных детекторов."""
        for i, detector in enumerate(detectors):
            detector_name = detector.__class__.__name__
            
            if detector_name == 'SequenceAnomalyDetector':
                # Аномалии последовательностей
                sequence_condition = (df['predicted_anomaly'] == 1) & (df[f'anomaly_{detector_name}'] == 1)
                df.loc[sequence_condition & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Sequence Anomaly'
            
            elif detector_name == 'IsolationForestDetector':
                # Изолированные точки
                isolation_condition = (df['predicted_anomaly'] == 1) & (df[f'anomaly_{detector_name}'] == 1)
                if isolation_condition.sum() > 0:
                    isolation_score = df.loc[isolation_condition, f'score_{detector_name}']
                    
                    # Высокие оценки аномальности от Isolation Forest
                    if len(isolation_score) > 0:
                        percentile = np.percentile(isolation_score, 75) if len(isolation_score) > 4 else isolation_score.max()
                        high_score_condition = isolation_condition & (df[f'score_{detector_name}'] > percentile)
                        df.loc[high_score_condition & (df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Isolation Anomaly'
        
        return df


class EnsembleAnomalyDetector:
    """
    Ансамблевый детектор аномалий, объединяющий результаты нескольких базовых детекторов.
    
    Подходы к ансамблированию:
    1. Majority voting - голосование большинством (hard voting)
    2. Weighted average - взвешенное среднее оценок аномальности (soft voting)
    3. Rank-based fusion - объединение на основе ранжирования
    4. Stacking - использование метамодели
    
    Примеры:
    --------
    >>> from intellectshield.detectors.ensemble import EnsembleAnomalyDetector
    >>> from intellectshield.detectors.isolation_forest import IsolationForestDetector
    >>> from intellectshield.detectors.lof import LOFDetector
    >>>
    >>> # Создание ансамбля
    >>> ensemble = EnsembleAnomalyDetector()
    >>>
    >>> # Добавление детекторов
    >>> ensemble.add_detector(IsolationForestDetector(), weight=1.5)
    >>> ensemble.add_detector(LOFDetector(), weight=1.0)
    >>>
    >>> # Установка метода ансамблирования
    >>> ensemble.set_ensemble_method("weighted_average")
    >>>
    >>> # Обучение и предсказание
    >>> ensemble.train(train_data)
    >>> results = ensemble.predict(test_data)
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Инициализация ансамблевого детектора.
        
        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        """
        self.detectors: List[BaseAnomalyDetector] = []  # Список базовых детекторов
        self.weights: List[float] = []                 # Веса детекторов
        self.model_dir: str = model_dir
        self.ensemble_method: str = "weighted_average"  # По умолчанию используем взвешенное среднее
        self.training_summary: Dict[str, Any] = {}
        self.scaler = MinMaxScaler()  # Для нормализации оценок аномальности
        
        # Стратегии ансамблирования
        self.strategies: Dict[str, EnsembleStrategy] = {
            "majority_voting": MajorityVotingStrategy(),
            "weighted_average": WeightedAverageStrategy(),
            "rank_fusion": RankFusionStrategy(),
            "stacking": StackingStrategy()
        }
        
        # Классификатор типов аномалий
        self.anomaly_classifier = AnomalyTypeClassifier()
        
        # Создание директории для моделей, если не существует
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def add_detector(self, detector: BaseAnomalyDetector, weight: float = 1.0) -> 'EnsembleAnomalyDetector':
        """
        Добавление детектора в ансамбль.
        
        Parameters:
        -----------
        detector : BaseAnomalyDetector
            Детектор аномалий, наследующий от BaseAnomalyDetector
        weight : float
            Вес детектора в ансамбле (по умолчанию 1.0)
            
        Returns:
        --------
        EnsembleAnomalyDetector
            Текущий экземпляр для цепочки вызовов
        """
        self.detectors.append(detector)
        self.weights.append(weight)
        
        # Нормализуем веса, чтобы их сумма была равна 1
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"Детектор {detector.__class__.__name__} добавлен с весом {weight / total_weight:.4f}")
        
        return self
    
    def set_ensemble_method(self, method: str) -> 'EnsembleAnomalyDetector':
        """
        Установка метода ансамблирования.
        
        Parameters:
        -----------
        method : str
            Метод ансамблирования ("majority_voting", "weighted_average", "rank_fusion", "stacking")
            
        Returns:
        --------
        EnsembleAnomalyDetector
            Текущий экземпляр для цепочки вызовов
        """
        valid_methods = list(self.strategies.keys())
        
        if method not in valid_methods:
            raise ValueError(f"Недопустимый метод ансамблирования. Допустимые методы: {valid_methods}")
        
        self.ensemble_method = method
        print(f"Установлен метод ансамблирования: {method}")
        
        return self
    
    def train(self, data: pd.DataFrame) -> 'EnsembleAnomalyDetector':
        """
        Обучение всех детекторов в ансамбле.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
            
        Returns:
        --------
        EnsembleAnomalyDetector
            Текущий экземпляр для цепочки вызовов
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")
        
        print(f"Начало обучения ансамбля из {len(self.detectors)} детекторов...")
        start_time = time.time()
        
        # Обучаем каждый детектор
        for i, detector in enumerate(self.detectors):
            print(f"Обучение детектора {i+1}/{len(self.detectors)}: {detector.__class__.__name__}")
            detector.train(data)
        
        # Если используется stacking, обучаем метамодель
        if self.ensemble_method == "stacking":
            self._train_stacking_metamodel(data)
        
        # Запись статистики обучения
        training_time = time.time() - start_time
        self.training_summary = {
            'ensemble_method': self.ensemble_method,
            'detectors': [detector.__class__.__name__ for detector in self.detectors],
            'weights': self.weights,
            'training_time': training_time
        }
        
        print(f"Обучение ансамбля завершено за {training_time:.2f} секунд")
        
        return self
    
    def _train_stacking_metamodel(self, data: pd.DataFrame) -> None:
        """
        Обучение метамодели для стекинга.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
        """
        print("Обучение метамодели для stacking...")
        
        # Получаем предсказания от всех базовых детекторов
        predictions = self._get_all_detector_predictions(data)
        
        # Если есть метки аномалий, можем обучить метамодель
        if 'is_anomaly' in data.columns:
            # Проверяем на NaN значения
            feature_cols = [f'anomaly_score_{i}' for i in range(len(self.detectors))]
            X_meta = predictions[feature_cols]
            
            if X_meta.isna().any().any():
                print("Предупреждение: обнаружены NaN значения в данных для stacking.")
                print("Метамодель не будет обучена. Будет использоваться метод взвешенного среднего.")
                self.ensemble_method = "weighted_average"
            else:
                try:
                    # Подготавливаем обучающий набор
                    y_meta = data['is_anomaly'].values
                    
                    # Обучаем метамодель через стратегию stacking
                    stacking_strategy = self.strategies["stacking"]
                    if isinstance(stacking_strategy, StackingStrategy):
                        stacking_strategy.fit(X_meta, y_meta)
                    else:
                        print("Ошибка: стратегия stacking имеет неверный тип.")
                        self.ensemble_method = "weighted_average"
                except Exception as e:
                    print(f"Ошибка при обучении метамодели: {e}")
                    self.ensemble_method = "weighted_average"
        else:
            print("Предупреждение: данные не содержат меток аномалий. Метамодель не будет обучена.")
            self.ensemble_method = "weighted_average"
            print(f"Метод ансамблирования изменен на: {self.ensemble_method}")
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий с использованием ансамбля детекторов.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Исходные данные с добавленными предсказаниями и аномальными оценками
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")
        
        # Получаем предсказания от всех детекторов
        predictions = self._get_all_detector_predictions(data)
        
        # Применяем выбранный метод ансамблирования через стратегию
        try:
            if self.ensemble_method in self.strategies:
                strategy = self.strategies[self.ensemble_method]
                result = strategy.combine(predictions, self.weights)
            else:
                # По умолчанию используем взвешенное среднее
                strategy = self.strategies["weighted_average"]
                result = strategy.combine(predictions, self.weights)
        except Exception as e:
            print(f"Ошибка при применении метода {self.ensemble_method}: {e}")
            print("Используем метод взвешенного среднего как запасной вариант.")
            strategy = self.strategies["weighted_average"]
            result = strategy.combine(predictions, self.weights)
        
        # Создаем итоговый результат
        result_df = data.copy()
        result_df['predicted_anomaly'] = result['predicted_anomaly']
        result_df['anomaly_score'] = result['anomaly_score']
        
        # Добавляем прогнозы и оценки от каждого детектора
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__
            result_df[f'anomaly_{detector_name}'] = predictions[f'predicted_anomaly_{i}']
            result_df[f'score_{detector_name}'] = predictions[f'anomaly_score_{i}']
        
        # Дополнительная информация о типах аномалий
        result_df = self.anomaly_classifier.classify(result_df, self.detectors)
        
        return result_df
    
    def _get_all_detector_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Получение предсказаний от всех детекторов в ансамбле.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Датафрейм с предсказаниями и оценками от всех детекторов
        """
        all_predictions = pd.DataFrame(index=data.index)
        
        for i, detector in enumerate(self.detectors):
            # Получаем предсказания от текущего детектора
            detector_result = detector.predict(data)
            
            # Добавляем предсказания и оценки в общий датафрейм
            all_predictions[f'predicted_anomaly_{i}'] = detector_result['predicted_anomaly']
            all_predictions[f'anomaly_score_{i}'] = detector_result['anomaly_score']
        
        return all_predictions
    
    def evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Оценка производительности ансамблевого детектора.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные с истинными метками аномалий
            
        Returns:
        --------
        Dict[str, Any]
            Словарь с метриками производительности
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Для оценки необходимы данные с колонкой 'is_anomaly'")
        
        # Получаем предсказания
        result_df = self.predict(data)
        
        # Вычисляем метрики
        y_true = data['is_anomaly'].values
        y_pred = result_df['predicted_anomaly'].values
        y_score = result_df['anomaly_score'].values
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)
        
        # Precision, Recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_score)
        except Exception as e:
            print(f"Ошибка при вычислении AUC-ROC: {e}")
            auc_roc = None
        
        # Вычисляем метрики для каждого детектора
        detector_metrics = []
        
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__
            
            detector_pred = result_df[f'anomaly_{detector_name}'].values
            detector_score = result_df[f'score_{detector_name}'].values
            
            # Precision, Recall, F1 для текущего детектора
            d_precision, d_recall, d_f1, _ = precision_recall_fscore_support(y_true, detector_pred, average='binary')
            
            # AUC-ROC для текущего детектора
            try:
                d_auc_roc = roc_auc_score(y_true, detector_score)
            except Exception as e:
                print(f"Ошибка при вычислении AUC-ROC для детектора {detector_name}: {e}")
                d_auc_roc = None
            
            detector_metrics.append({
                'detector': detector_name,
                'precision': d_precision,
                'recall': d_recall,
                'f1_score': d_f1,
                'auc_roc': d_auc_roc
            })
        
        # Результаты
        evaluation = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'detector_metrics': detector_metrics,
            'ensemble_method': self.ensemble_method
        }
        
        return evaluation
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """
        Сохранение ансамблевого детектора в файл.
        
        Parameters:
        -----------
        filename : Optional[str]
            Имя файла для сохранения
            
        Returns:
        --------
        str
            Путь к сохраненному файлу
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")
        
        # Если имя файла не указано, генерируем его из типа модели и времени
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"EnsembleDetector_{timestamp}.joblib"
        
        # Полный путь к файлу
        filepath = os.path.join(self.model_dir, filename)
        
        # Создаем словарь с моделью и метаданными
        model_data = {
            'detectors': self.detectors,
            'weights': self.weights,
            'ensemble_method': self.ensemble_method,
            'training_summary': self.training_summary,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Если используется stacking, сохраняем и метамодель
        stacking_strategy = self.strategies.get("stacking")
        if isinstance(stacking_strategy, StackingStrategy) and stacking_strategy.metamodel is not None:
            model_data['metamodel'] = stacking_strategy.metamodel
        
        # Сохраняем в файл
        try:
            joblib.dump(model_data, filepath)
            print(f"Ансамблевый детектор сохранен в {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении модели: {e}")
            raise
        
        return filepath
    
    def load_model(self, filepath: str) -> 'EnsembleAnomalyDetector':
        """
        Загрузка ансамблевого детектора из файла.
        
        Parameters:
        -----------
        filepath : str
            Путь к файлу с сохраненной моделью
            
        Returns:
        --------
        EnsembleAnomalyDetector
            Текущий экземпляр для цепочки вызовов
        """
        try:
            # Загружаем данные из файла
            model_data = joblib.load(filepath)
            
            # Загружаем компоненты
            self.detectors = model_data['detectors']
            self.weights = model_data['weights']
            self.ensemble_method = model_data['ensemble_method']
            self.training_summary = model_data['training_summary']
            
            # Если есть метамодель, загружаем и ее
            if 'metamodel' in model_data:
                stacking_strategy = self.strategies.get("stacking")
                if isinstance(stacking_strategy, StackingStrategy):
                    stacking_strategy.metamodel = model_data['metamodel']
            
            print(f"Ансамблевый детектор загружен из {filepath}")
            print(f"Метод ансамблирования: {self.ensemble_method}")
            print(f"Количество детекторов: {len(self.detectors)}")
            
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            raise
        
        return self
