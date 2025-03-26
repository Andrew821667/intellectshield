"""
Модуль для расчета оценок аномальности.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

from .profile_manager import ProfileManager
from .threshold_manager import ThresholdManager
from .ml_models_manager import MLModelsManager


class AnomalyScoreCalculator:
    """
    Класс для расчета различных оценок аномальности.
    
    Этот класс отвечает за вычисление оценок аномальности
    различных типов (статистические, контекстуальные, ML и др.)
    и их комбинирование.
    """
    
    def __init__(self, profile_manager: ProfileManager, 
               threshold_manager: ThresholdManager,
               ml_models_manager: MLModelsManager):
        """
        Инициализация калькулятора оценок аномальности.
        
        Parameters:
        -----------
        profile_manager : ProfileManager
            Менеджер профилей
        threshold_manager : ThresholdManager
            Менеджер порогов
        ml_models_manager : MLModelsManager
            Менеджер моделей ML
        """
        self.profile_manager = profile_manager
        self.threshold_manager = threshold_manager
        self.ml_models_manager = ml_models_manager
    
    def calculate_scores(self, data: pd.DataFrame, threshold_multiplier: float) -> Dict[str, np.ndarray]:
        """
        Вычисляет оценки аномальности различных типов.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        dict
            Словарь с оценками аномальности разных типов
        """
        # Вычисляем оценки для разных типов аномалий
        statistical_scores = self.calculate_statistical_scores(data, threshold_multiplier)
        contextual_scores = self.calculate_contextual_scores(data, threshold_multiplier)
        ml_scores = self.calculate_ml_scores(data)
        collective_scores = self.calculate_collective_scores(data)
        
        return {
            'statistical': statistical_scores,
            'contextual': contextual_scores,
            'ml': ml_scores,
            'collective': collective_scores
        }
    
    def combine_scores(self, scores: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Комбинирует оценки аномальности с весами.
        
        Parameters:
        -----------
        scores : dict
            Словарь с оценками аномальности
            
        Returns:
        --------
        numpy.ndarray
            Комбинированные оценки
        """
        # Веса для разных типов аномалий
        weights = {
            'statistical': 0.4,
            'contextual': 0.3,
            'ml': 0.2,
            'collective': 0.1
        }
        
        # Комбинируем оценки
        combined_scores = np.zeros(len(scores['statistical']))
        for score_type, score_values in scores.items():
            combined_scores += weights[score_type] * score_values
        
        return combined_scores
    
    def calculate_statistical_scores(self, data: pd.DataFrame, threshold_multiplier: float) -> np.ndarray:
        """
        Вычисляет оценки статистических аномалий.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        numpy.ndarray
            Оценки статистических аномалий
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # Получаем глобальный профиль
        global_profile = self.profile_manager.get_global_profile()
        
        # Анализируем только числовые признаки из глобального профиля
        for feature, stats in global_profile.items():
            if feature in data.columns:
                # Проверяем наличие необходимых статистик
                if all(key in stats for key in ['mean', 'std']):
                    # Вычисляем z-score для каждого образца
                    feature_data = data[feature].fillna(0).values
                    z_scores = np.abs((feature_data - stats['mean']) / (stats['std'] + 1e-10))
                    
                    # Увеличиваем общую оценку аномальности для выбросов
                    # Чем больше z-score, тем больше увеличиваем оценку
                    anomaly_contribution = np.maximum(0, z_scores - threshold_multiplier)
                    scores += anomaly_contribution
        
        return scores
    
    def calculate_contextual_scores(self, data: pd.DataFrame, threshold_multiplier: float) -> np.ndarray:
        """
        Вычисляет оценки контекстуальных аномалий.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        numpy.ndarray
            Оценки контекстуальных аномалий
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # 1. Временной контекст (час дня)
        if 'hour_of_day' in data.columns:
            temporal_profile = self.profile_manager.get_temporal_profile('hour_of_day')
            
            for i, hour in enumerate(data['hour_of_day']):
                hour_str = str(int(hour))
                
                if hour_str in temporal_profile:
                    hour_profile = temporal_profile[hour_str]
                    
                    # Проверяем соответствие профилю для каждого признака
                    for feature, stats in hour_profile.items():
                        if feature in data.columns:
                            # Вычисляем отклонение от профиля
                            value = data.iloc[i][feature]
                            mean = stats.get('mean', 0)
                            std = stats.get('std', 1) + 1e-10  # Избегаем деления на ноль
                            
                            z_score = np.abs((value - mean) / std)
                            
                            # Увеличиваем оценку аномальности, если отклонение значительное
                            if z_score > threshold_multiplier:
                                scores[i] += z_score - threshold_multiplier
        
        # 2. Рабочие/нерабочие часы
        if 'is_working_hours' in data.columns:
            for i, is_working in enumerate(data['is_working_hours']):
                context_key = 'working' if is_working == 1 else 'non_working'
                context_profile = self.profile_manager.get_contextual_profile('working_hours', context_key)
                
                # Проверяем соответствие профилю для каждого признака
                for feature, stats in context_profile.items():
                    if feature in data.columns:
                        # Вычисляем отклонение от профиля
                        value = data.iloc[i][feature]
                        mean = stats.get('mean', 0)
                        std = stats.get('std', 1) + 1e-10
                        
                        z_score = np.abs((value - mean) / std)
                        
                        # Увеличиваем оценку аномальности, если отклонение значительное
                        if z_score > threshold_multiplier:
                            scores[i] += z_score - threshold_multiplier
        
        # 3. Протокол-порт контекст
        if 'protocol_num' in data.columns and 'dst_port' in data.columns:
            for i in range(len(data)):
                protocol = data.iloc[i]['protocol_num']
                port = data.iloc[i]['dst_port']
                
                # Получаем профиль для этой комбинации протокол-порт
                context_profile = self.profile_manager.get_protocol_port_profile(protocol, port)
                
                # Проверяем соответствие профилю для каждого признака
                for feature, stats in context_profile.items():
                    if feature in data.columns:
                        # Вычисляем отклонение от профиля
                        value = data.iloc[i][feature]
                        mean = stats.get('mean', 0)
                        std = stats.get('std', 1) + 1e-10
                        
                        z_score = np.abs((value - mean) / std)
                        
                        # Увеличиваем оценку аномальности, если отклонение значительное
                        if z_score > threshold_multiplier:
                            scores[i] += z_score - threshold_multiplier
        
        return scores
    
    def calculate_ml_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет оценки аномалий машинного обучения.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномалий машинного обучения
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # Получаем оценки от различных моделей
        if_scores = self.ml_models_manager.get_isolation_forest_scores(data, self.profile_manager.profiles.get('feature_groups', {}))
        lof_scores = self.ml_models_manager.get_lof_scores(data, self.profile_manager.profiles.get('feature_groups', {}))
        dbscan_scores = self.ml_models_manager.get_dbscan_scores(data, self.profile_manager.profiles.get('feature_groups', {}))
        
        # Нормализуем оценки
        if_scores_norm = self._normalize_scores(if_scores)
        lof_scores_norm = self._normalize_scores(lof_scores)
        
        # Комбинируем оценки с весами
        scores = 0.5 * if_scores_norm + 0.3 * lof_scores_norm + 0.2 * dbscan_scores
        
        return scores
    
    
    
    
    def calculate_collective_scores(self, data: pd.DataFrame) -> np.ndarray:
        """
        Вычисляет оценки коллективных аномалий.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        numpy.ndarray
            Оценки коллективных аномалий
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # Проверяем наличие временной метки
        if 'timestamp' not in data.columns:
            return scores
        
        # Анализируем временные паттерны для ключевых признаков
        key_features = ['bytes_per_second', 'packets_per_second', 'connection_count']
        available_features = [f for f in key_features if f in data.columns]
        
        if not available_features:
            return scores
        
        try:
            # Сохраняем оригинальный индекс для последующего маппинга
            data_with_pos = data.copy().reset_index(drop=False)
            data_with_pos['_pos_in_array'] = range(len(data))
            
            # Сортируем данные по времени
            sorted_data = data_with_pos.sort_values('timestamp')
            
            # Размер окна для анализа последовательностей
            window_size = min(10, len(data) // 10)
            if window_size < 2:  # Предотвращаем слишком маленькие окна
                return scores
            
            # Анализируем каждый доступный признак
            for feature in available_features:
                # Создаем скользящее окно
                for i in range(len(sorted_data) - window_size + 1):
                    window_data = sorted_data.iloc[i:i+window_size]
                    window = window_data[feature].values
                    
                    # Анализ резких изменений в окне
                    if len(window) > 1:
                        # Вычисляем разности между соседними значениями
                        diffs = np.abs(np.diff(window))
                        if len(diffs) == 0:
                            continue
                            
                        mean_diff = np.mean(diffs)
                        if mean_diff == 0:
                            continue
                        
                        # Если есть резкие изменения, увеличиваем оценку аномальности
                        for j in range(len(diffs)):
                            if diffs[j] > 3 * mean_diff:
                                try:
                                    # Получаем позиции в оригинальном массиве scores
                                    pos1 = window_data['_pos_in_array'].iloc[j]
                                    pos2 = window_data['_pos_in_array'].iloc[j+1]
                                    
                                    anomaly_score = diffs[j] / (mean_diff + 1e-10) - 3
                                    scores[pos1] += anomaly_score
                                    scores[pos2] += anomaly_score
                                except Exception as e:
                                    # В случае ошибки с индексами, пропускаем эту итерацию
                                    pass
                                    
        except Exception as e:
            # Логируем ошибку и возвращаем нулевые оценки
            print(f"Ошибка при обнаружении коллективных аномалий: {e.__class__.__name__}")
            
        return scores
        
        # Анализируем временные паттерны для ключевых признаков
        key_features = ['bytes_per_second', 'packets_per_second', 'connection_count']
        available_features = [f for f in key_features if f in data.columns]
        
        if not available_features:
            print("ОТЛАДКА: Не найдены ключевые признаки в данных", data.columns)
            return scores
        
        try:
            print(f"ОТЛАДКА: Доступные признаки: {available_features}")
            
            # Сортируем данные по времени
            sorted_data = data.sort_values('timestamp').reset_index(drop=False)
            
            print(f"ОТЛАДКА: Размер отсортированных данных: {sorted_data.shape}")
            
            # Получаем маппинг между отсортированными и оригинальными индексами
            idx_mapping = dict(enumerate(sorted_data.index))
            
            # Размер окна для анализа последовательностей
            window_size = min(10, len(data) // 10)
            if window_size < 2:  # Предотвращаем слишком маленькие окна
                print("ОТЛАДКА: Слишком маленький размер окна:", window_size)
                return scores
            
            print(f"ОТЛАДКА: Размер окна: {window_size}")
            
            # Анализируем каждый доступный признак
            for feature in available_features:
                print(f"ОТЛАДКА: Анализ признака: {feature}")
                # Создаем скользящее окно
                for i in range(len(sorted_data) - window_size + 1):
                    window = sorted_data[feature].iloc[i:i+window_size].values
                    
                    # Анализ резких изменений в окне
                    if len(window) > 1:
                        # Вычисляем разности между соседними значениями
                        diffs = np.abs(np.diff(window))
                        if len(diffs) == 0:
                            continue
                            
                        mean_diff = np.mean(diffs)
                        if mean_diff == 0:
                            continue
                        
                        # Если есть резкие изменения, увеличиваем оценку аномальности
                        for j in range(len(diffs)):
                            if diffs[j] > 3 * mean_diff:
                                try:
                                    orig_idx1 = idx_mapping[i + j]
                                    orig_idx2 = idx_mapping[i + j + 1]
                                    
                                    print(f"ОТЛАДКА: Найдено резкое изменение: {diffs[j]:.2f} vs среднее {mean_diff:.2f}")
                                    print(f"ОТЛАДКА: Индексы: i={i}, j={j}, orig_idx1={orig_idx1}, orig_idx2={orig_idx2}")
                                    
                                    # Используем позиции в оригинальном массиве scores
                                    pos1 = data.index.get_loc(orig_idx1)
                                    pos2 = data.index.get_loc(orig_idx2)
                                    
                                    anomaly_score = diffs[j] / (mean_diff + 1e-10) - 3
                                    scores[pos1] += anomaly_score
                                    scores[pos2] += anomaly_score
                                except Exception as e:
                                    print(f"ОТЛАДКА: Ошибка при обработке изменений: {e}, тип: {type(e)}")
                                    print(f"ОТЛАДКА: data.index = {data.index}, orig_idx1 = {orig_idx1}")
        except Exception as e:
            print(f"ОТЛАДКА: Общая ошибка в collective_scores: {e}, тип: {type(e)}")
            
        return scores
        
        # Анализируем временные паттерны для ключевых признаков
        key_features = ['bytes_per_second', 'packets_per_second', 'connection_count']
        available_features = [f for f in key_features if f in data.columns]
        
        if not available_features:
            return scores
        
        try:
            # Сортируем данные по времени
            sorted_data = data.sort_values('timestamp').reset_index(drop=False)
            
            # Получаем маппинг между отсортированными и оригинальными индексами
            idx_mapping = dict(enumerate(sorted_data.index))
            
            # Размер окна для анализа последовательностей
            window_size = min(10, len(data) // 10)
            if window_size < 2:  # Предотвращаем слишком маленькие окна
                return scores
            
            # Анализируем каждый доступный признак
            for feature in available_features:
                # Создаем скользящее окно
                for i in range(len(sorted_data) - window_size + 1):
                    window = sorted_data[feature].iloc[i:i+window_size].values
                    
                    # Анализ резких изменений в окне
                    if len(window) > 1:
                        # Вычисляем разности между соседними значениями
                        diffs = np.abs(np.diff(window))
                        if len(diffs) == 0:
                            continue
                            
                        mean_diff = np.mean(diffs)
                        if mean_diff == 0:
                            continue
                        
                        # Если есть резкие изменения, увеличиваем оценку аномальности
                        for j in range(len(diffs)):
                            if diffs[j] > 3 * mean_diff:
                                orig_idx1 = idx_mapping[i + j]
                                orig_idx2 = idx_mapping[i + j + 1]
                                
                                # Используем позиции в оригинальном массиве scores
                                pos1 = data.index.get_loc(orig_idx1)
                                pos2 = data.index.get_loc(orig_idx2)
                                
                                anomaly_score = diffs[j] / (mean_diff + 1e-10) - 3
                                scores[pos1] += anomaly_score
                                scores[pos2] += anomaly_score
        except Exception as e:
            print(f"Ошибка при обнаружении коллективных аномалий: {e}")
            
        return scores
        
        # Анализируем временные паттерны для ключевых признаков
        key_features = ['bytes_per_second', 'packets_per_second', 'connection_count']
        available_features = [f for f in key_features if f in data.columns]
        
        if not available_features:
            return scores
        
        # Сортируем данные по времени
        sorted_indices = data['timestamp'].argsort()
        
        # Размер окна для анализа последовательностей
        window_size = min(10, len(data) // 10)
        
        # Анализируем каждый доступный признак
        for feature in available_features:
            # Создаем скользящее окно
            for i in range(len(sorted_indices) - window_size + 1):
                window_indices = sorted_indices[i:i+window_size]
                window = data.loc[window_indices, feature].values
                
                # Анализ резких изменений в окне
                if len(window) > 1:
                    # Вычисляем разности между соседними значениями
                    diffs = np.abs(np.diff(window))
                    mean_diff = np.mean(diffs)
                    
                    # Если есть резкие изменения, увеличиваем оценку аномальности
                    for j in range(len(diffs)):
                        if diffs[j] > 3 * mean_diff:
                            scores[window_indices[j]] += diffs[j] / (mean_diff + 1e-10) - 3
                            scores[window_indices[j + 1]] += diffs[j] / (mean_diff + 1e-10) - 3
        
        return scores
    
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
