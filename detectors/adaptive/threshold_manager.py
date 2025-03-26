"""
Модуль для управления порогами обнаружения аномалий.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd


class ThresholdManager:
    """
    Класс для управления порогами обнаружения аномалий.
    
    Этот класс отвечает за вычисление, хранение и применение
    порогов для обнаружения аномалий в данных.
    """
    
    def __init__(self):
        """
        Инициализация менеджера порогов.
        """
        self.thresholds = {}
    
    def calculate_thresholds(self, data: pd.DataFrame, profiles: Dict, feature_groups: Dict) -> None:
        """
        Вычисляет пороги для обнаружения аномалий.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для вычисления порогов
        profiles : dict
            Профили нормального поведения
        feature_groups : dict
            Словарь с группами признаков
        """
        # Вычисляем пороги для числовых признаков на основе глобального профиля
        if 'global' in profiles:
            for feature, stats in profiles['global'].items():
                if feature in feature_groups.get('numeric', []):
                    # Проверяем наличие необходимых статистик
                    if all(key in stats for key in ['mean', 'std', 'q1', 'q3', 'iqr']):
                        # Порог на основе z-score (количество стандартных отклонений от среднего)
                        z_threshold = stats['mean'] + 3 * stats['std']
                        
                        # Порог на основе межквартильного размаха (IQR)
                        iqr_threshold = stats['q3'] + 1.5 * stats['iqr']
                        
                        # Используем более консервативный порог
                        self.thresholds[feature] = max(z_threshold, iqr_threshold)
                        
                        # Если порог слишком близок к среднему, увеличиваем его
                        if self.thresholds[feature] < stats['mean'] * 1.2:
                            self.thresholds[feature] = stats['mean'] * 1.5
    
    def get_threshold(self, feature: str) -> float:
        """
        Возвращает порог для признака.
        
        Parameters:
        -----------
        feature : str
            Имя признака
            
        Returns:
        --------
        float
            Порог для признака
        """
        return self.thresholds.get(feature, float('inf'))
    
    def apply_threshold_multiplier(self, feature: str, threshold_multiplier: float) -> float:
        """
        Применяет множитель к порогу.
        
        Parameters:
        -----------
        feature : str
            Имя признака
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        float
            Скорректированный порог
        """
        threshold = self.get_threshold(feature)
        return threshold * threshold_multiplier
