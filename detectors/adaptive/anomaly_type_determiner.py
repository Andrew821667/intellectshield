"""
Модуль для определения типов аномалий.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

from .profile_manager import ProfileManager


class AnomalyTypeDeterminer:
    """
    Класс для определения типов аномалий.
    
    Этот класс отвечает за определение конкретных типов аномалий
    на основе различных оценок и характеристик данных.
    """
    
    def __init__(self, profile_manager: ProfileManager):
        """
        Инициализация определителя типов аномалий.
        
        Parameters:
        -----------
        profile_manager : ProfileManager
            Менеджер профилей
        """
        self.profile_manager = profile_manager
    
    def determine_anomaly_types(self, result_df: pd.DataFrame, data: pd.DataFrame, 
                              scores: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Определяет типы обнаруженных аномалий.
        
        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты обнаружения аномалий
        data : pandas.DataFrame
            Предобработанные данные
        scores : dict
            Словарь с оценками аномальности разных типов
            
        Returns:
        --------
        pandas.DataFrame
            Результаты с определенными типами аномалий
        """
        # Добавляем колонку для типа аномалии
        result_df['anomaly_type'] = 'Normal'
        
        # Определяем индексы аномальных образцов
        anomaly_indices = result_df[result_df['predicted_anomaly'] == 1].index
        
        # Определяем основной тип для каждой аномалии
        for i in anomaly_indices:
            # Получаем индекс в массивах оценок
            idx = result_df.index.get_loc(i)
            
            # Определяем наиболее значимый тип аномалии
            stat_score = scores['statistical'][idx]
            context_score = scores['contextual'][idx]
            ml_score = scores['ml'][idx]
            collective_score = scores['collective'][idx]
            
            # Определяем максимальную оценку
            max_score = max(stat_score, context_score, ml_score, collective_score)
            
            # Определяем тип аномалии
            if max_score == stat_score:
                result_df.at[i, 'anomaly_type'] = self.determine_statistical_anomaly_type(data.iloc[idx])
            elif max_score == context_score:
                result_df.at[i, 'anomaly_type'] = 'Contextual Anomaly'
            elif max_score == ml_score:
                result_df.at[i, 'anomaly_type'] = 'Complex Anomaly'
            elif max_score == collective_score:
                result_df.at[i, 'anomaly_type'] = 'Collective Anomaly'
        
        return result_df
    
    def determine_statistical_anomaly_type(self, sample: pd.Series) -> str:
        """
        Определяет конкретный тип статистической аномалии.
        
        Parameters:
        -----------
        sample : pandas.Series
            Образец для анализа
            
        Returns:
        --------
        str
            Тип аномалии
        """
        # Признаки для обнаружения DoS-атак
        if 'is_dos_like' in sample and sample['is_dos_like'] == 1:
            return 'DoS Attack'
        
        # Признаки для обнаружения сканирования портов
        if 'is_port_scan_like' in sample and sample['is_port_scan_like'] == 1:
            return 'Port Scan'
        
        # Признаки для обнаружения аномалий объема
        if 'bytes_per_second' in sample:
            global_profile = self.profile_manager.get_global_profile()
            if ('bytes_per_second' in global_profile and 
                sample['bytes_per_second'] > 3 * global_profile['bytes_per_second'].get('mean', 0)):
                return 'Volume Anomaly'
        
        # Признаки для обнаружения аномалий портов
        if 'dst_port_suspicious' in sample and sample['dst_port_suspicious'] == 1:
            return 'Suspicious Port'
        
        # Признаки для обнаружения временных аномалий
        if ('hour_of_day' in sample and 
            (sample['hour_of_day'] < 6 or sample['hour_of_day'] > 22)):
            return 'Time Anomaly'
        
        # По умолчанию - просто статистическая аномалия
        return 'Statistical Anomaly'
