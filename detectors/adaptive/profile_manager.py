"""
Модуль для управления профилями нормального поведения.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd


class ProfileManager:
    """
    Класс для управления профилями нормального поведения.
    
    Профили содержат статистические характеристики для различных контекстов:
    - Глобальный профиль (общие статистики)
    - Временные профили (для разных временных интервалов)
    - Контекстуальные профили (для различных условий)
    """
    
    def __init__(self):
        """
        Инициализация менеджера профилей.
        """
        self.profiles = {}
    
    def set_profiles(self, profiles: Dict) -> None:
        """
        Устанавливает профили для использования.
        
        Parameters:
        -----------
        profiles : dict
            Словарь с профилями
        """
        self.profiles = profiles
    
    def get_global_profile(self) -> Dict:
        """
        Возвращает глобальный профиль.
        
        Returns:
        --------
        dict
            Глобальный профиль
        """
        return self.profiles.get('global', {})
    
    def get_temporal_profile(self, key: str) -> Dict:
        """
        Возвращает временной профиль.
        
        Parameters:
        -----------
        key : str
            Ключ временного профиля (например, 'hour_of_day')
            
        Returns:
        --------
        dict
            Временной профиль
        """
        return self.profiles.get('temporal', {}).get(key, {})
    
    def get_contextual_profile(self, context_type: str, context_key: str) -> Dict:
        """
        Возвращает контекстуальный профиль.
        
        Parameters:
        -----------
        context_type : str
            Тип контекста (например, 'working_hours')
        context_key : str
            Ключ контекста (например, 'working')
            
        Returns:
        --------
        dict
            Контекстуальный профиль
        """
        return self.profiles.get('contextual', {}).get(context_type, {}).get(context_key, {})
    
    def get_protocol_port_profile(self, protocol: int, port: int) -> Dict:
        """
        Возвращает профиль для комбинации протокол-порт.
        
        Parameters:
        -----------
        protocol : int
            Номер протокола
        port : int
            Номер порта
            
        Returns:
        --------
        dict
            Профиль протокол-порт
        """
        context_key = f'protocol_{protocol}_port_{port}'
        return self.get_contextual_profile('protocol_port', context_key)
    
    def get_feature_stats(self, feature: str, profile_type: str = 'global', 
                        context_type: Optional[str] = None, context_key: Optional[str] = None) -> Dict:
        """
        Возвращает статистики для признака из указанного профиля.
        
        Parameters:
        -----------
        feature : str
            Имя признака
        profile_type : str
            Тип профиля ('global', 'temporal', 'contextual')
        context_type : str, optional
            Тип контекста
        context_key : str, optional
            Ключ контекста
            
        Returns:
        --------
        dict
            Статистики признака
        """
        if profile_type == 'global':
            return self.get_global_profile().get(feature, {})
        elif profile_type == 'temporal':
            if context_type:
                return self.get_temporal_profile(context_type).get(context_key, {}).get(feature, {})
        elif profile_type == 'contextual':
            if context_type and context_key:
                return self.get_contextual_profile(context_type, context_key).get(feature, {})
        
        return {}
