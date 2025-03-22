import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseAnomalyDetector(ABC):
    """
    Базовый абстрактный класс для детекторов аномалий.
    Все детекторы аномалий должны наследоваться от этого класса и реализовывать
    методы fit и predict.
    """
    
    def __init__(self):
        """
        Инициализация базового детектора аномалий.
        """
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Метод для обучения детектора аномалий.
        
        Параметры:
        ----------
        data : pd.DataFrame
            Обучающие данные
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Метод для обнаружения аномалий.
        
        Параметры:
        ----------
        data : pd.DataFrame
            Данные для анализа
            
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с результатами обнаружения аномалий
        """
        pass
    
    def fit_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение и обнаружение аномалий в одном вызове.
        
        Параметры:
        ----------
        data : pd.DataFrame
            Данные для обучения и анализа
            
        Возвращает:
        -----------
        pd.DataFrame
            DataFrame с результатами обнаружения аномалий
        """
        self.fit(data)
        return self.predict(data)
