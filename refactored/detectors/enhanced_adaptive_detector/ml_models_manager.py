"""
Модуль для управления моделями машинного обучения.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


class MLModelsManager:
    """
    Класс для управления моделями машинного обучения.
    
    Этот класс отвечает за создание, обучение и применение
    моделей машинного обучения для обнаружения аномалий.
    """
    
    def __init__(self):
        """
        Инициализация менеджера моделей.
        """
        # Инициализируем модели
        self.ml_models = {
            'isolation_forest': None,
            'lof': None,
            'dbscan': None
        }
    
    def train_models(self, data: pd.DataFrame, feature_groups: Dict) -> None:
        """
        Обучает модели машинного обучения.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
        feature_groups : dict
            Словарь с группами признаков
        """
        # Отбираем числовые признаки для моделей МО
        numeric_features = [f for f in feature_groups.get('numeric', []) 
                          if f in data.columns]
        
        if not numeric_features:
            print("Предупреждение: не найдены числовые признаки для обучения ML-моделей")
            return
        
        # Создаем обучающий набор
        X_train = data[numeric_features].fillna(0)
        
        # 1. Isolation Forest
        print("Обучение Isolation Forest...")
        self.ml_models['isolation_forest'] = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
            n_jobs=-1
        )
        self.ml_models['isolation_forest'].fit(X_train)
        
        # 2. Local Outlier Factor
        print("Обучение Local Outlier Factor...")
        self.ml_models['lof'] = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05,
            novelty=True,
            n_jobs=-1
        )
        self.ml_models['lof'].fit(X_train)
        
        # 3. DBSCAN (для кластеризации)
        print("Обучение DBSCAN...")
        self.ml_models['dbscan'] = DBSCAN(
            eps=0.5,
            min_samples=5,
            n_jobs=-1
        )
        # DBSCAN не имеет отдельного метода fit
        _ = self.ml_models['dbscan'].fit_predict(X_train)
    
    def get_isolation_forest_scores(self, data: pd.DataFrame, feature_groups: Dict) -> np.ndarray:
        """
        Возвращает оценки аномальности от Isolation Forest.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        feature_groups : dict
            Словарь с группами признаков
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        if self.ml_models['isolation_forest'] is None:
            return np.zeros(len(data))
        
        # Отбираем числовые признаки
        numeric_features = [f for f in feature_groups.get('numeric', []) 
                          if f in data.columns]
        
        if not numeric_features:
            return np.zeros(len(data))
        
        # Создаем тестовый набор
        X_test = data[numeric_features].fillna(0)
        
        try:
            # Получаем аномальные оценки от Isolation Forest
            # Отрицательные значения для аномалий, положительные для нормальных точек
            return -self.ml_models['isolation_forest'].decision_function(X_test)
        except Exception as e:
            print(f"Ошибка при использовании Isolation Forest: {e}")
            return np.zeros(len(data))
    
    def get_lof_scores(self, data: pd.DataFrame, feature_groups: Dict) -> np.ndarray:
        """
        Возвращает оценки аномальности от Local Outlier Factor.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        feature_groups : dict
            Словарь с группами признаков
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        if self.ml_models['lof'] is None:
            return np.zeros(len(data))
        
        # Отбираем числовые признаки
        numeric_features = [f for f in feature_groups.get('numeric', []) 
                          if f in data.columns]
        
        if not numeric_features:
            return np.zeros(len(data))
        
        # Создаем тестовый набор
        X_test = data[numeric_features].fillna(0)
        
        try:
            # Получаем аномальные оценки от LOF
            return -self.ml_models['lof'].decision_function(X_test)
        except Exception as e:
            print(f"Ошибка при использовании Local Outlier Factor: {e}")
            return np.zeros(len(data))
    
    def get_dbscan_scores(self, data: pd.DataFrame, feature_groups: Dict) -> np.ndarray:
        """
        Возвращает оценки аномальности от DBSCAN.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        feature_groups : dict
            Словарь с группами признаков
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        if self.ml_models['dbscan'] is None:
            return np.zeros(len(data))
        
        # Отбираем числовые признаки
        numeric_features = [f for f in feature_groups.get('numeric', []) 
                          if f in data.columns]
        
        if not numeric_features:
            return np.zeros(len(data))
        
        # Создаем тестовый набор
        X_test = data[numeric_features].fillna(0)
        
        try:
            # Предсказываем кластеры
            dbscan_labels = self.ml_models['dbscan'].fit_predict(X_test)
            
            # Точки с меткой -1 являются выбросами (шум)
            dbscan_scores = np.zeros(len(data))
            dbscan_scores[dbscan_labels == -1] = 1.0
            
            return dbscan_scores
        except Exception as e:
            print(f"Ошибка при использовании DBSCAN: {e}")
            return np.zeros(len(data))
