"""
Улучшенная версия детектора аномалий на основе Isolation Forest.

Эта версия включает оптимизации для повышения показателя Recall
при сохранении высокой Precision.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
import time
import os
import pickle

from intellectshield.detectors.anomaly.isolation_forest import IsolationForestDetector

class EnhancedIsolationForestDetector(IsolationForestDetector):
    """
    Улучшенная версия детектора аномалий Isolation Forest.
    
    Включает:
    - Автоматический подбор параметров
    - Предварительную обработку с использованием PCA
    - Динамический порог для определения аномалий
    """
    
    def __init__(self, model_dir="models", n_estimators=100, 
                 max_features=1.0, contamination='auto', 
                 use_pca=True, pca_components=0.95,
                 dynamic_threshold=True):
        """
        Инициализация улучшенного детектора Isolation Forest.
        
        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        n_estimators : int
            Количество деревьев
        max_features : float или int
            Максимальное количество признаков для каждого дерева
        contamination : float или 'auto'
            Ожидаемая доля аномалий в данных
        use_pca : bool
            Использовать ли PCA для предобработки
        pca_components : float или int
            Количество компонент PCA
        dynamic_threshold : bool
            Использовать ли динамический порог для определения аномалий
        """
        super().__init__(model_dir)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.contamination = contamination
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.dynamic_threshold = dynamic_threshold
        self.pca = None
        self.scaler = None
        self.threshold = None
        self.calibration_scores = None
        self.feature_names = None
    
    def preprocess_data(self, data, train=False):
        """
        Расширенная предобработка данных для Isolation Forest.
        """
        # Базовая предобработка
        df = super().preprocess_data(data, train)
        
        # Сохраняем is_anomaly перед удалением
        is_anomaly = None
        if 'is_anomaly' in df.columns:
            is_anomaly = df['is_anomaly'].copy()
            df = df.drop('is_anomaly', axis=1)
        
        # Сохраняем имена признаков при первом обучении
        if train and self.feature_names is None:
            self.feature_names = df.columns.tolist()
        
        # Проверка, содержит ли набор данных все необходимые признаки
        if not train and self.feature_names is not None:
            # Если обнаружены новые признаки, оставляем только те, которые были при обучении
            df = df[df.columns.intersection(self.feature_names)]
            # Если каких-то признаков не хватает, заполняем их нулями
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            # Сортируем колонки в том же порядке, как при обучении
            df = df[self.feature_names]
        
        # Дополнительная нормализация
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_features) > 0:
            if train or self.scaler is None:
                self.scaler = StandardScaler()
                df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
            else:
                df[numeric_features] = self.scaler.transform(df[numeric_features])
        
        # Применение PCA для уменьшения размерности
        if self.use_pca:
            if train or self.pca is None:
                self.pca = PCA(n_components=self.pca_components)
                pca_features = self.pca.fit_transform(df)
            else:
                pca_features = self.pca.transform(df)
            
            # Создаем новый DataFrame с PCA компонентами
            pca_df = pd.DataFrame(
                pca_features, 
                columns=[f'pca_{i}' for i in range(pca_features.shape[1])]
            )
            
            if is_anomaly is not None:
                pca_df['is_anomaly'] = is_anomaly
            return pca_df
        else:
            if is_anomaly is not None:
                df['is_anomaly'] = is_anomaly
            return df
    
    def train(self, data, auto_tune=True, cv_folds=3):
        """
        Обучение модели с автоматической настройкой параметров.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
        auto_tune : bool
            Автоматическая настройка параметров
        cv_folds : int
            Количество фолдов для кросс-валидации
        """
        print("Начало обучения улучшенной модели Isolation Forest...")
        start_time = time.time()
        
        # Предобработка данных
        df = self.preprocess_data(data, train=True)
        
        # Разделяем признаки и метку
        X = df.drop('is_anomaly', axis=1, errors='ignore')
        
        if auto_tune and 'is_anomaly' in df.columns:
            y = df['is_anomaly']
            
            # Создаем кастомный скорер для Isolation Forest
            def isolation_forest_scorer(estimator, X, y):
                # Предсказание и преобразование в метки (1: аномалия, 0: норма)
                y_pred = (estimator.decision_function(X) < 0).astype(int)
                return f1_score(y, y_pred)
            
            # Определяем параметры для поиска
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_features': [0.5, 0.75, 1.0],
                'contamination': [0.01, 0.05, 0.1, 0.2, 'auto']
            }
            
            # Создаем базовую модель
            base_model = IsolationForest(random_state=42)
            
            # Создаем объект GridSearchCV с кастомным скорером
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds,
                scoring=isolation_forest_scorer,
                n_jobs=-1  # Используем все доступные ядра
            )
            
            # Выполняем поиск по сетке
            print("Запуск автоматической настройки параметров...")
            grid_search.fit(X, y)
            
            # Получаем лучшие параметры
            best_params = grid_search.best_params_
            print(f"Лучшие параметры: {best_params}")
            
            # Обновляем параметры
            self.n_estimators = best_params['n_estimators']
            self.max_features = best_params['max_features']
            self.contamination = best_params['contamination']
        
        # Создаем и обучаем модель с лучшими параметрами
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            contamination=self.contamination,
            random_state=42
        )
        
        self.model.fit(X)
        
        # Сохраняем распределение аномальных скоров для калибровки
        self.calibration_scores = self.model.decision_function(X)
        
        # Если используется динамический порог и есть метки аномалий
        if self.dynamic_threshold and 'is_anomaly' in df.columns:
            self._calibrate_threshold(X, df['is_anomaly'])
        
        training_time = time.time() - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        # Дополнительная информация
        if self.use_pca and self.pca is not None:
            print(f"Сокращение размерности с {len(self.feature_names)} до {X.shape[1]} признаков")
            print(f"Объясненная дисперсия: {np.sum(self.pca.explained_variance_ratio_):.4f}")
        
        return self.model
    
    def _calibrate_threshold(self, X, y_true):
        """
        Калибровка порога аномальности для оптимизации F1 score.
        """
        scores = self.model.decision_function(X)
        
        # Перебираем различные пороги и выбираем тот, который дает лучший F1
        thresholds = np.linspace(np.min(scores), np.max(scores), 100)
        best_f1 = 0
        best_threshold = 0
        
        for threshold in thresholds:
            y_pred = (scores < threshold).astype(int)
            prec = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Калибровка порога: {best_threshold:.4f}, F1: {best_f1:.4f}")
    
    def predict(self, data):
        """
        Предсказание аномалий с динамическим порогом.
        """
        # Сохраняем оригинальные данные
        result = data.copy()
        
        if self.model is None:
            print("Ошибка: модель не обучена")
            return result
        
        # Предобработка данных
        df = self.preprocess_data(data, train=False)
        
        # Убираем целевую переменную для предсказания
        X = df.drop('is_anomaly', axis=1, errors='ignore')
        
        # Получаем аномальные скоры (отрицательные значения = аномалии)
        anomaly_scores = self.model.decision_function(X)
        
        # Добавляем скоры в результат
        result['anomaly_score'] = -anomaly_scores  # Инвертируем для понятности (больше = аномальнее)
        
        # Определяем аномалии и добавляем обе колонки для совместимости
        if self.dynamic_threshold and self.threshold is not None:
            # Используем калиброванный порог
            predicted_anomalies = (anomaly_scores < self.threshold).astype(int)
        else:
            # Используем встроенный метод
            predictions = self.model.predict(X)
            # Преобразуем -1/1 в 1/0 (1 = аномалия, 0 = норма)
            predicted_anomalies = (predictions == -1).astype(int)
        
        # Добавляем предсказания в результат
        result['predicted_anomaly'] = predicted_anomalies
        
        return result
    
    def save_model(self, filepath):
        """
        Сохранение модели и дополнительных компонентов в файл.
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'use_pca': self.use_pca,
            'dynamic_threshold': self.dynamic_threshold,
            'n_estimators': self.n_estimators,
            'max_features': self.max_features,
            'contamination': self.contamination,
            'timestamp': pd.Timestamp.now()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Модель сохранена в {filepath}")
    
    def load_model(self, filepath):
        """
        Загрузка модели и дополнительных компонентов из файла.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.threshold = model_data['threshold']
        self.feature_names = model_data['feature_names']
        self.use_pca = model_data['use_pca']
        self.dynamic_threshold = model_data['dynamic_threshold']
        self.n_estimators = model_data['n_estimators']
        self.max_features = model_data['max_features']
        self.contamination = model_data['contamination']
        
        print(f"Модель загружена из {filepath}")
        print(f"Модель была сохранена: {model_data['timestamp']}")
