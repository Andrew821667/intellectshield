import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import time
import datetime

class AnomalyDetection:
    """
    Модуль для обнаружения аномалий в сетевом трафике.
    """
    
    def __init__(self):
        """
        Инициализация модуля обнаружения аномалий.
        """
        # Модели машинного обучения
        self.ml_models = {
            'isolation_forest': None,
            'lof': None,
            'dbscan': None
        }
        
        # Пороги для обнаружения аномалий
        self.thresholds = {}
        
        # Текущие профили
        self.profiles = None
        
        # Группы признаков
        self.feature_groups = None
        
        # Множители порогов для разных уровней чувствительности
        self.threshold_multipliers = None
    
    def initialize(self, data, profiles, feature_groups, threshold_multipliers):
        """
        Инициализирует модуль обнаружения аномалий.
        
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
        self.profiles = profiles
        self.feature_groups = feature_groups
        self.threshold_multipliers = threshold_multipliers
        
        # Вычисляем пороги для обнаружения аномалий
        self._calculate_thresholds(data)
    
    def _calculate_thresholds(self, data):
        """
        Вычисляет пороги для обнаружения аномалий.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для вычисления порогов
        """
        # Вычисляем пороги для числовых признаков на основе глобального профиля
        for feature, stats in self.profiles['global'].items():
            if feature in self.feature_groups.get('numeric', []):
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
    
    def train_ml_models(self, data, feature_groups):
        """
        Обучает модели машинного обучения.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения моделей
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
    
    def detect_anomalies(self, data, original_data, sensitivity='medium'):
        """
        Обнаруживает аномалии в данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Предобработанные данные для анализа
        original_data : pandas.DataFrame
            Исходные данные (для добавления результатов)
        sensitivity : str
            Уровень чувствительности ('low', 'medium', 'high')
            
        Returns:
        --------
        pandas.DataFrame
            Результаты обнаружения аномалий
        """
        print(f"Обнаружение аномалий с уровнем чувствительности '{sensitivity}'...")
        
        # Получаем множитель порога для выбранного уровня чувствительности
        threshold_multiplier = self.threshold_multipliers.get(sensitivity, 3.0)
        
        # Создаем копию исходных данных для результатов
        result_df = original_data.copy()
        
        # Вычисляем оценки аномальности для различных типов аномалий
        
        # 1. Статистические аномалии (глобальный профиль)
        statistical_scores = self._detect_statistical_anomalies(data, threshold_multiplier)
        
        # 2. Контекстуальные аномалии (временные и другие профили)
        contextual_scores = self._detect_contextual_anomalies(data, threshold_multiplier)
        
        # 3. Аномалии машинного обучения
        ml_scores = self._detect_ml_anomalies(data)
        
        # 4. Коллективные аномалии (паттерны в последовательности)
        collective_scores = self._detect_collective_anomalies(data)
        
        # Комбинируем оценки аномальности с различными весами
        combined_scores = (
            0.4 * statistical_scores +
            0.3 * contextual_scores +
            0.2 * ml_scores +
            0.1 * collective_scores
        )
        
        # Нормализуем оценки от 0 до 1
        if np.max(combined_scores) > np.min(combined_scores):
            normalized_scores = (combined_scores - np.min(combined_scores)) / (np.max(combined_scores) - np.min(combined_scores))
        else:
            normalized_scores = np.zeros_like(combined_scores)
        
        # Добавляем оценку аномальности в результаты
        result_df['anomaly_score'] = normalized_scores
        
        # Определяем аномалии на основе порога
        # Адаптивный порог: верхние X% рассматриваются как аномалии
        # или все, что выше абсолютного порога
        anomaly_percentile = 95  # Верхние 5%
        percentile_threshold = np.percentile(normalized_scores, anomaly_percentile)
        absolute_threshold = 0.7  # Абсолютный порог
        
        # Выбираем более консервативный порог
        anomaly_threshold = min(percentile_threshold, absolute_threshold)
        
        # Определяем аномалии
        result_df['predicted_anomaly'] = (normalized_scores >= anomaly_threshold).astype(int)
        
        # Определяем типы аномалий
        result_df = self._determine_anomaly_types(result_df, data, 
                                               statistical_scores, contextual_scores, 
                                               ml_scores, collective_scores)
        
        # Выводим результаты
        anomaly_count = result_df['predicted_anomaly'].sum()
        print(f"Обнаружено {anomaly_count} аномалий ({anomaly_count/len(result_df)*100:.2f}%)")
        
        return result_df
    
    def _detect_statistical_anomalies(self, data, threshold_multiplier):
        """
        Обнаруживает статистические аномалии.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # Анализируем только числовые признаки из глобального профиля
        for feature, stats in self.profiles['global'].items():
            if feature in self.feature_groups.get('numeric', []) and feature in data.columns:
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
    
    def _detect_contextual_anomalies(self, data, threshold_multiplier):
        """
        Обнаруживает контекстуальные аномалии.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
        threshold_multiplier : float
            Множитель порога
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # 1. Временной контекст (час дня)
        if 'hour_of_day' in data.columns and 'hour_of_day' in self.profiles['temporal']:
            for i, hour in enumerate(data['hour_of_day']):
                hour_str = str(int(hour))
                
                if hour_str in self.profiles['temporal']['hour_of_day']:
                    hour_profile = self.profiles['temporal']['hour_of_day'][hour_str]
                    
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
        if ('is_working_hours' in data.columns and 
            'working_hours' in self.profiles['contextual']):
            
            for i, is_working in enumerate(data['is_working_hours']):
                context_key = 'working' if is_working == 1 else 'non_working'
                
                if context_key in self.profiles['contextual']['working_hours']:
                    context_profile = self.profiles['contextual']['working_hours'][context_key]
                    
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
        if ('protocol_num' in data.columns and 'dst_port' in data.columns and 
            'protocol_port' in self.profiles['contextual']):
            
            for i in range(len(data)):
                protocol = data.iloc[i]['protocol_num']
                port = data.iloc[i]['dst_port']
                
                # Ключ профиля: 'protocol_X_port_Y'
                context_key = f'protocol_{protocol}_port_{port}'
                
                if context_key in self.profiles['contextual']['protocol_port']:
                    context_profile = self.profiles['contextual']['protocol_port'][context_key]
                    
                    # Проверяем соответствие профилю для каждого признака
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
    
    def _detect_ml_anomalies(self, data):
        """
        Обнаруживает аномалии с помощью моделей машинного обучения.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
        """
        # Инициализируем оценки аномальности нулями
        scores = np.zeros(len(data))
        
        # Отбираем числовые признаки для моделей МО
        numeric_features = [f for f in self.feature_groups.get('numeric', []) 
                          if f in data.columns]
        
        if not numeric_features:
            print("Предупреждение: не найдены числовые признаки для ML-моделей")
            return scores
        
        # Создаем тестовый набор
        X_test = data[numeric_features].fillna(0)
        
        # Если модели не обучены, возвращаем нулевые оценки
        if not all(self.ml_models.values()):
            print("Предупреждение: ML-модели не обучены")
            return scores
        
        # 1. Isolation Forest
        if self.ml_models['isolation_forest'] is not None:
            try:
                # Получаем аномальные оценки от Isolation Forest
                # Отрицательные значения для аномалий, положительные для нормальных точек
                if_scores = -self.ml_models['isolation_forest'].decision_function(X_test)
                
                # Нормализуем оценки от 0 до 1
                if np.max(if_scores) > np.min(if_scores):
                    if_scores = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores))
                else:
                    if_scores = np.zeros_like(if_scores)
                
                # Добавляем в общие оценки с весом
                scores += 0.5 * if_scores
            except Exception as e:
                print(f"Ошибка при использовании Isolation Forest: {e}")
        
        # 2. Local Outlier Factor
        if self.ml_models['lof'] is not None:
            try:
                # Получаем аномальные оценки от LOF
                lof_scores = -self.ml_models['lof'].decision_function(X_test)
                
                # Нормализуем оценки от 0 до 1
                if np.max(lof_scores) > np.min(lof_scores):
                    lof_scores = (lof_scores - np.min(lof_scores)) / (np.max(lof_scores) - np.min(lof_scores))
                else:
                    lof_scores = np.zeros_like(lof_scores)
                
                # Добавляем в общие оценки с весом
                scores += 0.5 * lof_scores
            except Exception as e:
                print(f"Ошибка при использовании Local Outlier Factor: {e}")
        
        # 3. DBSCAN (для обнаружения выбросов)
        if self.ml_models['dbscan'] is not None:
            try:
                # Предсказываем кластеры
                dbscan_labels = self.ml_models['dbscan'].fit_predict(X_test)
                
                # Точки с меткой -1 являются выбросами (шум)
                dbscan_scores = np.zeros(len(data))
                dbscan_scores[dbscan_labels == -1] = 1.0
                
                # Добавляем в общие оценки с весом
                scores += 0.3 * dbscan_scores
            except Exception as e:
                print(f"Ошибка при использовании DBSCAN: {e}")
        
        return scores
    
    def _detect_collective_anomalies(self, data):
        """
        Обнаруживает коллективные аномалии (аномалии в последовательностях).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        numpy.ndarray
            Оценки аномальности
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
        
        # Сортируем данные по времени
        sorted_data = data.sort_values('timestamp')
        
        # Размер окна для анализа последовательностей
        window_size = min(10, len(data) // 10)
        
        # Анализируем каждый доступный признак
        for feature in available_features:
            # Создаем скользящее окно
            for i in range(len(sorted_data) - window_size + 1):
                window = sorted_data[feature].iloc[i:i+window_size].values
                
                # Анализ резких изменений в окне
                if len(window) > 1:
                    # Вычисляем разности между соседними значениями
                    diffs = np.abs(np.diff(window))
                    mean_diff = np.mean(diffs)
                    
                    # Если есть резкие изменения, увеличиваем оценку аномальности
                    for j in range(len(diffs)):
                        if diffs[j] > 3 * mean_diff:
                            scores[i + j] += diffs[j] / (mean_diff + 1e-10) - 3
                            scores[i + j + 1] += diffs[j] / (mean_diff + 1e-10) - 3
        
        return scores
    
    def _determine_anomaly_types(self, result_df, data, statistical_scores, 
                               contextual_scores, ml_scores, collective_scores):
        """
        Определяет типы обнаруженных аномалий.
        
        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты обнаружения аномалий
        data : pandas.DataFrame
            Предобработанные данные
        statistical_scores : numpy.ndarray
            Оценки статистических аномалий
        contextual_scores : numpy.ndarray
            Оценки контекстуальных аномалий
        ml_scores : numpy.ndarray
            Оценки аномалий машинного обучения
        collective_scores : numpy.ndarray
            Оценки коллективных аномалий
            
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
            stat_score = statistical_scores[idx]
            context_score = contextual_scores[idx]
            ml_score = ml_scores[idx]
            collective_score = collective_scores[idx]
            
            # Определяем максимальную оценку
            max_score = max(stat_score, context_score, ml_score, collective_score)
            
            # Определяем тип аномалии
            if max_score == stat_score:
                result_df.at[i, 'anomaly_type'] = self._determine_statistical_anomaly_type(data.iloc[idx])
            elif max_score == context_score:
                result_df.at[i, 'anomaly_type'] = 'Contextual Anomaly'
            elif max_score == ml_score:
                result_df.at[i, 'anomaly_type'] = 'Complex Anomaly'
            elif max_score == collective_score:
                result_df.at[i, 'anomaly_type'] = 'Collective Anomaly'
        
        return result_df
    
    def _determine_statistical_anomaly_type(self, sample):
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
        if ('bytes_per_second' in sample and 
            'bytes_per_second' in self.profiles['global'] and
            sample['bytes_per_second'] > 3 * self.profiles['global']['bytes_per_second'].get('mean', 0)):
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
