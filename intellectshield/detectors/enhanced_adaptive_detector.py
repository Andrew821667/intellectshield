from intellectshield.detectors.base import BaseAnomalyDetector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import joblib
import time
import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedAdaptiveDetector(BaseAnomalyDetector):
    """
    Улучшенный детектор с адаптивными параметрами в зависимости от текущих характеристик трафика.
    
    Ключевые улучшения:
    1. Многоуровневый профиль нормального поведения
    2. Динамическая адаптация порогов
    3. Обнаружение контекстуальных и коллективных аномалий
    4. Анализ временных паттернов и сезонности
    5. Интеграция машинного обучения и статистических методов
    """
    
    def __init__(self, model_dir="models"):
        """
        Инициализация улучшенного адаптивного детектора.
        
        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        """
        super().__init__(model_dir)
        
        # Многоуровневый профиль нормального трафика
        self.baseline_profiles = {
            'global': {},       # Общий профиль для всех данных
            'temporal': {},     # Профили для разных временных интервалов (час дня, день недели)
            'protocol': {},     # Профили для разных протоколов
            'service': {},      # Профили для разных сервисов/портов
            'contextual': {}    # Контекстуальные профили (комбинации условий)
        }
        
        # Параметры временных окон для анализа
        self.time_windows = [1, 5, 15, 60]  # Временные окна в минутах
        
        # Параметры адаптивных порогов
        self.threshold_multipliers = {
            'low': 2.0,     # Низкий порог (высокая чувствительность)
            'medium': 3.0,  # Средний порог
            'high': 4.0     # Высокий порог (низкая чувствительность)
        }
        self.current_sensitivity = 'medium'  # Текущий уровень чувствительности
        
        # Параметры обработки аномалий
        self.min_anomaly_score = 0.0   # Минимальная оценка аномальности
        self.max_anomaly_score = 10.0  # Максимальная оценка аномальности
        
        # Веса для различных типов аномалий
        self.anomaly_weights = {
            'point': 1.0,         # Точечные аномалии (выбросы)
            'contextual': 1.5,    # Контекстуальные аномалии (аномалии в контексте)
            'collective': 2.0     # Коллективные аномалии (группы связанных аномалий)
        }
        
        # Интеграция с ML-моделями
        self.ml_models = {
            'isolation_forest': None,  # Модель Isolation Forest
            'lof': None,               # Модель Local Outlier Factor
            'dbscan': None             # Модель DBSCAN для кластеризации
        }
        
        # Словарь для хранения скалеров признаков
        self.scalers = {}
        
        # Словарь для хранения признаков
        self.feature_groups = {
            'numeric': [],      # Числовые признаки
            'categorical': [],  # Категориальные признаки
            'temporal': [],     # Временные признаки
            'network': [],      # Сетевые признаки (IP, порты)
            'derived': []       # Производные признаки
        }
        
        # История обновлений профилей для отслеживания дрейфа
        self.profile_history = []
        
        # Флаг, указывающий, был ли детектор обучен
        self.is_trained = False

    def _extract_temporal_features(self, df):
        """
        Извлекает временные признаки из данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Исходные данные
            
        Returns:
        --------
        pandas.DataFrame
            Данные с добавленными временными признаками
        """
        result = df.copy()
        
        # Проверяем наличие временной метки
        if 'timestamp' in result.columns:
            # Преобразуем в datetime, если не является datetime
            if result['timestamp'].dtype != 'datetime64[ns]':
                result['timestamp'] = pd.to_datetime(result['timestamp'])
            
            # Извлекаем различные временные признаки
            result['hour_of_day'] = result['timestamp'].dt.hour
            result['day_of_week'] = result['timestamp'].dt.dayofweek
            result['is_weekend'] = result['day_of_week'].isin([5, 6]).astype(int)
            result['is_working_hours'] = ((result['hour_of_day'] >= 9) & 
                                          (result['hour_of_day'] <= 17) & 
                                          ~result['is_weekend']).astype(int)
            result['month'] = result['timestamp'].dt.month
            result['day_of_month'] = result['timestamp'].dt.day
            result['week_of_year'] = result['timestamp'].dt.isocalendar().week
            
            # Добавляем в список временных признаков
            self.feature_groups['temporal'] = [
                'hour_of_day', 'day_of_week', 'is_weekend', 
                'is_working_hours', 'month', 'day_of_month', 'week_of_year'
            ]
        
        return result
    
    def _extract_network_features(self, df):
        """
        Извлекает и обрабатывает сетевые признаки.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Исходные данные
            
        Returns:
        --------
        pandas.DataFrame
            Данные с обработанными сетевыми признаками
        """
        result = df.copy()
        network_features = []
        
        # Обработка IP-адресов
        if 'src_ip' in result.columns:
            if result['src_ip'].dtype == 'object':
                if 'src_ip_hash' not in result.columns:
                    result['src_ip_hash'] = result['src_ip'].apply(lambda x: hash(str(x)) % 10000)
                network_features.append('src_ip_hash')
            
        if 'dst_ip' in result.columns:
            if result['dst_ip'].dtype == 'object':
                if 'dst_ip_hash' not in result.columns:
                    result['dst_ip_hash'] = result['dst_ip'].apply(lambda x: hash(str(x)) % 10000)
                network_features.append('dst_ip_hash')
        
        # Обработка портов
        port_features = ['src_port', 'dst_port']
        for feature in port_features:
            if feature in result.columns:
                # Создаем категории для известных портов
                common_ports = [20, 21, 22, 23, 25, 53, 80, 123, 443, 3389]
                result[f'{feature}_category'] = result[feature].apply(
                    lambda x: x if x in common_ports else (
                        1 if 0 < x < 1024 else (
                            2 if 1024 <= x < 49152 else 3
                        )
                    )
                )
                network_features.append(f'{feature}_category')
                
                # Добавляем признак, указывающий на использование необычных портов
                suspicious_ports = [6667, 31337, 4444, 9001, 1337, 8080]
                result[f'{feature}_suspicious'] = result[feature].isin(suspicious_ports).astype(int)
                network_features.append(f'{feature}_suspicious')
        
        # Обработка протоколов
        if 'protocol' in result.columns:
            if result['protocol'].dtype == 'object':
                protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
                result['protocol_num'] = result['protocol'].map(protocol_map).fillna(0).astype(int)
                network_features.append('protocol_num')
            else:
                network_features.append('protocol')
        
        self.feature_groups['network'] = network_features
        return result
    
    def _create_derived_features(self, df):
        """
        Создает производные признаки на основе существующих.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Исходные данные
            
        Returns:
        --------
        pandas.DataFrame
            Данные с добавленными производными признаками
        """
        result = df.copy()
        derived_features = []
        
        # Признаки для анализа объема трафика
        if 'bytes' in result.columns and 'duration' in result.columns:
            # Скорость передачи данных (байт в секунду)
            result['bytes_per_second'] = result['bytes'] / (result['duration'] + 0.1)  # +0.1 чтобы избежать деления на 0
            derived_features.append('bytes_per_second')
        
        # Признаки для анализа пакетов
        if 'packets' in result.columns and 'duration' in result.columns:
            # Скорость передачи пакетов (пакетов в секунду)
            result['packets_per_second'] = result['packets'] / (result['duration'] + 0.1)
            derived_features.append('packets_per_second')
        
        # Признаки для анализа соотношения входящего и исходящего трафика
        if 'src_bytes' in result.columns and 'dst_bytes' in result.columns:
            # Соотношение входящего и исходящего трафика
            result['bytes_ratio'] = result['src_bytes'] / (result['dst_bytes'] + 1)  # +1 чтобы избежать деления на 0
            derived_features.append('bytes_ratio')
            
            # Общий объем трафика
            result['total_bytes'] = result['src_bytes'] + result['dst_bytes']
            derived_features.append('total_bytes')
        
        # Признаки для анализа частоты соединений
        if 'src_ip_hash' in result.columns and 'timestamp' in result.columns:
            # Группируем по IP-адресу источника и временному окну (5 минут)
            result['time_window'] = pd.to_datetime(result['timestamp']).dt.floor('5T')
            ip_counts = result.groupby(['time_window', 'src_ip_hash']).size().reset_index(name='connection_count')
            result = pd.merge(result, ip_counts, on=['time_window', 'src_ip_hash'], how='left')
            derived_features.append('connection_count')
            
            # Можно удалить временное окно, если оно больше не нужно
            result.drop('time_window', axis=1, inplace=True)
        
        # Признаки для анализа ошибок
        error_features = ['serror_rate', 'rerror_rate', 'srv_serror_rate', 'srv_rerror_rate']
        error_features_present = [f for f in error_features if f in result.columns]
        
        if error_features_present:
            # Общая частота ошибок
            result['total_error_rate'] = result[error_features_present].sum(axis=1)
            derived_features.append('total_error_rate')
        
        # Признаки для анализа разнообразия сервисов
        diversity_features = ['same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate']
        diversity_features_present = [f for f in diversity_features if f in result.columns]
        
        if diversity_features_present:
            # Средняя оценка разнообразия сервисов
            result['service_diversity'] = result[diversity_features_present].mean(axis=1)
            derived_features.append('service_diversity')
        
        # Признаки для обнаружения сканирования портов
        if 'packets' in result.columns and 'duration' in result.columns:
            # Сканирование портов обычно имеет малое количество пакетов и короткую продолжительность
            result['is_port_scan_like'] = ((result['packets'] <= 3) & 
                                          (result['duration'] < 0.5)).astype(int)
            derived_features.append('is_port_scan_like')
        
        # Признаки для обнаружения DoS-атак
        if 'connection_count' in result.columns:
            # DoS-атаки обычно имеют большое количество соединений от одного источника
            connection_count_threshold = result['connection_count'].quantile(0.95)
            result['is_dos_like'] = (result['connection_count'] > connection_count_threshold).astype(int)
            derived_features.append('is_dos_like')
        
        self.feature_groups['derived'] = derived_features
        return result
    
    def _categorize_features(self, df):
        """
        Категоризирует признаки на числовые и категориальные.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для категоризации признаков
        """
        # Сбрасываем существующие категории
        self.feature_groups['numeric'] = []
        self.feature_groups['categorical'] = []
        
        # Исключаем метки и технические колонки
        exclude_cols = ['label', 'is_anomaly', 'predicted_anomaly', 'anomaly_score', 'timestamp']
        
        # Категоризируем признаки
        for col in df.columns:
            if col not in exclude_cols:
                if df[col].dtype.kind in 'ifc':  # integer, float, complex
                    self.feature_groups['numeric'].append(col)
                else:
                    self.feature_groups['categorical'].append(col)

    def preprocess_data(self, data, train=False):
        """
        Комплексная предобработка данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для предобработки
        train : bool
            Флаг режима обучения (True) или предсказания (False)
            
        Returns:
        --------
        pandas.DataFrame
            Предобработанные данные
        """
        # Создаем копию данных для обработки
        df = data.copy()
        
        # 1. Извлечение временных признаков
        df = self._extract_temporal_features(df)
        
        # 2. Обработка сетевых признаков
        df = self._extract_network_features(df)
        
        # 3. Создание производных признаков
        df = self._create_derived_features(df)
        
        # 4. Категоризация признаков
        self._categorize_features(df)
        
        # 5. Масштабирование числовых признаков
        if train:
            # В режиме обучения создаем новые скалеры
            for feature in self.feature_groups['numeric']:
                if feature in df.columns:
                    self.scalers[feature] = StandardScaler()
                    # Проверяем наличие NaN и inf
                    feature_data = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    if not feature_data.empty:
                        try:
                            self.scalers[feature].fit(feature_data.values.reshape(-1, 1))
                        except Exception as e:
                            print(f"Ошибка при обучении скалера для {feature}: {e}")
        
        # Применяем масштабирование (если возможно)
        for feature in self.feature_groups['numeric']:
            if feature in df.columns and feature in self.scalers:
                try:
                    # Заменяем inf и NaN на 0 перед масштабированием
                    feature_data = df[feature].replace([np.inf, -np.inf], np.nan).fillna(0)
                    df[feature] = self.scalers[feature].transform(feature_data.values.reshape(-1, 1))
                except Exception as e:
                    print(f"Ошибка при масштабировании признака {feature}: {e}")
        
        return df
    
    def _update_global_profile(self, df):
        """
        Обновляет глобальный профиль на основе данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обновления профиля
        """
        # Обновляем профиль для числовых признаков
        for feature in self.feature_groups['numeric']:
            if feature in df.columns:
                # Получаем данные признака без выбросов
                feature_data = df[feature].replace([np.inf, -np.inf], np.nan).dropna()
                
                if not feature_data.empty:
                    # Вычисляем статистики
                    self.baseline_profiles['global'][feature] = {
                        'mean': feature_data.mean(),
                        'std': feature_data.std() if feature_data.std() > 0 else 1.0,  # Избегаем деления на 0
                        'min': feature_data.min(),
                        'max': feature_data.max(),
                        'q1': feature_data.quantile(0.25),
                        'median': feature_data.quantile(0.5),
                        'q3': feature_data.quantile(0.75),
                        'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25),
                        'skew': feature_data.skew(),
                        'kurtosis': feature_data.kurtosis(),
                        'hist': np.histogram(feature_data, bins=10)[0].tolist()
                    }
    
    def _update_temporal_profiles(self, df):
        """
        Обновляет временные профили на основе данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обновления профилей
        """
        if 'hour_of_day' in df.columns:
            # Инициализируем профили по часам дня, если их еще нет
            if 'hourly' not in self.baseline_profiles['temporal']:
                self.baseline_profiles['temporal']['hourly'] = {}
            
            # Обновляем профили для каждого часа дня
            for hour in range(24):
                hour_data = df[df['hour_of_day'] == hour]
                if not hour_data.empty:
                    self.baseline_profiles['temporal']['hourly'][hour] = {}
                    
                    for feature in self.feature_groups['numeric']:
                        if feature in df.columns:
                            feature_data = hour_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            if not feature_data.empty:
                                self.baseline_profiles['temporal']['hourly'][hour][feature] = {
                                    'mean': feature_data.mean(),
                                    'std': feature_data.std() if feature_data.std() > 0 else 1.0,
                                    'median': feature_data.quantile(0.5),
                                    'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25)
                                }
        
        if 'day_of_week' in df.columns:
            # Инициализируем профили по дням недели, если их еще нет
            if 'daily' not in self.baseline_profiles['temporal']:
                self.baseline_profiles['temporal']['daily'] = {}
            
            # Обновляем профили для каждого дня недели
            for day in range(7):
                day_data = df[df['day_of_week'] == day]
                if not day_data.empty:
                    self.baseline_profiles['temporal']['daily'][day] = {}
                    
                    for feature in self.feature_groups['numeric']:
                        if feature in df.columns:
                            feature_data = day_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            if not feature_data.empty:
                                self.baseline_profiles['temporal']['daily'][day][feature] = {
                                    'mean': feature_data.mean(),
                                    'std': feature_data.std() if feature_data.std() > 0 else 1.0,
                                    'median': feature_data.quantile(0.5),
                                    'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25)
                                }
    
    def train(self, data, **kwargs):
        """
        Обучение улучшенного адаптивного детектора.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения детектора
        **kwargs : dict
            Дополнительные параметры
            
        Returns:
        --------
        self
            Обученный детектор
        """
        print("Начало обучения улучшенного адаптивного детектора...")
        start_time = time.time()
        
        # Разделяем данные на нормальные и аномальные
        if 'is_anomaly' in data.columns:
            normal_data = data[data['is_anomaly'] == 0]
            anomaly_data = data[data['is_anomaly'] == 1]
            print(f"Данные содержат {len(normal_data)} нормальных и {len(anomaly_data)} аномальных образцов")
        else:
            normal_data = data  # Предполагаем, что все данные нормальные, если нет меток
            anomaly_data = pd.DataFrame()
            print(f"Данные не содержат меток аномалий. Предполагаем, что все {len(data)} образцов нормальные.")
        
        # Предобработка данных
        print("Предобработка данных...")
        preprocessed_data = self.preprocess_data(normal_data, train=True)
        
        # Обновление профилей
        print("Обновление профилей...")
        self._update_global_profile(preprocessed_data)
        self._update_temporal_profiles(preprocessed_data)
        
        # Обучение ML-моделей
        print("Обучение ML-моделей...")
        self._train_ml_models(preprocessed_data)
        
        # Отмечаем, что детектор обучен
        self.is_trained = True
        
        # Завершение обучения
        training_time = time.time() - start_time
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return self
        
    def _detect_point_anomalies(self, df):
        """
        Обнаружение точечных аномалий (выбросов в отдельных признаках).
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.Series
            Оценки аномальности для каждой записи
        """
        # Инициализируем оценки аномалий
        anomaly_scores = pd.Series(0.0, index=df.index)
        
        # Для каждого числового признака
        for feature in self.feature_groups['numeric']:
            if feature in df.columns and feature in self.baseline_profiles['global']:
                # Получаем профиль признака
                profile = self.baseline_profiles['global'][feature]
                
                # Вычисляем Z-score (стандартизованное отклонение)
                if profile['std'] > 0:
                    z_scores = np.abs(df[feature] - profile['mean']) / profile['std']
                    
                    # Нормализуем Z-score с учетом текущей чувствительности
                    threshold_multiplier = self.threshold_multipliers[self.current_sensitivity]
                    feature_scores = np.maximum(0, z_scores - threshold_multiplier) / threshold_multiplier
                    
                    # Добавляем к общей оценке аномальности
                    anomaly_scores += feature_scores
        
        # Нормализуем общую оценку по количеству признаков
        if len(self.feature_groups['numeric']) > 0:
            anomaly_scores = anomaly_scores / len(self.feature_groups['numeric'])
            
        return anomaly_scores
        
    def predict(self, data):
        """
        Обнаружение аномалий в данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Результаты анализа с оценками аномальности и предсказаниями
        """
        if not self.is_trained:
            raise ValueError("Детектор не обучен. Сначала вызовите метод train().")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=False)
        
        # Обнаружение точечных аномалий (статистический подход)
        point_anomaly_scores = self._detect_point_anomalies(preprocessed_data)
        
        # Обнаружение аномалий с помощью ML-моделей
        ml_anomaly_scores = pd.Series(0.0, index=preprocessed_data.index)
        
        # Используем Isolation Forest, если он обучен
        if self.ml_models['isolation_forest'] is not None:
            try:
                # Выбираем числовые признаки для анализа
                numeric_features = [f for f in self.feature_groups['numeric'] if f in preprocessed_data.columns]
                X = preprocessed_data[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Получаем оценки аномальности от Isolation Forest
                # Isolation Forest возвращает отрицательные оценки для аномалий
                if_scores = -self.ml_models['isolation_forest'].decision_function(X)
                
                # Нормализуем оценки в диапазон [0, 1]
                if_scores = (if_scores - np.min(if_scores)) / (np.max(if_scores) - np.min(if_scores) + 1e-10)
                
                # Добавляем к общим ML-оценкам
                ml_anomaly_scores += if_scores
            except Exception as e:
                print(f"Ошибка при использовании Isolation Forest: {e}")
        
        # Используем Local Outlier Factor, если он обучен
        if self.ml_models['lof'] is not None:
            try:
                # Выбираем числовые признаки для анализа
                numeric_features = [f for f in self.feature_groups['numeric'] if f in preprocessed_data.columns]
                X = preprocessed_data[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                # Получаем оценки аномальности от LOF
                # LOF возвращает отрицательные оценки для аномалий в режиме novelty=True
                lof_scores = -self.ml_models['lof'].decision_function(X)
                
                # Нормализуем оценки в диапазон [0, 1]
                lof_scores = (lof_scores - np.min(lof_scores)) / (np.max(lof_scores) - np.min(lof_scores) + 1e-10)
                
                # Добавляем к общим ML-оценкам
                ml_anomaly_scores += lof_scores
            except Exception as e:
                print(f"Ошибка при использовании Local Outlier Factor: {e}")
        
        # Нормализуем ML-оценки
        if self.ml_models['isolation_forest'] is not None and self.ml_models['lof'] is not None:
            ml_anomaly_scores = ml_anomaly_scores / 2.0
        
        # Объединяем оценки от разных подходов
        combined_scores = (
            self.anomaly_weights['point'] * point_anomaly_scores + 
            self.anomaly_weights['collective'] * ml_anomaly_scores
        ) / (self.anomaly_weights['point'] + self.anomaly_weights['collective'])
        
        # Создаем результирующий датафрейм
        result_df = data.copy()
        result_df['anomaly_score'] = combined_scores
        
        # Определяем порог аномальности (верхние 5%)
        threshold = np.percentile(combined_scores, 95)
        result_df['predicted_anomaly'] = (combined_scores >= threshold).astype(int)
        
        # Определяем типы аномалий
        result_df['anomaly_type'] = 'normal'
        
        # Точечные аномалии (статистические выбросы)
        point_mask = (result_df['predicted_anomaly'] == 1) & (point_anomaly_scores > threshold)
        result_df.loc[point_mask, 'anomaly_type'] = 'point'
        
        # Коллективные аномалии (обнаруженные ML-моделями)
        collective_mask = (result_df['predicted_anomaly'] == 1) & (ml_anomaly_scores > threshold)
        result_df.loc[collective_mask, 'anomaly_type'] = 'collective'
        
        # Контекстуальные аномалии (точечные аномалии в определенном контексте)
        # Пример: аномалии в нерабочее время
        if 'is_working_hours' in result_df.columns:
            context_mask = (result_df['predicted_anomaly'] == 1) & (result_df['is_working_hours'] == 0)
            result_df.loc[context_mask, 'anomaly_type'] = 'contextual'
        
        return result_df

    def _update_protocol_profiles(self, df):
        """
        Обновляет профили протоколов на основе данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обновления профилей
        """
        protocol_feature = None
        if 'protocol_num' in df.columns:
            protocol_feature = 'protocol_num'
        elif 'protocol' in df.columns:
            protocol_feature = 'protocol'
        
        if protocol_feature:
            # Получаем уникальные протоколы
            protocols = df[protocol_feature].unique()
            
            for protocol in protocols:
                protocol_data = df[df[protocol_feature] == protocol]
                if not protocol_data.empty:
                    self.baseline_profiles['protocol'][protocol] = {}
                    
                    for feature in self.feature_groups['numeric']:
                        if feature in df.columns and feature != protocol_feature:
                            feature_data = protocol_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            if not feature_data.empty:
                                self.baseline_profiles['protocol'][protocol][feature] = {
                                    'mean': feature_data.mean(),
                                    'std': feature_data.std() if feature_data.std() > 0 else 1.0,
                                    'median': feature_data.quantile(0.5),
                                    'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25)
                                }
    
    def _update_service_profiles(self, df):
        """
        Обновляет профили сервисов/портов на основе данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обновления профилей
        """
        if 'dst_port' in df.columns:
            # Получаем наиболее популярные порты (топ 10)
            top_ports = df['dst_port'].value_counts().head(10).index.tolist()
            
            for port in top_ports:
                port_data = df[df['dst_port'] == port]
                if not port_data.empty:
                    self.baseline_profiles['service'][port] = {}
                    
                    for feature in self.feature_groups['numeric']:
                        if feature in df.columns and feature != 'dst_port':
                            feature_data = port_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            if not feature_data.empty:
                                self.baseline_profiles['service'][port][feature] = {
                                    'mean': feature_data.mean(),
                                    'std': feature_data.std() if feature_data.std() > 0 else 1.0,
                                    'median': feature_data.quantile(0.5),
                                    'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25)
                                }
    
    def _update_contextual_profiles(self, df):
        """
        Обновляет контекстуальные профили на основе данных.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обновления профилей
        """
        # Создаем контексты на основе комбинаций условий
        contexts = []
        
        # Контекст: рабочее время (рабочие часы в будние дни)
        if 'is_working_hours' in df.columns:
            contexts.append(('working_hours', df['is_working_hours'] == 1))
        
        # Контекст: нерабочее время (ночь, выходные)
        if 'is_working_hours' in df.columns:
            contexts.append(('non_working_hours', df['is_working_hours'] == 0))
        
        # Контекст: веб-трафик (порты 80, 443)
        if 'dst_port' in df.columns:
            contexts.append(('web_traffic', df['dst_port'].isin([80, 443])))
        
        # Контекст: SSH/администрирование (порт 22)
        if 'dst_port' in df.columns:
            contexts.append(('admin_traffic', df['dst_port'] == 22))
        
        # Обновляем профили для каждого контекста
        for context_name, context_mask in contexts:
            context_data = df[context_mask]
            if not context_data.empty:
                self.baseline_profiles['contextual'][context_name] = {}
                
                for feature in self.feature_groups['numeric']:
                    if feature in df.columns:
                        feature_data = context_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                        if not feature_data.empty:
                            self.baseline_profiles['contextual'][context_name][feature] = {
                                'mean': feature_data.mean(),
                                'std': feature_data.std() if feature_data.std() > 0 else 1.0,
                                'median': feature_data.quantile(0.5),
                                'iqr': feature_data.quantile(0.75) - feature_data.quantile(0.25)
                            }
    
    def _train_ml_models(self, df):
        """
        Обучает ML-модели для обнаружения аномалий.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Данные для обучения моделей
        """
        # Выбираем числовые признаки для обучения моделей
        numeric_features = [f for f in self.feature_groups['numeric'] if f in df.columns]
        if not numeric_features:
            print("Предупреждение: нет числовых признаков для обучения ML-моделей")
            return
        
        # Получаем данные без выбросов
        X = df[numeric_features].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Обучаем Isolation Forest
        try:
            self.ml_models['isolation_forest'] = IsolationForest(
                n_estimators=100, 
                contamination=0.05,  # Ожидаемая доля аномалий
                random_state=42
            )
            self.ml_models['isolation_forest'].fit(X)
        except Exception as e:
            print(f"Ошибка при обучении Isolation Forest: {e}")
        
        # Обучаем Local Outlier Factor
        try:
            self.ml_models['lof'] = LocalOutlierFactor(
                n_neighbors=20,
                contamination=0.05,
                novelty=True  # Для возможности использовать predict
            )
            self.ml_models['lof'].fit(X)
        except Exception as e:
            print(f"Ошибка при обучении Local Outlier Factor: {e}")
        
        # Обучаем DBSCAN для кластеризации
        try:
            self.ml_models['dbscan'] = DBSCAN(
                eps=0.5,
                min_samples=5
            )
            self.ml_models['dbscan'].fit(X)
        except Exception as e:
            print(f"Ошибка при обучении DBSCAN: {e}")
            
    def visualize_anomalies(self, result_df, max_samples=1000):
        """
        Визуализация обнаруженных аномалий.
        
        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты обнаружения аномалий (выход метода predict)
        max_samples : int
            Максимальное количество образцов для визуализации
        """
        # Проверка наличия аномалий
        if 'predicted_anomaly' not in result_df.columns or 'anomaly_score' not in result_df.columns:
            print("Ошибка: входные данные не содержат результатов обнаружения аномалий")
            return
        
        anomaly_count = result_df['predicted_anomaly'].sum()
        if anomaly_count == 0:
            print("Аномалии не обнаружены. Визуализация невозможна.")
            return
        
        # Ограничиваем количество образцов для визуализации
        if len(result_df) > max_samples:
            # Стратифицированная выборка для сохранения пропорции аномалий
            normal_samples = min(int(max_samples * 0.7), (result_df['predicted_anomaly'] == 0).sum())
            anomaly_samples = min(int(max_samples * 0.3), anomaly_count)
            
            normal_df = result_df[result_df['predicted_anomaly'] == 0].sample(normal_samples, random_state=42)
            anomaly_df = result_df[result_df['predicted_anomaly'] == 1].sample(anomaly_samples, random_state=42)
            
            vis_df = pd.concat([normal_df, anomaly_df])
        else:
            vis_df = result_df.copy()
        
        # Создаем фигуру для визуализации
        plt.figure(figsize=(16, 12))
        
        # 1. Распределение аномальных оценок
        plt.subplot(2, 2, 1)
        plt.hist([
            vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
            vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score']
        ], bins=50, label=['Нормальные', 'Аномальные'])
        plt.xlabel('Оценка аномальности')
        plt.ylabel('Количество')
        plt.title('Распределение оценок аномальности')
        plt.legend()
        plt.grid(True)
        
        # 2. Типы аномалий (если есть)
        plt.subplot(2, 2, 2)
        if 'anomaly_type' in vis_df.columns:
            anomaly_types = vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
            anomaly_types.plot(kind='bar', color='coral')
            plt.title('Типы обнаруженных аномалий')
            plt.xlabel('Тип аномалии')
            plt.ylabel('Количество')
            plt.xticks(rotation=45)
            plt.grid(True, axis='y')
        else:
            plt.text(0.5, 0.5, 'Информация о типах аномалий отсутствует',
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # 3. Временной ряд аномальных оценок (если есть временная метка)
        plt.subplot(2, 2, 3)
        if 'timestamp' in vis_df.columns:
            plt.scatter(
                vis_df[vis_df['predicted_anomaly'] == 0]['timestamp'], 
                vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
                alpha=0.5, label='Нормальные', s=20, color='blue'
            )
            plt.scatter(
                vis_df[vis_df['predicted_anomaly'] == 1]['timestamp'], 
                vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score'],
                alpha=0.7, label='Аномальные', s=30, color='red'
            )
            plt.title('Временной ряд аномальных оценок')
            plt.xlabel('Время')
            plt.ylabel('Оценка аномальности')
            plt.legend()
            plt.grid(True)
        else:
            # Если нет временной метки, визуализируем по индексу
            plt.scatter(
                range(len(vis_df[vis_df['predicted_anomaly'] == 0])),
                vis_df[vis_df['predicted_anomaly'] == 0]['anomaly_score'],
                alpha=0.5, label='Нормальные', s=20, color='blue'
            )
            plt.scatter(
                range(len(vis_df[vis_df['predicted_anomaly'] == 0]), len(vis_df)),
                vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_score'],
                alpha=0.7, label='Аномальные', s=30, color='red'
            )
            plt.title('Оценки аномальности по индексу')
            plt.xlabel('Индекс')
            plt.ylabel('Оценка аномальности')
            plt.legend()
            plt.grid(True)
        
        # 4. Матрица ошибок (если есть истинные метки)
        plt.subplot(2, 2, 4)
        if 'is_anomaly' in vis_df.columns:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(vis_df['is_anomaly'], vis_df['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Нормальные', 'Аномальные'],
                        yticklabels=['Нормальные', 'Аномальные'])
            plt.title('Матрица ошибок')
            plt.xlabel('Предсказание')
            plt.ylabel('Истинное значение')
            
            # Рассчитываем метрики качества
            tp = cm[1, 1]
            tn = cm[0, 0]
            fp = cm[0, 1]
            fn = cm[1, 0]
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_text = (
                f"Accuracy: {accuracy:.4f}\n"
                f"Precision: {precision:.4f}\n"
                f"Recall: {recall:.4f}\n"
                f"F1 Score: {f1:.4f}"
            )
            
            plt.text(1.5, 0.5, metrics_text, fontsize=11, 
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            plt.text(0.5, 0.5, 'Истинные метки аномалий отсутствуют',
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Выводим сводную информацию
        print(f"Всего записей: {len(result_df)}")
        print(f"Обнаруженных аномалий: {anomaly_count} ({anomaly_count/len(result_df)*100:.2f}%)")
        
        if 'anomaly_type' in result_df.columns:
        print("\nРаспределение типов аномалий:")
                    type_counts = result_df[result_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
                    for anomaly_type, count in type_counts.items():
                        print(f"  {anomaly_type}: {count} ({count/anomaly_count*100:.2f}%)")
                
        if 'is_anomaly' in result_df.columns:
            # Вычисляем метрики
            from sklearn.metrics import classification_report
            print("\nМетрики качества обнаружения:")
            print(classification_report(result_df['is_anomaly'], result_df['predicted_anomaly']))
