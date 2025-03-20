import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureExtractor:
    """
    Модуль для извлечения и обработки признаков из данных сетевого трафика.
    """
    
    def __init__(self):
        """
        Инициализация экстрактора признаков.
        """
        # Словарь для хранения признаков
        self.feature_groups = {
            'numeric': [],      # Числовые признаки
            'categorical': [],  # Категориальные признаки
            'temporal': [],     # Временные признаки
            'network': [],      # Сетевые признаки (IP, порты)
            'derived': []       # Производные признаки
        }
        
        # Словарь для хранения скалеров признаков
        self.scalers = {}
    
    def process(self, data, train=False):
        """
        Комплексная обработка данных и извлечение признаков.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Исходные данные
        train : bool
            Флаг режима обучения
            
        Returns:
        --------
        pandas.DataFrame
            Обработанные данные с извлеченными признаками
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
