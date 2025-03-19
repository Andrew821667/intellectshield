import numpy as np
import pandas as pd
from scipy import stats
import joblib
import time
import datetime
import json

class ProfileManager:
    """
    Модуль для управления профилями нормального поведения трафика.
    """
    
    def __init__(self):
        """
        Инициализация менеджера профилей.
        """
        # Многоуровневый профиль нормального трафика
        self.profiles = {
            'global': {},       # Общий профиль для всех данных
            'temporal': {},     # Профили для разных временных интервалов (час дня, день недели)
            'protocol': {},     # Профили для разных протоколов
            'service': {},      # Профили для разных сервисов/портов
            'contextual': {}    # Контекстуальные профили (комбинации условий)
        }
        
    def update_profiles(self, data, feature_groups):
        """
        Обновляет профили на основе новых данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профилей
        feature_groups : dict
            Словарь с группами признаков
        """
        print("Обновление профилей поведения...")
        
        # 1. Обновление глобального профиля
        self._update_global_profile(data, feature_groups)
        
        # 2. Обновление временных профилей
        self._update_temporal_profiles(data, feature_groups)
        
        # 3. Обновление профилей протоколов
        self._update_protocol_profiles(data, feature_groups)
        
        # 4. Обновление профилей сервисов/портов
        self._update_service_profiles(data, feature_groups)
        
        # 5. Обновление контекстуальных профилей
        self._update_contextual_profiles(data, feature_groups)
        
        print(f"Обновление профилей завершено. Создано {self._count_profiles()} профилей.")
    
    def _update_global_profile(self, data, feature_groups):
        """
        Обновляет глобальный профиль на основе всех данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профиля
        feature_groups : dict
            Словарь с группами признаков
        """
        # Обрабатываем числовые признаки
        numeric_features = feature_groups.get('numeric', [])
        for feature in numeric_features:
            if feature in data.columns:
                # Очищаем данные от выбросов для более надежной статистики
                feature_data = data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                
                if not feature_data.empty:
                    # Вычисляем основные статистики
                    mean_val = feature_data.mean()
                    std_val = feature_data.std()
                    median_val = feature_data.median()
                    q1_val = feature_data.quantile(0.25)
                    q3_val = feature_data.quantile(0.75)
                    iqr_val = q3_val - q1_val
                    min_val = feature_data.min()
                    max_val = feature_data.max()
                    
                    # Обновляем профиль
                    self.profiles['global'][feature] = {
                        'mean': mean_val,
                        'std': std_val,
                        'median': median_val,
                        'q1': q1_val,
                        'q3': q3_val,
                        'iqr': iqr_val,
                        'min': min_val,
                        'max': max_val,
                        'last_updated': datetime.datetime.now().isoformat()
                    }
        
        # Обрабатываем категориальные признаки
        categorical_features = feature_groups.get('categorical', [])
        for feature in categorical_features:
            if feature in data.columns:
                # Вычисляем распределение категорий
                value_counts = data[feature].value_counts(normalize=True).to_dict()
                
                # Обновляем профиль
                self.profiles['global'][feature] = {
                    'distribution': value_counts,
                    'last_updated': datetime.datetime.now().isoformat()
                }
    
    def _update_temporal_profiles(self, data, feature_groups):
        """
        Обновляет временные профили.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профилей
        feature_groups : dict
            Словарь с группами признаков
        """
        # Проверяем наличие временных признаков
        if not feature_groups.get('temporal', []):
            return
        
        # Создаем профили для разных часов дня
        if 'hour_of_day' in data.columns:
            self.profiles['temporal']['hour_of_day'] = {}
            
            for hour in range(24):
                hour_data = data[data['hour_of_day'] == hour]
                if not hour_data.empty:
                    # Создаем профиль для каждого часа
                    self.profiles['temporal']['hour_of_day'][str(hour)] = {}
                    
                    # Добавляем статистики по ключевым числовым признакам
                    for feature in feature_groups.get('numeric', []):
                        if feature in hour_data.columns:
                            feature_data = hour_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if not feature_data.empty:
                                mean_val = feature_data.mean()
                                std_val = feature_data.std()
                                
                                self.profiles['temporal']['hour_of_day'][str(hour)][feature] = {
                                    'mean': mean_val,
                                    'std': std_val
                                }
        
        # Создаем профили для разных дней недели
        if 'day_of_week' in data.columns:
            self.profiles['temporal']['day_of_week'] = {}
            
            for day in range(7):
                day_data = data[data['day_of_week'] == day]
                if not day_data.empty:
                    # Создаем профиль для каждого дня
                    self.profiles['temporal']['day_of_week'][str(day)] = {}
                    
                    # Добавляем статистики по ключевым числовым признакам
                    for feature in feature_groups.get('numeric', []):
                        if feature in day_data.columns:
                            feature_data = day_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if not feature_data.empty:
                                mean_val = feature_data.mean()
                                std_val = feature_data.std()
                                
                                self.profiles['temporal']['day_of_week'][str(day)][feature] = {
                                    'mean': mean_val,
                                    'std': std_val
                                }
    
    def _update_protocol_profiles(self, data, feature_groups):
        """
        Обновляет профили протоколов.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профилей
        feature_groups : dict
            Словарь с группами признаков
        """
        # Проверяем наличие информации о протоколах
        protocol_col = None
        if 'protocol' in data.columns:
            protocol_col = 'protocol'
        elif 'protocol_num' in data.columns:
            protocol_col = 'protocol_num'
        
        if protocol_col is None:
            return
        
        # Создаем профили для разных протоколов
        self.profiles['protocol'] = {}
        
        # Получаем уникальные протоколы
        unique_protocols = data[protocol_col].unique()
        
        for protocol in unique_protocols:
            protocol_data = data[data[protocol_col] == protocol]
            if not protocol_data.empty:
                # Создаем профиль для каждого протокола
                self.profiles['protocol'][str(protocol)] = {}
                
                # Добавляем статистики по ключевым числовым признакам
                for feature in feature_groups.get('numeric', []):
                    if feature in protocol_data.columns:
                        feature_data = protocol_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if not feature_data.empty:
                            mean_val = feature_data.mean()
                            std_val = feature_data.std()
                            median_val = feature_data.median()
                            
                            self.profiles['protocol'][str(protocol)][feature] = {
                                'mean': mean_val,
                                'std': std_val,
                                'median': median_val
                            }
    
    def _update_service_profiles(self, data, feature_groups):
        """
        Обновляет профили сервисов/портов.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профилей
        feature_groups : dict
            Словарь с группами признаков
        """
        # Проверяем наличие информации о портах
        if 'dst_port' not in data.columns:
            return
        
        # Создаем профили для разных сервисов (портов назначения)
        self.profiles['service'] = {}
        
        # Определяем пороговое количество соединений для создания профиля
        min_connections = max(10, int(len(data) * 0.01))  # Не менее 10 или 1% от общего объема
        
        # Получаем наиболее распространенные порты
        port_counts = data['dst_port'].value_counts()
        common_ports = port_counts[port_counts >= min_connections].index.tolist()
        
        for port in common_ports:
            port_data = data[data['dst_port'] == port]
            # Создаем профиль для каждого распространенного порта
            self.profiles['service'][str(port)] = {}
            
            # Добавляем статистики по ключевым числовым признакам
            for feature in feature_groups.get('numeric', []):
                if feature in port_data.columns:
                    feature_data = port_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if not feature_data.empty:
                        mean_val = feature_data.mean()
                        std_val = feature_data.std()
                        
                        self.profiles['service'][str(port)][feature] = {
                            'mean': mean_val,
                            'std': std_val
                        }
    
    def _update_contextual_profiles(self, data, feature_groups):
        """
        Обновляет контекстуальные профили.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обновления профилей
        feature_groups : dict
            Словарь с группами признаков
        """
        # Создаем контекстуальные профили для комбинаций условий
        self.profiles['contextual'] = {}
        
        # 1. Профиль рабочие часы / нерабочие часы
        if 'is_working_hours' in data.columns:
            self.profiles['contextual']['working_hours'] = {}
            
            for is_working in [0, 1]:
                working_data = data[data['is_working_hours'] == is_working]
                if not working_data.empty:
                    # Ключ профиля: 'working' или 'non_working'
                    key = 'working' if is_working == 1 else 'non_working'
                    self.profiles['contextual']['working_hours'][key] = {}
                    
                    # Добавляем статистики по ключевым числовым признакам
                    for feature in feature_groups.get('numeric', []):
                        if feature in working_data.columns:
                            feature_data = working_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                            
                            if not feature_data.empty:
                                mean_val = feature_data.mean()
                                std_val = feature_data.std()
                                
                                self.profiles['contextual']['working_hours'][key][feature] = {
                                    'mean': mean_val,
                                    'std': std_val
                                }
        
        # 2. Профиль для разных комбинаций протокол-порт
        if 'protocol_num' in data.columns and 'dst_port' in data.columns:
            self.profiles['contextual']['protocol_port'] = {}
            
            # Определяем распространенные комбинации
            protocol_port_counts = data.groupby(['protocol_num', 'dst_port']).size()
            min_connections = max(10, int(len(data) * 0.01))
            common_combinations = protocol_port_counts[protocol_port_counts >= min_connections].index.tolist()
            
            for protocol, port in common_combinations:
                combo_data = data[(data['protocol_num'] == protocol) & (data['dst_port'] == port)]
                
                # Ключ профиля: 'protocol_X_port_Y'
                key = f'protocol_{protocol}_port_{port}'
                self.profiles['contextual']['protocol_port'][key] = {}
                
                # Добавляем статистики по ключевым числовым признакам
                for feature in feature_groups.get('numeric', []):
                    if feature in combo_data.columns:
                        feature_data = combo_data[feature].replace([np.inf, -np.inf], np.nan).dropna()
                        
                        if not feature_data.empty:
                            mean_val = feature_data.mean()
                            std_val = feature_data.std()
                            
                            self.profiles['contextual']['protocol_port'][key][feature] = {
                                'mean': mean_val,
                                'std': std_val
                            }
    
    def _count_profiles(self):
        """
        Подсчитывает общее количество профилей.
        
        Returns:
        --------
        int
            Общее количество профилей
        """
        count = 0
        
        # Глобальный профиль
        count += len(self.profiles['global'])
        
        # Временные профили
        for time_type, time_profiles in self.profiles['temporal'].items():
            for time_value, features in time_profiles.items():
                count += len(features)
        
        # Профили протоколов
        for protocol, features in self.profiles['protocol'].items():
            count += len(features)
        
        # Профили сервисов
        for service, features in self.profiles['service'].items():
            count += len(features)
        
        # Контекстуальные профили
        for context_type, context_profiles in self.profiles['contextual'].items():
            for context_value, features in context_profiles.items():
                count += len(features)
        
        return count
    
    def save_profiles(self, filepath):
        """
        Сохраняет профили в файл.
        
        Parameters:
        -----------
        filepath : str
            Путь к файлу для сохранения
        """
        # Преобразуем объекты datetime в строки
        profiles_json = json.dumps(self.profiles, default=str)
        
        with open(filepath, 'w') as f:
            f.write(profiles_json)
        
        print(f"Профили сохранены в {filepath}")
    
    def load_profiles(self, filepath):
        """
        Загружает профили из файла.
        
        Parameters:
        -----------
        filepath : str
            Путь к файлу для загрузки
        """
        with open(filepath, 'r') as f:
            self.profiles = json.loads(f.read())
        
        print(f"Профили загружены из {filepath}")
