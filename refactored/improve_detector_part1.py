"""
Улучшение рефакторированного расширенного адаптивного детектора.

В этом скрипте мы внесем улучшения в детектор аномалий 
для повышения его точности и полноты обнаружения.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.abspath("."))

from intellectshield.refactored.detectors.enhanced_adaptive_detector import (
    EnhancedAdaptiveDetector,
    ProfileManager,
    ThresholdManager,
    MLModelsManager,
    AnomalyScoreCalculator,
    AnomalyTypeDeterminer
)

# Функции для генерации данных и создания профилей из предыдущего теста
def generate_test_data(n_samples=1000, anomaly_ratio=0.05):
    """
    Генерирует тестовые данные с аномалиями.
    """
    # Генерируем временные метки
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # Генерируем нормальные данные
    data = pd.DataFrame({
        'timestamp': timestamps,
        'bytes': np.random.normal(1024, 256, n_samples),
        'packets': np.random.normal(10, 3, n_samples),
        'duration': np.random.normal(30, 10, n_samples),
        'protocol_num': np.random.choice([6, 17], n_samples),  # TCP, UDP
        'dst_port': np.random.choice([80, 443, 8080, 22], n_samples)
    })
    
    # Добавляем контекстуальные признаки
    data['hour_of_day'] = data['timestamp'].dt.hour
    data['is_working_hours'] = ((data['hour_of_day'] >= 9) & (data['hour_of_day'] <= 17)).astype(int)
    
    # Вычисляем производные признаки
    data['bytes_per_second'] = data['bytes'] / (data['duration'] + 1)
    data['packets_per_second'] = data['packets'] / (data['duration'] + 1)
    
    # Добавляем аномалии
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Разные типы аномалий
    for i in anomaly_indices:
        anomaly_type = np.random.choice(['volume', 'port', 'time', 'rate'])
        
        if anomaly_type == 'volume':
            data.loc[i, 'bytes'] *= 10
            data.loc[i, 'bytes_per_second'] *= 10
            data.loc[i, 'is_dos_like'] = 1
        elif anomaly_type == 'port':
            data.loc[i, 'dst_port'] = np.random.choice([31337, 4444, 6667])  # Подозрительные порты
            data.loc[i, 'dst_port_suspicious'] = 1
        elif anomaly_type == 'time':
            data.loc[i, 'hour_of_day'] = np.random.choice([1, 2, 3, 4])  # Ночное время
        elif anomaly_type == 'rate':
            data.loc[i, 'packets_per_second'] *= 20
            data.loc[i, 'is_port_scan_like'] = 1
    
    # Заполняем пропуски нулями
    data = data.fillna(0)
    
    # Добавляем истинные метки аномалий
    data['is_anomaly'] = 0
    data.loc[anomaly_indices, 'is_anomaly'] = 1
    
    return data

def create_test_profiles(data):
    """
    Создает тестовые профили на основе данных.
    """
    # Глобальный профиль
    global_profile = {}
    
    numeric_cols = ['bytes', 'packets', 'duration', 'bytes_per_second', 'packets_per_second']
    for col in numeric_cols:
        # Вычисляем статистики
        mean = data[col].mean()
        std = data[col].std()
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        
        global_profile[col] = {
            'mean': mean,
            'std': std,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }
    
    # Временные профили
    temporal_profile = {'hour_of_day': {}}
    for hour in range(24):
        hour_data = data[data['hour_of_day'] == hour]
        if len(hour_data) > 0:
            temporal_profile['hour_of_day'][str(hour)] = {}
            for col in numeric_cols:
                temporal_profile['hour_of_day'][str(hour)][col] = {
                    'mean': hour_data[col].mean(),
                    'std': hour_data[col].std()
                }
    
    # Контекстуальные профили
    contextual_profile = {
        'working_hours': {
            'working': {},
            'non_working': {}
        },
        'protocol_port': {}
    }
    
    # Рабочие часы
    working_data = data[data['is_working_hours'] == 1]
    non_working_data = data[data['is_working_hours'] == 0]
    
    for col in numeric_cols:
        if len(working_data) > 0:
            contextual_profile['working_hours']['working'][col] = {
                'mean': working_data[col].mean(),
                'std': working_data[col].std()
            }
        
        if len(non_working_data) > 0:
            contextual_profile['working_hours']['non_working'][col] = {
                'mean': non_working_data[col].mean(),
                'std': non_working_data[col].std()
            }
    
    # Протокол-порт
    for protocol in data['protocol_num'].unique():
        for port in data['dst_port'].unique():
            proto_port_data = data[(data['protocol_num'] == protocol) & (data['dst_port'] == port)]
            if len(proto_port_data) > 0:
                key = f'protocol_{protocol}_port_{port}'
                contextual_profile['protocol_port'][key] = {}
                for col in numeric_cols:
                    contextual_profile['protocol_port'][key][col] = {
                        'mean': proto_port_data[col].mean(),
                        'std': proto_port_data[col].std()
                    }
    
    return {
        'global': global_profile,
        'temporal': temporal_profile,
        'contextual': contextual_profile
    }
