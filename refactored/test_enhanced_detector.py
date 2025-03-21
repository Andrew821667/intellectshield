"""
Тестирование рефакторированного расширенного адаптивного детектора.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

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

def main():
    """
    Основная функция для тестирования детектора.
    """
    print("Генерация тестовых данных...")
    data = generate_test_data(n_samples=1000, anomaly_ratio=0.05)
    
    print("Создание профилей...")
    profiles = create_test_profiles(data)
    
    # Группы признаков
    feature_groups = {
        'numeric': ['bytes', 'packets', 'duration', 'bytes_per_second', 'packets_per_second'],
        'categorical': ['protocol_num', 'dst_port', 'hour_of_day', 'is_working_hours']
    }
    
    # Множители порогов
    threshold_multipliers = {
        'low': 5.0,
        'medium': 3.0,
        'high': 1.5
    }
    
    print("Инициализация детектора...")
    detector = EnhancedAdaptiveDetector()
    detector.initialize(
        data=data, 
        profiles=profiles, 
        feature_groups=feature_groups,
        threshold_multipliers=threshold_multipliers
    )
    
    print("Обучение детектора...")
    training_summary = detector.train(data)
    print(f"Детектор обучен. Сводка: {training_summary}")
    
    print("Обнаружение аномалий...")
    results = detector.predict(data)
    
    print(f"Обнаружено {results['predicted_anomaly'].sum()} аномалий.")
    print("Типы аномалий:")
    print(results[results['predicted_anomaly'] == 1]['anomaly_type'].value_counts())
    
    # Оценка производительности
    evaluation = detector.evaluate(results)
    print("Метрики оценки:")
    print(f"Точность: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1-score: {evaluation['f1_score']:.4f}")
    
    # Визуализация результатов
    plt.figure(figsize=(15, 10))
    
    # 1. График аномальных оценок
    plt.subplot(2, 2, 1)
    plt.scatter(range(len(results)), results['anomaly_score'], c=results['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
    plt.title('Аномальные оценки')
    plt.xlabel('Индекс')
    plt.ylabel('Оценка аномальности')
    
    # 2. Гистограмма аномальных оценок
    plt.subplot(2, 2, 2)
    sns.histplot(data=results, x='anomaly_score', hue='predicted_anomaly', kde=True, palette={0: 'blue', 1: 'red'})
    plt.title('Распределение аномальных оценок')
    plt.xlabel('Оценка аномальности')
    plt.ylabel('Количество')
    
    # 3. Матрица ошибок
    plt.subplot(2, 2, 3)
    cm = evaluation['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Нормальный', 'Аномалия'], yticklabels=['Нормальный', 'Аномалия'])
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    
    # 4. Типы аномалий
    plt.subplot(2, 2, 4)
    anomaly_types = results[results['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
    sns.barplot(x=anomaly_types.values, y=anomaly_types.index)
    plt.title('Типы обнаруженных аномалий')
    plt.xlabel('Количество')
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_results.png')
    plt.show()

if __name__ == "__main__":
    main()
