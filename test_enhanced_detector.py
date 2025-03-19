
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import sys

# Добавляем путь к корневой директории проекта, если скрипт запускается из другой директории
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наш улучшенный детектор
from intellectshield.detectors.enhanced_adaptive_detector import EnhancedAdaptiveDetector

# 1. Создаем синтетические данные для тестирования
def generate_synthetic_data(n_samples=1000, anomaly_ratio=0.05, n_features=10):
    """
    Генерация синтетических данных для тестирования детектора.
    """
    print("Генерация синтетических данных...")
    
    # Генерируем нормальные данные
    normal_samples = int(n_samples * (1 - anomaly_ratio))
    normal_data = np.random.randn(normal_samples, n_features)
    
    # Генерируем аномальные данные (с большим разбросом)
    anomaly_samples = n_samples - normal_samples
    anomalies = np.random.randn(anomaly_samples, n_features) * 3 + 5
    
    # Объединяем данные
    X = np.vstack([normal_data, anomalies])
    y = np.zeros(n_samples)
    y[normal_samples:] = 1  # Метки для аномалий
    
    # Создаем DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['is_anomaly'] = y
    
    # Добавляем временные метки
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='5min')
    df['timestamp'] = timestamps
    
    # Добавляем сетевые признаки
    df['src_port'] = np.random.randint(1024, 65535, size=n_samples)
    df['dst_port'] = np.random.choice([80, 443, 22, 53, 3389] + 
                                    list(range(1024, 10000, 100)), 
                                    size=n_samples)
    
    # Некоторые из аномальных записей имеют подозрительные порты
    if anomaly_samples > 0:
        suspicious_ports = [6667, 31337, 4444, 9001, 1337, 8080]
        df.loc[normal_samples:normal_samples+anomaly_samples//3, 'dst_port'] = np.random.choice(suspicious_ports, 
                                                                                      size=anomaly_samples//3)
    
    # Добавляем признаки трафика
    df['bytes'] = np.random.exponential(1000, size=n_samples)
    df['packets'] = np.random.poisson(5, size=n_samples)
    df['duration'] = np.random.exponential(10, size=n_samples)
    
    # Аномальные записи имеют необычно большой объем трафика
    if anomaly_samples > 0:
        df.loc[normal_samples+anomaly_samples//3:normal_samples+2*anomaly_samples//3, 'bytes'] *= 10
        df.loc[normal_samples+anomaly_samples//3:normal_samples+2*anomaly_samples//3, 'packets'] *= 5
    
    # Добавляем IP-адреса
    df['src_ip'] = [f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)]
    df['dst_ip'] = [f"10.0.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)]
    
    # Разделяем данные на обучающую и тестовую выборки
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
    
    print(f"Создано {len(df)} записей, из них {anomaly_samples} аномалий ({anomaly_ratio*100:.1f}%)")
    print(f"Размер обучающей выборки: {len(train_data)}")
    print(f"Размер тестовой выборки: {len(test_data)}")
    
    return train_data, test_data

# 2. Тестирование детектора
def test_enhanced_detector():
    """
    Функция для тестирования улучшенного детектора аномалий.
    """
    print("\n--- Тестирование улучшенного адаптивного детектора ---")
    
    # Генерация синтетических данных
    train_data, test_data = generate_synthetic_data(n_samples=2000, anomaly_ratio=0.05, n_features=10)
    
    # Создание и обучение детектора
    print("\nСоздание и обучение детектора...")
    detector = EnhancedAdaptiveDetector()
    detector.train(train_data)
    
    # Предсказание аномалий
    print("\nОбнаружение аномалий в тестовых данных...")
    results = detector.predict(test_data)
    
    # Оценка результатов
    if 'is_anomaly' in results.columns:
        print("\nМетрики качества обнаружения:")
        print(classification_report(results['is_anomaly'], results['predicted_anomaly']))
    
    # Подсчет и вывод статистики обнаруженных аномалий
    anomaly_count = results['predicted_anomaly'].sum()
    print(f"\nОбнаружено {anomaly_count} аномалий ({anomaly_count/len(results)*100:.2f}%)")
    
    # Визуализация результатов
    print("\nВизуализация результатов обнаружения аномалий...")
    detector.visualize_anomalies(results)
    
    return detector, results

# Запуск тестирования
if __name__ == "__main__":
    detector, results = test_enhanced_detector()
