
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intellectshield.detectors.enhanced_adaptive_detector import EnhancedAdaptiveDetector
from intellectshield.data.data_loader import load_kdd_cup_data

# Загрузка данных KDD Cup
train_data, test_data = load_kdd_cup_data()

print(f"Размер обучающих данных: {train_data.shape}")
print(f"Размер тестовых данных: {test_data.shape}")

# Создание и обучение улучшенного адаптивного детектора
detector = EnhancedAdaptiveDetector()
detector.train(train_data)

# Обнаружение аномалий
results = detector.predict(test_data)

# Вывод результатов
anomaly_count = results['predicted_anomaly'].sum()
print(f"Обнаружено {anomaly_count} аномалий ({anomaly_count/len(results)*100:.2f}%)")

# Оценка качества обнаружения
if 'is_anomaly' in results.columns:
    from sklearn.metrics import classification_report
    print("\nМетрики качества обнаружения:")
    print(classification_report(results['is_anomaly'], results['predicted_anomaly']))

# Визуализация результатов
detector.visualize_anomalies(results)
