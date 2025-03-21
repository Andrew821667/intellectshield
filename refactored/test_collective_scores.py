
import pandas as pd
import numpy as np
import sys
import os

# Добавляем корневую директорию проекта в путь
sys.path.append(os.path.abspath("."))

from intellectshield.refactored.detectors.enhanced_adaptive_detector import (
    EnhancedAdaptiveDetector,
    AnomalyScoreCalculator,
    ProfileManager,
    ThresholdManager,
    MLModelsManager
)

def generate_test_data(n_samples=100):
    # Генерируем временные метки
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    
    # Генерируем нормальные данные
    data = pd.DataFrame({
        'timestamp': timestamps,
        'bytes': np.random.normal(1024, 256, n_samples),
        'packets': np.random.normal(10, 3, n_samples),
        'bytes_per_second': np.random.normal(100, 30, n_samples),
        'packets_per_second': np.random.normal(5, 2, n_samples)
    })
    
    # Добавляем аномалии в определенных точках для тестирования коллективных аномалий
    # Резкий скачок в bytes_per_second
    data.loc[30:32, 'bytes_per_second'] = [100, 500, 100]
    
    # Резкий скачок в packets_per_second
    data.loc[60:62, 'packets_per_second'] = [5, 20, 5]
    
    return data

def test_collective_scores():
    # Создаем тестовые данные
    data = generate_test_data(100)
    
    # Инициализируем необходимые компоненты
    profile_manager = ProfileManager()
    threshold_manager = ThresholdManager()
    ml_models_manager = MLModelsManager()
    
    # Инициализируем калькулятор оценок аномальности
    calculator = AnomalyScoreCalculator(
        profile_manager=profile_manager,
        threshold_manager=threshold_manager,
        ml_models_manager=ml_models_manager
    )
    
    # Проверяем функцию calculate_collective_scores
    print("Тестирование calculate_collective_scores...")
    
    try:
        scores = calculator.calculate_collective_scores(data)
        print(f"Успешно! Размер оценок: {len(scores)}")
        print(f"Ненулевые значения: {(scores > 0).sum()}")
        
        # Выводим индексы с высокими оценками
        high_score_indices = np.where(scores > 0)[0]
        if len(high_score_indices) > 0:
            print("Индексы с высокими оценками:")
            for idx in high_score_indices:
                print(f"Индекс {idx}: {scores[idx]:.4f}")
        
    except Exception as e:
        print(f"Ошибка при тестировании calculate_collective_scores: {e}")

if __name__ == "__main__":
    test_collective_scores()
