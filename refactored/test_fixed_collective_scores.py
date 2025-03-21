
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

def generate_complex_test_data(n_samples=100):
    """Генерирует тестовые данные с разными типами индексов."""
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
    
    # Добавляем аномалии
    data.loc[30:32, 'bytes_per_second'] = [100, 500, 100]
    data.loc[60:62, 'packets_per_second'] = [5, 20, 5]
    
    # Создаем сложные индексы
    return data

def test_with_various_indices():
    """Тестирует функцию с разными типами индексов."""
    
    # Инициализируем необходимые компоненты
    profile_manager = ProfileManager()
    threshold_manager = ThresholdManager()
    ml_models_manager = MLModelsManager()
    
    calculator = AnomalyScoreCalculator(
        profile_manager=profile_manager,
        threshold_manager=threshold_manager,
        ml_models_manager=ml_models_manager
    )
    
    print("Тест 1: Стандартные последовательные индексы")
    data1 = generate_complex_test_data(100)
    scores1 = calculator.calculate_collective_scores(data1)
    print(f"Размер оценок: {len(scores1)}, Ненулевые значения: {(scores1 > 0).sum()}")
    
    print("\nТест 2: Нестандартные индексы (строки)")
    data2 = generate_complex_test_data(100)
    data2.index = [f'row_{i}' for i in range(len(data2))]
    scores2 = calculator.calculate_collective_scores(data2)
    print(f"Размер оценок: {len(scores2)}, Ненулевые значения: {(scores2 > 0).sum()}")
    
    print("\nТест 3: Непоследовательные индексы")
    data3 = generate_complex_test_data(100)
    data3.index = np.random.choice(range(1000), size=len(data3), replace=False)
    scores3 = calculator.calculate_collective_scores(data3)
    print(f"Размер оценок: {len(scores3)}, Ненулевые значения: {(scores3 > 0).sum()}")
    
    print("\nТест 4: Индексы с дубликатами")
    data4 = generate_complex_test_data(100)
    data4.index = np.random.choice(range(50), size=len(data4), replace=True)
    scores4 = calculator.calculate_collective_scores(data4)
    print(f"Размер оценок: {len(scores4)}, Ненулевые значения: {(scores4 > 0).sum()}")
    
    print("\nТест 5: MultiIndex")
    data5 = generate_complex_test_data(100)
    data5.index = pd.MultiIndex.from_tuples([(i // 10, i % 10) for i in range(len(data5))], names=['group', 'item'])
    scores5 = calculator.calculate_collective_scores(data5)
    print(f"Размер оценок: {len(scores5)}, Ненулевые значения: {(scores5 > 0).sum()}")
    
    print("\nВсе тесты успешно завершены!")

if __name__ == "__main__":
    test_with_various_indices()
