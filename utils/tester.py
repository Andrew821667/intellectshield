"""Модуль для тестирования детекторов аномалий.

Предоставляет функции и классы для оценки производительности и точности
детекторов аномалий на различных наборах данных.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from intellectshield.visualizers.visualization import visualize_ensemble_results, visualize_sequence_analysis

def test_base_detectors(train_data, test_data, detectors):
    """
    Тестирование базовых детекторов аномалий.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Данные для обучения
    test_data : pandas.DataFrame
        Данные для тестирования
    detectors : dict
        Словарь с детекторами аномалий
        
    Returns:
    --------
    dict
        Результаты тестирования базовых детекторов
    """
    print("Тестирование базовых детекторов аномалий...")
    
    results = {}
    evaluations = {}
    
    for name, detector in detectors.items():
        print(f"\nТестирование детектора {name}:")
        
        # Засекаем время обучения
        start_time = time.time()
        
        # Обучаем детектор
        detector.train(train_data)
        
        training_time = time.time() - start_time
        print(f"Время обучения: {training_time:.2f} секунд")
        
        # Предсказание и оценка
        test_result = detector.predict(test_data)
        evaluation = detector.evaluate(test_data)
        
        # Выводим метрики
        print(f"Precision: {evaluation['precision']:.4f}")
        print(f"Recall: {evaluation['recall']:.4f}")
        print(f"F1 Score: {evaluation['f1_score']:.4f}")
        
        # Сохраняем результаты
        results[name] = test_result
        evaluations[name] = evaluation
        
        # Визуализация результатов (только если она требуется)
        if name == 'SequenceAnomalyDetector':
            visualize_sequence_analysis(detector, test_data)
    
    # Возвращаем детекторы, результаты и оценки
    return {
        'detectors': detectors,
        'results': results,
        'evaluations': evaluations,
        'data': {'train': train_data, 'test': test_data}
    }

def test_ensemble_detector(base_results, ensemble_method="weighted_average"):
    """
    Тестирование ансамблевого детектора аномалий.
    
    Parameters:
    -----------
    base_results : dict
        Результаты тестирования базовых детекторов
    ensemble_method : str
        Метод ансамблирования
        
    Returns:
    --------
    dict
        Результаты тестирования ансамблевого детектора
    """
    from intellectshield.detectors.ensemble import EnsembleAnomalyDetector
    
    print(f"\nТестирование ансамблевого детектора с методом {ensemble_method}:")
    
    # Создаем ансамблевый детектор
    ensemble_detector = EnsembleAnomalyDetector()
    
    # Добавляем базовые детекторы с весами на основе F1-метрики
    for name, detector in base_results['detectors'].items():
        f1_score = base_results['evaluations'][name]['f1_score']
        # Используем вес, пропорциональный F1-метрике
        weight = max(f1_score, 0.01)  # Минимальный вес 0.01
        ensemble_detector.add_detector(detector, weight=weight)
    
    # Устанавливаем метод ансамблирования
    ensemble_detector.set_ensemble_method(ensemble_method)
    
    # Обучаем ансамблевый детектор (на тех же данных, что и базовые)
    ensemble_detector.train(base_results['data']['train'])
    
    # Предсказание и оценка
    ensemble_result = ensemble_detector.predict(base_results['data']['test'])
    ensemble_evaluation = ensemble_detector.evaluate(base_results['data']['test'])
    
    # Выводим метрики
    print(f"Precision: {ensemble_evaluation['precision']:.4f}")
    print(f"Recall: {ensemble_evaluation['recall']:.4f}")
    print(f"F1 Score: {ensemble_evaluation['f1_score']:.4f}")
    
    # Визуализация результатов
    visualize_ensemble_results(ensemble_result)
    
    # Сравнение с базовыми детекторами
    compare_detectors(base_results, ensemble_evaluation, ensemble_method)
    
    return {
        'detector': ensemble_detector,
        'result': ensemble_result,
        'evaluation': ensemble_evaluation
    }

def compare_detectors(base_results, ensemble_evaluation, ensemble_method):
    """
    Сравнение эффективности базовых детекторов и ансамблевого метода.
    
    Parameters:
    -----------
    base_results : dict
        Результаты тестирования базовых детекторов
    ensemble_evaluation : dict
        Оценки ансамблевого детектора
    ensemble_method : str
        Метод ансамблирования
    """
    print("\nСравнение детекторов по F1-метрике:")
    
    # Создаем словарь с F1-метриками для всех детекторов
    f1_scores = {}
    for name, evaluation in base_results['evaluations'].items():
        f1_scores[name] = evaluation['f1_score']
    
    # Добавляем ансамблевый детектор
    f1_scores[f"Ensemble ({ensemble_method})"] = ensemble_evaluation['f1_score']
    
    # Создаем DataFrame для визуализации
    df = pd.DataFrame(list(f1_scores.items()), columns=['Detector', 'F1 Score'])
    df = df.sort_values('F1 Score', ascending=False)
    
    # Выводим таблицу
    print(df)
    
    # Визуализация сравнения
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Detector'], df['F1 Score'], color='skyblue')
    
    # Выделяем ансамблевый детектор другим цветом
    ensemble_index = df['Detector'].tolist().index(f"Ensemble ({ensemble_method})")
    bars[ensemble_index].set_color('coral')
    
    plt.title('Сравнение F1-метрик различных детекторов')
    plt.xlabel('Детектор')
    plt.ylabel('F1-метрика')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем значения над каждым столбцом
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.show()

def optimize_ensemble_threshold(ensemble_detector, test_data, threshold_range=range(70, 96, 5)):
    """
    Оптимизирует порог для ансамблевого метода.
    
    Parameters:
    -----------
    ensemble_detector : EnsembleAnomalyDetector
        Экземпляр ансамблевого детектора
    test_data : pandas.DataFrame
        Тестовые данные
    threshold_range : range
        Диапазон порогов (перцентили) для поиска оптимального порога
        
    Returns:
    --------
    tuple
        (best_threshold, best_f1)
    """
    # Получаем предсказания ансамбля
    ensemble_predictions = ensemble_detector.predict(test_data)
    
    if 'anomaly_score' not in ensemble_predictions.columns or 'is_anomaly' not in ensemble_predictions.columns:
        raise ValueError("DataFrame должен содержать колонки 'anomaly_score' и 'is_anomaly'")
    
    best_threshold = None
    best_f1 = 0
    results = []
    
    for percentile in threshold_range:
        # Вычисляем порог
        threshold = np.percentile(ensemble_predictions['anomaly_score'], percentile)
        
        # Определяем аномалии с текущим порогом
        predicted_anomalies = (ensemble_predictions['anomaly_score'] >= threshold).astype(int)
        
        # Вычисляем метрики
        tp = ((ensemble_predictions['is_anomaly'] == 1) & (predicted_anomalies == 1)).sum()
        fp = ((ensemble_predictions['is_anomaly'] == 0) & (predicted_anomalies == 1)).sum()
        fn = ((ensemble_predictions['is_anomaly'] == 1) & (predicted_anomalies == 0)).sum()
        tn = ((ensemble_predictions['is_anomaly'] == 0) & (predicted_anomalies == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'percentile': percentile,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"Percentile: {percentile}, Threshold: {threshold:.6f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Выводим результаты
    results_df = pd.DataFrame(results)
    
    # Визуализация результатов
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['percentile'], results_df['precision'], 'b-', label='Precision')
    plt.plot(results_df['percentile'], results_df['recall'], 'r-', label='Recall')
    plt.plot(results_df['percentile'], results_df['f1'], 'g-', label='F1')
    plt.xlabel('Percentile')
    plt.ylabel('Score')
    plt.title('Метрики при разных порогах')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.bar(results_df['percentile'], results_df['f1'])
    plt.xlabel('Percentile')
    plt.ylabel('F1 Score')
    plt.title('F1 Score при разных порогах')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nЛучший порог: {best_threshold:.6f} (перцентиль: {results_df.loc[results_df['f1'] == best_f1, 'percentile'].values[0]})")
    print(f"F1 score: {best_f1:.4f}")
    
    return best_threshold, best_f1
