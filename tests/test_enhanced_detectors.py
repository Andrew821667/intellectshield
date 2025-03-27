"""
Тесты для улучшенных детекторов аномалий IntellectShield.

Эти тесты сравнивают производительность стандартных и улучшенных 
детекторов на реальных данных.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

# Добавляем родительскую директорию в путь импорта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nsl_kdd_data(data_dir="datasets", sample_size=None):
    """
    Загрузка данных NSL-KDD.
    
    Parameters:
    -----------
    data_dir : str
        Директория, содержащая файлы NSL-KDD
    sample_size : int или None
        Размер выборки данных (если None, используются все данные)
        
    Returns:
    --------
    tuple
        (train_data, test_data) - наборы данных для обучения и тестирования
    """
    train_path = os.path.join(data_dir, "NSL_KDD_train.csv")
    test_path = os.path.join(data_dir, "NSL_KDD_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Файлы NSL-KDD не найдены. Запустите скрипт загрузки данных.")
        return None, None
    
    logger.info(f"Загрузка данных из {train_path} и {test_path}...")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Делаем выборку данных, если указан размер выборки
    if sample_size is not None:
        train_data = train_data.sample(n=min(sample_size, len(train_data)), random_state=42)
        test_data = test_data.sample(n=min(sample_size//4, len(test_data)), random_state=42)
    
    logger.info(f"Загружено: {len(train_data)} записей для обучения, {len(test_data)} для тестирования")
    logger.info(f"Процент аномалий в тестовом наборе: {test_data['is_anomaly'].mean() * 100:.2f}%")
    
    # Удаляем ненужные колонки
    drop_columns = ['class', 'difficulty']
    train_data = train_data.drop(drop_columns, axis=1, errors='ignore')
    test_data = test_data.drop(drop_columns, axis=1, errors='ignore')
    
    return train_data, test_data

def test_enhanced_isolation_forest(train_data, test_data, auto_tune=True, save_models=True):
    """
    Сравнительное тестирование стандартного и улучшенного Isolation Forest.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Данные для обучения
    test_data : pandas.DataFrame
        Данные для тестирования
    auto_tune : bool
        Включить автоматическую настройку параметров для улучшенного детектора
    save_models : bool
        Сохранять ли обученные модели
        
    Returns:
    --------
    dict
        Результаты сравнительного тестирования
    """
    from intellectshield.detectors.anomaly.isolation_forest import IsolationForestDetector
    from intellectshield.detectors.anomaly.enhanced_isolation_forest import EnhancedIsolationForestDetector
    
    # Создаем директорию для моделей
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("Запуск сравнительного тестирования детекторов Isolation Forest...")
    
    # Создаем стандартный и улучшенный детекторы
    standard_detector = IsolationForestDetector(model_dir=models_dir)
    enhanced_detector = EnhancedIsolationForestDetector(
        model_dir=models_dir,
        use_pca=True, 
        dynamic_threshold=True
    )
    
    # Тестирование стандартного детектора
    logger.info("--- Тестирование стандартного детектора ---")
    standard_model_path = os.path.join(models_dir, "standard_isolation_forest.pkl")
    
    if os.path.exists(standard_model_path) and save_models:
        logger.info(f"Загрузка сохраненной модели из {standard_model_path}")
        standard_detector.load_model(standard_model_path)
        standard_train_time = 0
    else:
        start_time = time.time()
        standard_detector.train(train_data)
        standard_train_time = time.time() - start_time
        
        if save_models:
            logger.info(f"Сохранение модели в {standard_model_path}")
            standard_detector.save_model(standard_model_path)
    
    standard_predictions = standard_detector.predict(test_data)
    standard_eval = standard_detector.evaluate(test_data)
    
    logger.info(f"Время обучения: {standard_train_time:.2f} секунд")
    logger.info(f"Precision: {standard_eval['precision']:.4f}")
    logger.info(f"Recall: {standard_eval['recall']:.4f}")
    logger.info(f"F1 Score: {standard_eval['f1_score']:.4f}")
    
    # Тестирование улучшенного детектора
    logger.info("--- Тестирование улучшенного детектора ---")
    enhanced_model_path = os.path.join(models_dir, "enhanced_isolation_forest.pkl")
    
    if os.path.exists(enhanced_model_path) and save_models:
        logger.info(f"Загрузка сохраненной модели из {enhanced_model_path}")
        enhanced_detector.load_model(enhanced_model_path)
        enhanced_train_time = 0
    else:
        start_time = time.time()
        enhanced_detector.train(train_data, auto_tune=auto_tune)
        enhanced_train_time = time.time() - start_time
        
        if save_models:
            logger.info(f"Сохранение модели в {enhanced_model_path}")
            enhanced_detector.save_model(enhanced_model_path)
    
    enhanced_predictions = enhanced_detector.predict(test_data)
    enhanced_eval = enhanced_detector.evaluate(test_data)
    
    logger.info(f"Время обучения: {enhanced_train_time:.2f} секунд")
    logger.info(f"Precision: {enhanced_eval['precision']:.4f}")
    logger.info(f"Recall: {enhanced_eval['recall']:.4f}")
    logger.info(f"F1 Score: {enhanced_eval['f1_score']:.4f}")
    
    # Сравнение результатов
    logger.info("--- Сравнение результатов ---")
    improvement_precision = (enhanced_eval['precision'] / standard_eval['precision'] * 100) - 100
    improvement_recall = (enhanced_eval['recall'] / standard_eval['recall'] * 100) - 100
    improvement_f1 = (enhanced_eval['f1_score'] / standard_eval['f1_score'] * 100) - 100
    
    logger.info(f"Улучшение Precision: {improvement_precision:.2f}%")
    logger.info(f"Улучшение Recall: {improvement_recall:.2f}%")
    logger.info(f"Улучшение F1 Score: {improvement_f1:.2f}%")
    
    # Визуализация результатов
    plt.figure(figsize=(10, 6))
    metrics = ['Precision', 'Recall', 'F1 Score']
    standard_values = [standard_eval['precision'], standard_eval['recall'], standard_eval['f1_score']]
    enhanced_values = [enhanced_eval['precision'], enhanced_eval['recall'], enhanced_eval['f1_score']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, standard_values, width, label='Стандартный')
    plt.bar(x + width/2, enhanced_values, width, label='Улучшенный')
    
    plt.ylabel('Значение')
    plt.title('Сравнение стандартного и улучшенного детекторов Isolation Forest')
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Добавляем значения над столбцами
    for i, v in enumerate(standard_values):
        plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    for i, v in enumerate(enhanced_values):
        plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'comparison_results.png'))
    
    return {
        'standard': {
            'detector': standard_detector,
            'predictions': standard_predictions,
            'evaluation': standard_eval,
            'training_time': standard_train_time
        },
        'enhanced': {
            'detector': enhanced_detector,
            'predictions': enhanced_predictions,
            'evaluation': enhanced_eval,
            'training_time': enhanced_train_time
        }
    }

def run_enhanced_tests(sample_size=10000):
    """
    Запуск тестов улучшенных детекторов на данных NSL-KDD.
    
    Parameters:
    -----------
    sample_size : int
        Размер выборки данных
        
    Returns:
    --------
    dict
        Результаты тестирования
    """
    # Загружаем данные
    logger.info(f"Загрузка данных NSL-KDD (размер выборки: {sample_size})...")
    train_data, test_data = load_nsl_kdd_data(sample_size=sample_size)
    
    if train_data is None or test_data is None:
        logger.error("Не удалось загрузить данные")
        return None
    
    # Тестируем улучшенный Isolation Forest
    logger.info("Запуск тестирования улучшенного Isolation Forest...")
    isolation_forest_results = test_enhanced_isolation_forest(train_data, test_data)
    
    # Здесь можно добавить тесты для других улучшенных детекторов
    
    return {
        'isolation_forest': isolation_forest_results,
        # Другие результаты можно добавить здесь
    }

if __name__ == "__main__":
    # Запуск тестов улучшенных детекторов с уменьшенной выборкой для быстроты
    start_time = time.time()
    results = run_enhanced_tests(sample_size=10000)
    end_time = time.time()
    
    if results:
        logger.info(f"Тестирование улучшенных детекторов завершено за {end_time - start_time:.2f} секунд!")
