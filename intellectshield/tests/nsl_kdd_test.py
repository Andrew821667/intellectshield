"""
Тест системы IntellectShield на данных NSL-KDD.

Этот модуль предоставляет функции для тестирования детекторов
на наборе данных NSL-KDD.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import logging

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

def run_nsl_kdd_test(sample_size=None):
    """
    Запуск тестов на данных NSL-KDD.
    
    Parameters:
    -----------
    sample_size : int или None
        Размер выборки для тестирования
        
    Returns:
    --------
    dict
        Результаты тестирования
    """
    from intellectshield.detectors.anomaly.isolation_forest import IsolationForestDetector
    from intellectshield.detectors.anomaly.lof import LOFDetector
    from intellectshield.detectors.anomaly.sequence import SequenceAnomalyDetector
    from intellectshield.utils.tester import test_base_detectors, test_ensemble_detector
    
    logger.info(f"Запуск тестов на данных NSL-KDD (размер выборки: {sample_size if sample_size else 'полный'})...")
    
    # 1. Загрузка данных
    logger.info("--- Шаг 1: Загрузка данных ---")
    train_data, test_data = load_nsl_kdd_data(sample_size=sample_size)
    
    if train_data is None or test_data is None:
        logger.error("Не удалось загрузить данные")
        return None
    
    # 2. Создание и тестирование базовых детекторов
    logger.info("--- Шаг 2: Тестирование базовых детекторов ---")
    detectors = {
        'IsolationForest': IsolationForestDetector(),
        'LOF': LOFDetector(),
        'SequenceAnomalyDetector': SequenceAnomalyDetector()
    }
    
    # Запуск тестирования базовых детекторов
    base_results = test_base_detectors(train_data, test_data, detectors)
    
    # 3. Тестирование ансамблевых детекторов
    logger.info("--- Шаг 3: Тестирование ансамблевых детекторов ---")
    ensemble_results = {}
    
    # Тестируем все методы ансамблирования
    for method in ["weighted_average", "majority_voting", "stacking"]:
        try:
            logger.info(f"Тестирование метода ансамблирования: {method}")
            ensemble_results[method] = test_ensemble_detector(base_results, ensemble_method=method)
        except Exception as e:
            logger.error(f"Ошибка при тестировании метода {method}: {e}")
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'base_results': base_results,
        'ensemble_results': ensemble_results
    }

def download_nsl_kdd_data(data_dir="datasets"):
    """
    Загрузка набора данных NSL-KDD.
    
    Parameters:
    -----------
    data_dir : str
        Директория для сохранения данных
    """
    import urllib.request
    
    # Создаем директорию для данных
    os.makedirs(data_dir, exist_ok=True)
    
    # URL для загрузки NSL-KDD
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    train_path = os.path.join(data_dir, "NSL_KDD_train.csv")
    test_path = os.path.join(data_dir, "NSL_KDD_test.csv")
    
    # Имена колонок для NSL-KDD
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'class', 'difficulty'
    ]
    
    logger.info(f"Загрузка данных NSL-KDD...")
    
    try:
        logger.info(f"Загрузка обучающего набора из {train_url}...")
        urllib.request.urlretrieve(train_url, train_path)
        logger.info(f"Загрузка тестового набора из {test_url}...")
        urllib.request.urlretrieve(test_url, test_path)
        
        # Загружаем данные
        train_df = pd.read_csv(train_path, header=None, names=column_names)
        test_df = pd.read_csv(test_path, header=None, names=column_names)
        
        # Обрабатываем данные
        for df in [train_df, test_df]:
            # Преобразуем класс в бинарную метку (0 - нормально, 1 - атака)
            df['is_anomaly'] = (df['class'] != 'normal').astype(int)
            
            # Преобразуем категориальные признаки
            categorical_features = ['protocol_type', 'service', 'flag']
            for feature in categorical_features:
                df[feature] = df[feature].astype('category').cat.codes
        
        # Сохраняем обработанные данные
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Загружено {len(train_df)} обучающих и {len(test_df)} тестовых записей")
        logger.info(f"Данные сохранены в {train_path} и {test_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return False

if __name__ == "__main__":
    # Выполнение тестов на данных NSL-KDD
    start_time = time.time()
    # Используем выборку для более быстрого тестирования
    results = run_nsl_kdd_test(sample_size=10000)
    end_time = time.time()
    
    if results:
        logger.info(f"Тесты на данных NSL-KDD успешно завершены за {end_time - start_time:.2f} секунд!")
