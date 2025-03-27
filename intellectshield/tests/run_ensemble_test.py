"""
Тестирование ансамблевых методов с улучшенным Isolation Forest на реальных данных NSL-KDD.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import time
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_nsl_kdd_data(data_dir="datasets", sample_size=None):
    """Загрузка данных NSL-KDD."""
    train_path = os.path.join(data_dir, "NSL_KDD_train.csv")
    test_path = os.path.join(data_dir, "NSL_KDD_test.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logger.error("Файлы NSL-KDD не найдены")
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

def improved_isolation_forest(train_data, test_data):
    """Улучшенный Isolation Forest с PCA и динамическим порогом."""
    # Получаем данные для обучения и тестирования
    X_train = train_data.drop('is_anomaly', axis=1)
    y_train = train_data['is_anomaly']
    
    X_test = test_data.drop('is_anomaly', axis=1)
    y_test = test_data['is_anomaly']
    
    # Предобработка данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Применение PCA
    pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    logger.info(f"Сокращение размерности с {X_train.shape[1]} до {X_train_pca.shape[1]} признаков")
    
    # Создаем детектор
    clf = IsolationForest(n_estimators=200, contamination=0.2, random_state=42)
    
    # Обучаем модель
    start_time = time.time()
    clf.fit(X_train_pca)
    training_time = time.time() - start_time
    
    # Получаем аномальные скоры для обучающих данных
    anomaly_scores_train = clf.decision_function(X_train_pca)
    
    # Калибровка порога для оптимизации F1 score
    thresholds = np.linspace(np.min(anomaly_scores_train), np.max(anomaly_scores_train), 100)
    best_f1 = 0
    best_threshold = 0
    
    for threshold in thresholds:
        y_pred = (anomaly_scores_train < threshold).astype(int)
        prec = precision_score(y_train, y_pred, zero_division=0)
        rec = recall_score(y_train, y_pred, zero_division=0)
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Калибровка порога: {best_threshold:.4f}, F1: {best_f1:.4f}")
    
    # Предсказание с калиброванным порогом на тестовых данных
    anomaly_scores_test = clf.decision_function(X_test_pca)
    y_pred = (anomaly_scores_test < best_threshold).astype(int)
    
    # Оценка
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Improved Isolation Forest - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # Возвращаем скоры и порог для использования в ансамбле
    return {
        'name': 'Improved Isolation Forest',
        'predictions': y_pred,
        'scores': anomaly_scores_test,
        'threshold': best_threshold,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1, 'training_time': training_time}
    }

def standard_isolation_forest(train_data, test_data):
    """Стандартный Isolation Forest."""
    # Получаем данные для обучения и тестирования
    X_train = train_data.drop('is_anomaly', axis=1)
    y_train = train_data['is_anomaly']
    
    X_test = test_data.drop('is_anomaly', axis=1)
    y_test = test_data['is_anomaly']
    
    # Создаем детектор
    clf = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
    
    # Обучаем модель
    start_time = time.time()
    clf.fit(X_train)
    training_time = time.time() - start_time
    
    # Предсказания
    anomaly_scores = clf.decision_function(X_test)
    y_pred = (clf.predict(X_test) == -1).astype(int)  # -1 для аномалий, 1 для нормальных точек
    
    # Оценка
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Standard Isolation Forest - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'Standard Isolation Forest',
        'predictions': y_pred,
        'scores': anomaly_scores,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1, 'training_time': training_time}
    }

def lof_detector(train_data, test_data):
    """Local Outlier Factor детектор."""
    # Получаем данные для обучения и тестирования
    X_train = train_data.drop('is_anomaly', axis=1)
    y_train = train_data['is_anomaly']
    
    X_test = test_data.drop('is_anomaly', axis=1)
    y_test = test_data['is_anomaly']
    
    # Предобработка данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Создаем детектор
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    
    # Обучаем модель
    start_time = time.time()
    clf.fit(X_train_scaled)
    training_time = time.time() - start_time
    
    # Предсказания
    anomaly_scores = -clf.decision_function(X_test_scaled)  # Инвертируем, чтобы было как в Isolation Forest
    y_pred = (clf.predict(X_test_scaled) == -1).astype(int)  # -1 для аномалий, 1 для нормальных точек
    
    # Оценка
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"LOF - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'LOF',
        'predictions': y_pred,
        'scores': anomaly_scores,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1, 'training_time': training_time}
    }

def majority_voting_ensemble(detectors_results, test_data):
    """Ансамбль на основе мажоритарного голосования."""
    predictions = []
    for detector in detectors_results:
        predictions.append(detector['predictions'])
    
    # Голосование (1, если большинство детекторов считает точку аномалией)
    ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
    
    # Оценка
    y_test = test_data['is_anomaly']
    prec = precision_score(y_test, ensemble_pred)
    rec = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    logger.info(f"Majority Voting Ensemble - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'Majority Voting Ensemble',
        'predictions': ensemble_pred,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1}
    }

def weighted_voting_ensemble(detectors_results, test_data, weights=None):
    """Ансамбль на основе взвешенного голосования."""
    if weights is None:
        # Если веса не указаны, используем F1-метрику в качестве весов
        weights = [result['metrics']['f1'] for result in detectors_results]
        logger.info(f"Веса на основе F1: {weights}")
    
    predictions = []
    for detector, weight in zip(detectors_results, weights):
        predictions.append(detector['predictions'] * weight)
    
    # Взвешенное голосование (1, если взвешенная сумма > 0.5)
    ensemble_pred = (np.sum(predictions, axis=0) / np.sum(weights) > 0.5).astype(int)
    
    # Оценка
    y_test = test_data['is_anomaly']
    prec = precision_score(y_test, ensemble_pred)
    rec = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    logger.info(f"Weighted Voting Ensemble - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'Weighted Voting Ensemble',
        'predictions': ensemble_pred,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1}
    }

def stacking_ensemble(detectors_results, train_data, test_data):
    """Ансамбль на основе стекинга."""
    # Готовим данные для метамодели
    X_train = train_data.drop('is_anomaly', axis=1)
    y_train = train_data['is_anomaly']
    
    X_test = test_data.drop('is_anomaly', axis=1)
    y_test = test_data['is_anomaly']
    
    # Получаем предсказания базовых моделей на обучающих данных
    train_predictions = []
    for detector in detectors_results:
        # Временно обучим детектор на части обучающих данных
        train_ratio = 0.7
        train_part = train_data.sample(frac=train_ratio, random_state=42)
        test_part = train_data.drop(train_part.index)
        
        if detector['name'] == 'Improved Isolation Forest':
            result = improved_isolation_forest(train_part, test_part)
        elif detector['name'] == 'Standard Isolation Forest':
            result = standard_isolation_forest(train_part, test_part)
        elif detector['name'] == 'LOF':
            result = lof_detector(train_part, test_part)
            
        train_predictions.append(result['predictions'])
    
    # Создаем обучающий набор для метамодели
    meta_X_train = np.column_stack(train_predictions)
    meta_y_train = test_part['is_anomaly'].values
    
    # Обучаем метамодель
    logger.info("Обучение метамодели для стекинга...")
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(meta_X_train, meta_y_train)
    
    # Получаем предсказания базовых моделей на тестовых данных
    test_predictions = [detector['predictions'] for detector in detectors_results]
    meta_X_test = np.column_stack(test_predictions)
    
    # Предсказания метамодели
    ensemble_pred = meta_model.predict(meta_X_test)
    
    # Оценка
    prec = precision_score(y_test, ensemble_pred)
    rec = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    logger.info(f"Stacking Ensemble - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    logger.info(f"Важность детекторов: {meta_model.feature_importances_}")
    
    return {
        'name': 'Stacking Ensemble',
        'predictions': ensemble_pred,
        'meta_model': meta_model,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1}
    }

def adaptive_ensemble(detectors_results, test_data):
    """Адаптивный ансамбль, выбирающий лучший детектор для каждого образца."""
    # Получаем предсказания и скоры каждого детектора
    predictions = [detector['predictions'] for detector in detectors_results]
    scores = []
    for detector in detectors_results:
        if 'scores' in detector:
            # Нормализуем скоры в диапазон [0, 1] и инвертируем при необходимости
            score = detector['scores']
            score_min, score_max = np.min(score), np.max(score)
            norm_score = (score - score_min) / (score_max - score_min)
            scores.append(norm_score)
    
    if not scores:
        logger.warning("Нет детекторов со скорами, использую обычное голосование")
        return majority_voting_ensemble(detectors_results, test_data)
    
    # Преобразуем в массив и транспонируем для удобства
    scores = np.array(scores).T
    
    # Для каждого образца выбираем детектор с наибольшей уверенностью
    ensemble_pred = np.zeros(len(test_data))
    for i in range(len(test_data)):
        # Определяем, какие детекторы считают точку аномалией
        anomaly_votes = [pred[i] for pred in predictions]
        
        if np.sum(anomaly_votes) == 0:
            # Если все детекторы считают точку нормальной, оставляем 0
            ensemble_pred[i] = 0
        elif np.sum(anomaly_votes) == len(detectors_results):
            # Если все детекторы считают точку аномалией, ставим 1
            ensemble_pred[i] = 1
        else:
            # В случае разногласий, берем решение детектора с наибольшей уверенностью
            max_score_idx = np.argmax(scores[i])
            ensemble_pred[i] = predictions[max_score_idx][i]
    
    # Оценка
    y_test = test_data['is_anomaly']
    prec = precision_score(y_test, ensemble_pred)
    rec = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    
    logger.info(f"Adaptive Ensemble - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'Adaptive Ensemble',
        'predictions': ensemble_pred,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1}
    }

def cascade_ensemble(detector_high_recall, detector_high_precision, test_data):
    """Каскадный ансамбль: сначала используем детектор с высоким recall, затем уточняем high precision."""
    # Первый этап: получаем все возможные аномалии (высокий recall)
    high_recall_pred = detector_high_recall['predictions']
    
    # Второй этап: из обнаруженных аномалий оставляем только те, которые подтверждает детектор с высокой точностью
    high_precision_pred = detector_high_precision['predictions']
    
    # Если первый детектор считает образец нормальным, доверяем ему
    # Если первый считает аномалией, проверяем вторым
    cascade_pred = np.copy(high_recall_pred)
    for i in range(len(cascade_pred)):
        if high_recall_pred[i] == 1 and high_precision_pred[i] == 0:
            # Если детектор с высоким recall считает точку аномалией, 
            # но детектор с высокой точностью не подтверждает, понижаем уверенность
            cascade_pred[i] = 0
    
    # Оценка
    y_test = test_data['is_anomaly']
    prec = precision_score(y_test, cascade_pred)
    rec = recall_score(y_test, cascade_pred)
    f1 = f1_score(y_test, cascade_pred)
    
    logger.info(f"Cascade Ensemble - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return {
        'name': 'Cascade Ensemble',
        'predictions': cascade_pred,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1}
    }

def visualize_results(all_results):
    """Визуализация сравнения результатов всех детекторов и ансамблей."""
    # Подготовка данных для визуализации
    names = []
    precision_values = []
    recall_values = []
    f1_values = []
    
    for result in all_results:
        names.append(result['name'])
        precision_values.append(result['metrics']['precision'])
        recall_values.append(result['metrics']['recall'])
        f1_values.append(result['metrics']['f1'])
    
    # Сортировка по F1
    sorted_indices = np.argsort(f1_values)[::-1]
    names = [names[i] for i in sorted_indices]
    precision_values = [precision_values[i] for i in sorted_indices]
    recall_values = [recall_values[i] for i in sorted_indices]
    f1_values = [f1_values[i] for i in sorted_indices]
    
    # Построение графика
    plt.figure(figsize=(12, 8))
    bar_width = 0.25
    index = np.arange(len(names))
    
    plt.bar(index, precision_values, bar_width, label='Precision', color='blue', alpha=0.7)
    plt.bar(index + bar_width, recall_values, bar_width, label='Recall', color='green', alpha=0.7)
    plt.bar(index + 2*bar_width, f1_values, bar_width, label='F1 Score', color='red', alpha=0.7)
    
    plt.xlabel('Detector/Ensemble')
    plt.ylabel('Score')
    plt.title('Comparison of Detectors and Ensembles')
    plt.xticks(index + bar_width, names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    
    # Добавляем значения над столбцами
    for i, v in enumerate(precision_values):
        plt.text(i - 0.05, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(recall_values):
        plt.text(i + bar_width - 0.05, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    for i, v in enumerate(f1_values):
        plt.text(i + 2*bar_width - 0.05, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    
    plt.savefig('ensemble_comparison.png')
    plt.show()
    
    # Создаем таблицу результатов
    results_df = pd.DataFrame({
        'Detector/Ensemble': names,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    })
    
    print("\nСравнительная таблица результатов:")
    print(results_df.to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    # Загрузка данных
    sample_size = 10000
    logger.info(f"Загрузка данных NSL-KDD (размер выборки: {sample_size})...")
    train_data, test_data = load_nsl_kdd_data(sample_size=sample_size)
    
    if train_data is None or test_data is None:
        logger.error("Не удалось загрузить данные")
        sys.exit(1)
    
    # Запуск базовых детекторов
    logger.info("Запуск базовых детекторов...")
    improved_iso_result = improved_isolation_forest(train_data, test_data)
    standard_iso_result = standard_isolation_forest(train_data, test_data)
    lof_result = lof_detector(train_data, test_data)
    
    # Сохраняем результаты базовых детекторов
    base_detectors_results = [improved_iso_result, standard_iso_result, lof_result]
    
    # Запуск ансамблевых методов
    logger.info("Запуск ансамблевых методов...")
    majority_ensemble_result = majority_voting_ensemble(base_detectors_results, test_data)
    weighted_ensemble_result = weighted_voting_ensemble(base_detectors_results, test_data)
    adaptive_ensemble_result = adaptive_ensemble(base_detectors_results, test_data)
    
    # Каскадный ансамбль: используем улучшенный Isolation Forest (высокий recall) 
    # и стандартный Isolation Forest (высокая precision)
    cascade_ensemble_result = cascade_ensemble(improved_iso_result, standard_iso_result, test_data)
    
    # Стекинг требует обучения на независимой выборке, поэтому его мы запускаем отдельно
    stacking_ensemble_result = stacking_ensemble(base_detectors_results, train_data, test_data)
    
    # Собираем все результаты вместе
    all_results = base_detectors_results + [
        majority_ensemble_result,
        weighted_ensemble_result,
        adaptive_ensemble_result,
        cascade_ensemble_result,
        stacking_ensemble_result
    ]
    
    # Визуализируем результаты
    results_df = visualize_results(all_results)
    
    logger.info("Тестирование ансамблевых методов завершено!")
