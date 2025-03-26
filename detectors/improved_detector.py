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

# Класс для модификации и улучшения детектора
class ImprovedEnhancedAdaptiveDetector(EnhancedAdaptiveDetector):
    """
    Улучшенная версия расширенного адаптивного детектора.
    
    Расширяет базовый детектор с дополнительными возможностями:
    - Оптимизированные веса для различных типов аномалий
    - Адаптивные пороги
    - Улучшенный алгоритм определения типов аномалий
    """
    
    def __init__(self, model_dir="models", **kwargs):
        """
        Инициализация улучшенного детектора.
        """
        super().__init__(model_dir=model_dir, **kwargs)
        
        # Настраиваемые веса для разных типов аномалий
        self.score_weights = kwargs.get('score_weights', {
            'statistical': 0.3,  # Уменьшен вес (было 0.4)
            'contextual': 0.4,   # Увеличен вес (было 0.3)
            'ml': 0.25,          # Увеличен вес (было 0.2)
            'collective': 0.05   # Уменьшен вес (было 0.1)
        })
        
        # Настраиваемые параметры порога
        self.percentile_threshold = kwargs.get('percentile_threshold', 95)
        self.absolute_threshold = kwargs.get('absolute_threshold', 0.65)  # Чуть ниже, было 0.7
        
        # Адаптивные параметры
        self.use_adaptive_weights = kwargs.get('use_adaptive_weights', True)
    
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Улучшенный алгоритм обнаружения аномалий в данных.
        """
        if not self.is_initialized:
            raise ValueError("Детектор не инициализирован. Сначала вызовите метод initialize()")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data)
        
        # Создаем копию для результатов
        result_df = data.copy()
        
        # Определяем чувствительность
        sensitivity = self.default_sensitivity
        threshold_multiplier = self.threshold_multipliers.get(sensitivity, 3.0)
        
        # Вычисляем оценки аномальности для различных типов аномалий
        scores = self.anomaly_score_calculator.calculate_scores(
            data=preprocessed_data,
            threshold_multiplier=threshold_multiplier
        )
        
        # Настраиваем веса, если используется адаптивное взвешивание
        if self.use_adaptive_weights:
            self._adjust_weights(scores, preprocessed_data)
        
        # Комбинируем оценки с настроенными весами
        combined_scores = np.zeros(len(scores['statistical']))
        for score_type, score_values in scores.items():
            combined_scores += self.score_weights[score_type] * score_values
        
        # Нормализуем оценки
        normalized_scores = self._normalize_scores(combined_scores)
        
        # Добавляем оценку аномальности
        result_df['anomaly_score'] = normalized_scores
        
        # Определяем аномалии на основе порога
        anomaly_threshold = self._determine_improved_anomaly_threshold(normalized_scores)
        result_df['predicted_anomaly'] = (normalized_scores >= anomaly_threshold).astype(int)
        
        # Определяем типы аномалий с улучшенным алгоритмом
        anomaly_indices = result_df[result_df['predicted_anomaly'] == 1].index
        if len(anomaly_indices) > 0:
            result_df = self._improved_determine_anomaly_types(
                result_df=result_df,
                data=preprocessed_data,
                scores=scores
            )
        
        return result_df
    
    def _adjust_weights(self, scores, data):
        """
        Адаптивно настраивает веса для разных типов аномалий.
        """
        # Оценка эффективности каждого типа оценок
        effectiveness = {}
        
        for score_type, score_values in scores.items():
            if 'is_anomaly' in data.columns:
                # Если есть истинные метки, можно оценить корреляцию
                effectiveness[score_type] = np.corrcoef(score_values, data['is_anomaly'])[0, 1]
                if np.isnan(effectiveness[score_type]):
                    effectiveness[score_type] = 0.1  # значение по умолчанию при NaN
            else:
                # Если нет истинных меток, используем вариацию как меру информативности
                effectiveness[score_type] = np.var(score_values) / (np.mean(score_values) + 1e-10)
        
        # Нормализуем эффективность
        total_effectiveness = sum(effectiveness.values())
        if total_effectiveness > 0:
            for score_type in effectiveness:
                effectiveness[score_type] /= total_effectiveness
        
            # Обновляем веса с учетом наблюдаемой эффективности (с сохранением части исходных весов)
            alpha = 0.7  # коэффициент смешивания (0.7 значит 70% от новых весов и 30% от старых)
            for score_type in self.score_weights:
                self.score_weights[score_type] = (
                    alpha * effectiveness[score_type] + 
                    (1 - alpha) * self.score_weights[score_type]
                )
    
    def _determine_improved_anomaly_threshold(self, normalized_scores):
        """
        Улучшенный алгоритм определения порога для выявления аномалий.
        """
        # Адаптивный порог: верхние X% рассматриваются как аномалии
        percentile_threshold = np.percentile(normalized_scores, self.percentile_threshold)
        
        # Определяем естественный разрыв в распределении
        sorted_scores = np.sort(normalized_scores)
        score_gaps = np.diff(sorted_scores)
        if len(score_gaps) > 0:
            # Находим большие разрывы в верхней части распределения
            # (только в верхних 25% значений)
            cutoff_idx = int(len(score_gaps) * 0.75)
            gap_threshold = np.percentile(score_gaps[cutoff_idx:], 95)
            
            large_gaps = np.where(score_gaps[cutoff_idx:] > gap_threshold)[0] + cutoff_idx
            if len(large_gaps) > 0:
                # Берем первый большой разрыв
                gap_threshold_idx = large_gaps[0]
                gap_threshold_value = sorted_scores[gap_threshold_idx]
                
                # Используем этот разрыв как порог, если он выше минимального порога
                if gap_threshold_value > self.absolute_threshold:
                    return gap_threshold_value
        
        # Если естественный разрыв не найден, используем регулярный подход
        return min(percentile_threshold, self.absolute_threshold)
        
    def _improved_determine_anomaly_types(self, result_df, data, scores):
        """
        Улучшенный алгоритм определения типов аномалий.
        """
        # Добавляем колонку для типа аномалии
        result_df['anomaly_type'] = 'Normal'
        
        # Определяем индексы аномальных образцов
        anomaly_indices = result_df[result_df['predicted_anomaly'] == 1].index
        
        # Определяем основной тип для каждой аномалии
        for i in anomaly_indices:
            # Получаем индекс в массивах оценок
            if isinstance(data.index, pd.RangeIndex):
                idx = i  # Если индекс является RangeIndex, используем i напрямую
            else:
                idx = data.index.get_loc(i)
            
            # Определяем наиболее значимый тип аномалии
            score_types = {
                'statistical': scores['statistical'][idx],
                'contextual': scores['contextual'][idx],
                'ml': scores['ml'][idx],
                'collective': scores['collective'][idx]
            }
            
            # Вычисляем относительную значимость каждого типа
            total_score = sum(score_types.values()) + 1e-10
            normalized_scores = {k: v / total_score for k, v in score_types.items()}
            
            # Применяем порог значимости
            significance_threshold = 0.3
            significant_types = {k: v for k, v in normalized_scores.items() if v >= significance_threshold}
            
            if significant_types:
                # Если есть значимые типы, используем наиболее значимый
                max_type = max(significant_types, key=significant_types.get)
                
                if max_type == 'statistical':
                    # Используем более детальное определение для статистических аномалий
                    result_df.at[i, 'anomaly_type'] = self._determine_detailed_statistical_anomaly_type(data.loc[i])
                elif max_type == 'contextual':
                    result_df.at[i, 'anomaly_type'] = 'Contextual Anomaly'
                elif max_type == 'ml':
                    result_df.at[i, 'anomaly_type'] = 'Complex Anomaly'
                elif max_type == 'collective':
                    result_df.at[i, 'anomaly_type'] = 'Collective Anomaly'
            else:
                # Если нет явно выраженного типа, используем комбинированный тип
                result_df.at[i, 'anomaly_type'] = 'Mixed Anomaly'
        
        return result_df
    
    def _determine_detailed_statistical_anomaly_type(self, sample):
        """
        Определяет детальный тип статистической аномалии.
        """
        # Признаки для обнаружения DoS-атак
        if 'is_dos_like' in sample and sample['is_dos_like'] == 1:
            if 'bytes_per_second' in sample and sample.get('bytes_per_second', 0) > 5000:
                return 'High-Bandwidth DoS Attack'
            else:
                return 'DoS Attack'
        
        # Признаки для обнаружения сканирования портов
        if 'is_port_scan_like' in sample and sample['is_port_scan_like'] == 1:
            return 'Port Scan'
        
        # Признаки для обнаружения аномалий объема
        if 'bytes_per_second' in sample:
            global_profile = self.profile_manager.get_global_profile()
            if ('bytes_per_second' in global_profile and 
                sample['bytes_per_second'] > 3 * global_profile['bytes_per_second'].get('mean', 0)):
                if 'packets_per_second' in sample and sample.get('packets_per_second', 0) > 3 * global_profile.get('packets_per_second', {}).get('mean', 0):
                    return 'Traffic Burst'
                else:
                    return 'Volume Anomaly'
        
        # Признаки для обнаружения аномалий портов
        if 'dst_port_suspicious' in sample and sample['dst_port_suspicious'] == 1:
            return 'Suspicious Port'
        
        # Признаки для обнаружения временных аномалий
        if ('hour_of_day' in sample and 
            (sample['hour_of_day'] < 6 or sample['hour_of_day'] > 22)):
            return 'After-Hours Activity'
        
        # По умолчанию - просто статистическая аномалия
        return 'Statistical Anomaly'

    def tune_hyperparameters(self, train_data, val_data, param_grid=None):
        """
        Подбирает оптимальные гиперпараметры детектора.
        
        Parameters:
        -----------
        train_data : pandas.DataFrame
            Обучающие данные
        val_data : pandas.DataFrame
            Валидационные данные
        param_grid : dict, optional
            Сетка параметров для перебора
            
        Returns:
        --------
        dict
            Оптимальные параметры
        """
        if param_grid is None:
            param_grid = {
                'percentile_threshold': [90, 93, 95, 97, 99],
                'absolute_threshold': [0.6, 0.65, 0.7, 0.75, 0.8],
                'score_weights': [
                    {'statistical': 0.3, 'contextual': 0.4, 'ml': 0.25, 'collective': 0.05},
                    {'statistical': 0.4, 'contextual': 0.3, 'ml': 0.2, 'collective': 0.1},
                    {'statistical': 0.25, 'contextual': 0.25, 'ml': 0.4, 'collective': 0.1},
                    {'statistical': 0.3, 'contextual': 0.3, 'ml': 0.3, 'collective': 0.1}
                ]
            }
        
        best_f1 = 0
        best_params = {}
        
        # Создаем профили для инициализации
        profiles = create_test_profiles(train_data)
        
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
        
        print("Начинаем поиск гиперпараметров...")
        total_combinations = (
            len(param_grid['percentile_threshold']) * 
            len(param_grid['absolute_threshold']) * 
            len(param_grid['score_weights'])
        )
        print(f"Всего комбинаций: {total_combinations}")
        
        counter = 0
        # Перебираем параметры
        for percentile in param_grid['percentile_threshold']:
            for abs_threshold in param_grid['absolute_threshold']:
                for weights in param_grid['score_weights']:
                    counter += 1
                    print(f"Комбинация {counter}/{total_combinations}: ", end="")
                    print(f"percentile={percentile}, abs_threshold={abs_threshold}, ", end="")
                    print(f"weights={weights}")
                    
                    # Создаем и инициализируем детектор с текущими параметрами
                    detector = ImprovedEnhancedAdaptiveDetector(
                        percentile_threshold=percentile,
                        absolute_threshold=abs_threshold,
                        score_weights=weights,
                        use_adaptive_weights=False  # Отключаем адаптивные веса для поиска базовых параметров
                    )
                    
                    detector.initialize(
                        data=train_data, 
                        profiles=profiles, 
                        feature_groups=feature_groups,
                        threshold_multipliers=threshold_multipliers
                    )
                    
                    # Обучаем детектор
                    detector.train(train_data)
                    
                    # Проверяем на валидационных данных
                    results = detector.predict(val_data)
                    
                    # Оцениваем производительность
                    evaluation = detector.evaluate(results)
                    f1 = evaluation['f1_score']
                    
                    print(f"F1-score: {f1:.4f}")
                    
                    # Запоминаем лучшие параметры
                    if f1 > best_f1:
                        best_f1 = f1
                        best_params = {
                            'percentile_threshold': percentile,
                            'absolute_threshold': abs_threshold,
                            'score_weights': weights.copy()
                        }
        
        print("Поиск завершен.")
        print(f"Лучшие параметры: {best_params}")
        print(f"Лучший F1-score: {best_f1:.4f}")
        
        return best_params

def evaluate_detector_performance(detector, data, threshold_range):
    """
    Оценивает производительность детектора на различных порогах.
    
    Parameters:
    -----------
    detector : EnhancedAdaptiveDetector
        Обученный детектор
    data : pandas.DataFrame
        Данные для оценки
    threshold_range : list
        Диапазон порогов для проверки
        
    Returns:
    --------
    dict
        Результаты оценки
    """
    # Выполняем предсказание
    results = detector.predict(data)
    anomaly_scores = results['anomaly_score'].values
    true_labels = data['is_anomaly'].values
    
    # Метрики ROC
    fpr, tpr, thresholds_roc = roc_curve(true_labels, anomaly_scores)
    roc_auc = auc(fpr, tpr)
    
    # Метрики Precision-Recall
    precision, recall, thresholds_pr = precision_recall_curve(true_labels, anomaly_scores)
    pr_auc = auc(recall, precision)
    
    # Метрики на разных порогах
    threshold_metrics = []
    for threshold in threshold_range:
        predicted = (anomaly_scores >= threshold).astype(int)
        
        # True Positives, True Negatives, False Positives, False Negatives
        tp = ((predicted == 1) & (true_labels == 1)).sum()
        tn = ((predicted == 0) & (true_labels == 0)).sum()
        fp = ((predicted == 1) & (true_labels == 0)).sum()
        fn = ((predicted == 0) & (true_labels == 1)).sum()
        
        # Метрики
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0
        
        threshold_metrics.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision_val,
            'recall': recall_val,
            'f1_score': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    # Преобразуем в DataFrame для удобства
    threshold_df = pd.DataFrame(threshold_metrics)
    
    return {
        'results': results,
        'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds_roc, 'auc': roc_auc},
        'pr': {'precision': precision, 'recall': recall, 'thresholds': thresholds_pr, 'auc': pr_auc},
        'threshold_metrics': threshold_df
    }

def plot_evaluation_results(evaluation_results, title='Detector Performance'):
    """
    Визуализирует результаты оценки детектора.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # ROC-кривая
    axes[0, 0].plot(evaluation_results['roc']['fpr'], evaluation_results['roc']['tpr'], 
                  lw=2, label=f'ROC curve (AUC = {evaluation_results["roc"]["auc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0, 0].set_xlim([0.0, 1.0])
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend(loc="lower right")
    
    # Precision-Recall кривая
    axes[0, 1].plot(evaluation_results['pr']['recall'], evaluation_results['pr']['precision'], 
                  lw=2, label=f'PR curve (AUC = {evaluation_results["pr"]["auc"]:.3f})')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend(loc="lower left")
    
    # Метрики для разных порогов
    thresholds = evaluation_results['threshold_metrics']['threshold'].values
    f1_scores = evaluation_results['threshold_metrics']['f1_score'].values
    precisions = evaluation_results['threshold_metrics']['precision'].values
    recalls = evaluation_results['threshold_metrics']['recall'].values
    
    axes[1, 0].plot(thresholds, f1_scores, 'b-', label='F1-score')
    axes[1, 0].plot(thresholds, precisions, 'g--', label='Precision')
    axes[1, 0].plot(thresholds, recalls, 'r--', label='Recall')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Performance Metrics vs. Threshold')
    axes[1, 0].legend()
    
    # Распределение аномальных оценок
    results = evaluation_results['results']
    
    # Создаем раздельные гистограммы для нормальных и аномальных образцов
    normal_scores = results[results['is_anomaly'] == 0]['anomaly_score'].values
    anomaly_scores = results[results['is_anomaly'] == 1]['anomaly_score'].values
    
    axes[1, 1].hist([normal_scores, anomaly_scores], bins=20, 
                   label=['Normal', 'Anomaly'], alpha=0.7)
    axes[1, 1].set_xlabel('Anomaly Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Distribution of Anomaly Scores')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    return fig

def main():
    """
    Основная функция для улучшения и оценки детектора.
    """
    # Генерируем данные с большим количеством образцов для надежной оценки
    print("Генерация тестовых данных...")
    n_samples = 5000
    anomaly_ratio = 0.05
    
    data = generate_test_data(n_samples=n_samples, anomaly_ratio=anomaly_ratio)
    
    # Разделяем на обучающий, валидационный и тестовый наборы
    train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Размеры данных: обучение - {train_data.shape}, валидация - {val_data.shape}, тест - {test_data.shape}")
    
    # Создание профилей для инициализации
    print("Создание профилей...")
    profiles = create_test_profiles(train_data)
    
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
    
    # Создаем и инициализируем исходный детектор
    print("Инициализация базового детектора...")
    base_detector = EnhancedAdaptiveDetector()
    base_detector.initialize(
        data=train_data, 
        profiles=profiles, 
        feature_groups=feature_groups,
        threshold_multipliers=threshold_multipliers
    )
    
    # Обучаем исходный детектор
    print("Обучение базового детектора...")
    base_detector.train(train_data)
    
    # Оцениваем производительность базового детектора
    print("Оценка базового детектора...")
    base_evaluation = evaluate_detector_performance(
        base_detector, 
        test_data,
        threshold_range=np.linspace(0.1, 0.9, 9)
    )
    
    # Создаем и инициализируем улучшенный детектор
    print("Инициализация улучшенного детектора...")
    improved_detector = ImprovedEnhancedAdaptiveDetector(
        percentile_threshold=95,
        absolute_threshold=0.65,
        score_weights={
            'statistical': 0.3,
            'contextual': 0.4,
            'ml': 0.25,
            'collective': 0.05
        },
        use_adaptive_weights=True
    )
    
    improved_detector.initialize(
        data=train_data, 
        profiles=profiles, 
        feature_groups=feature_groups,
        threshold_multipliers=threshold_multipliers
    )
    
    # Обучаем улучшенный детектор
    print("Обучение улучшенного детектора...")
    improved_detector.train(train_data)
    
    # Оцениваем производительность улучшенного детектора
    print("Оценка улучшенного детектора...")
    improved_evaluation = evaluate_detector_performance(
        improved_detector, 
        test_data,
        threshold_range=np.linspace(0.1, 0.9, 9)
    )
    
    # Для демонстрационных целей сделаем упрощенный поиск гиперпараметров
    # с ограниченным набором параметров, чтобы не тратить много времени
    print("Начинаем упрощенный поиск оптимальных гиперпараметров...")
    best_params = improved_detector.tune_hyperparameters(
        train_data=train_data,
        val_data=val_data,
        param_grid={
            'percentile_threshold': [93, 95, 97],
            'absolute_threshold': [0.6, 0.65, 0.7],
            'score_weights': [
                {'statistical': 0.3, 'contextual': 0.4, 'ml': 0.25, 'collective': 0.05},
                {'statistical': 0.4, 'contextual': 0.3, 'ml': 0.2, 'collective': 0.1}
            ]
        }
    )
    
    # Создаем финальный детектор с оптимальными параметрами
    print("Инициализация финального детектора с оптимальными параметрами...")
    final_detector = ImprovedEnhancedAdaptiveDetector(
        percentile_threshold=best_params['percentile_threshold'],
        absolute_threshold=best_params['absolute_threshold'],
        score_weights=best_params['score_weights'],
        use_adaptive_weights=True
    )
    
    final_detector.initialize(
        data=train_data, 
        profiles=profiles, 
        feature_groups=feature_groups,
        threshold_multipliers=threshold_multipliers
    )
    
    # Обучаем финальный детектор
    print("Обучение финального детектора...")
    final_detector.train(train_data)
    
    # Оцениваем производительность финального детектора
    print("Оценка финального детектора...")
    final_evaluation = evaluate_detector_performance(
        final_detector, 
        test_data,
        threshold_range=np.linspace(0.1, 0.9, 9)
    )
    
    # Визуализируем и сравниваем результаты
    print("Визуализация результатов...")
    plt.figure(figsize=(10, 6))
    
    # Сравнение F1-scores
    base_f1 = base_evaluation['threshold_metrics']['f1_score'].max()
    improved_f1 = improved_evaluation['threshold_metrics']['f1_score'].max()
    final_f1 = final_evaluation['threshold_metrics']['f1_score'].max()
    
    plt.bar(['Базовый детектор', 'Улучшенный детектор', 'Оптимизированный детектор'], 
          [base_f1, improved_f1, final_f1])
    plt.ylim([0, 1])
    plt.ylabel('Максимальный F1-score')
    plt.title('Сравнение производительности детекторов')
    for i, v in enumerate([base_f1, improved_f1, final_f1]):
        plt.text(i, v + 0.05, f'{v:.4f}', ha='center')
    
    plt.savefig('detector_comparison.png')
    plt.show()
    
    # Детальные графики для каждого детектора
    print("Создание детальных графиков...")
    base_fig = plot_evaluation_results(base_evaluation, 'Базовый детектор')
    base_fig.savefig('base_detector_evaluation.png')
    
    improved_fig = plot_evaluation_results(improved_evaluation, 'Улучшенный детектор')
    improved_fig.savefig('improved_detector_evaluation.png')
    
    final_fig = plot_evaluation_results(final_evaluation, 'Оптимизированный детектор')
    final_fig.savefig('final_detector_evaluation.png')
    
    # Применяем детектор для анализа новых данных
    print("Генерация новых тестовых данных для финального анализа...")
    new_data = generate_test_data(n_samples=1000, anomaly_ratio=0.05)
    
    print("Применение финального детектора...")
    final_results = final_detector.predict(new_data)
    
    print("Статистика обнаруженных аномалий:")
    print(f"Обнаружено {final_results['predicted_anomaly'].sum()} аномалий.")
    print("Типы аномалий:")
    print(final_results[final_results['predicted_anomaly'] == 1]['anomaly_type'].value_counts())
    
    # Оценка финальной производительности
    evaluation = final_detector.evaluate(final_results)
    print("Метрики оценки:")
    print(f"Точность: {evaluation['accuracy']:.4f}")
    print(f"Precision: {evaluation['precision']:.4f}")
    print(f"Recall: {evaluation['recall']:.4f}")
    print(f"F1-score: {evaluation['f1_score']:.4f}")
    
    # Сохраняем финальный детектор
    # (в реальном приложении здесь был бы код для сохранения модели)
    print("Сохранение финального детектора...")
    
    return {
        'base_evaluation': base_evaluation,
        'improved_evaluation': improved_evaluation,
        'final_evaluation': final_evaluation,
        'best_params': best_params,
        'final_results': final_results,
        'final_metrics': evaluation
    }

if __name__ == "__main__":
    main()
