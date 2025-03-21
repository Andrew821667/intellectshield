
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
