
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
