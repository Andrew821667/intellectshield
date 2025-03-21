
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
