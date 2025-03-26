
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve
import time

class EnsembleStrategy:
    """Базовый класс для различных стратегий объединения результатов детекторов"""
    
    def combine(self, detector_results, weights=None):
        """Объединение результатов нескольких детекторов"""
        raise NotImplementedError("Должен быть реализован в дочернем классе")

class WeightedVotingStrategy(EnsembleStrategy):
    """Стратегия взвешенного голосования (soft voting)"""
    
    def combine(self, detector_results, weights=None):
        """Объединение результатов детекторов методом взвешенного голосования"""
        if not detector_results:
            raise ValueError("Список результатов детекторов пуст")
        
        # Если веса не указаны, используем равные веса
        if weights is None:
            weights = [1.0] * len(detector_results)
        
        # Нормализуем веса
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Базовые результаты (копируем из первого детектора)
        base_result = detector_results[0].copy()
        
        # Получаем предсказания и оценки от всех детекторов
        all_predictions = []
        all_scores = []
        
        for result in detector_results:
            all_predictions.append(result['predicted_anomaly'].values)
            all_scores.append(result['anomaly_score'].values)
        
        # Вычисляем взвешенные оценки аномальности
        weighted_scores = np.zeros(len(base_result))
        for i, weight in enumerate(normalized_weights):
            weighted_scores += all_scores[i] * weight
        
        # Определяем порог для предсказаний (можно настроить)
        threshold = 0.5
        
        # Формируем предсказания на основе взвешенных оценок
        weighted_predictions = (weighted_scores >= threshold).astype(int)
        
        # Обновляем базовые результаты
        base_result['predicted_anomaly'] = weighted_predictions
        base_result['anomaly_score'] = weighted_scores
        base_result['ensemble_method'] = 'weighted_voting'
        
        return base_result

class RankFusionStrategy(EnsembleStrategy):
    """Стратегия слияния на основе рангов (rank-based fusion)"""
    
    def combine(self, detector_results, weights=None):
        """Объединение результатов детекторов методом слияния рангов"""
        if not detector_results:
            raise ValueError("Список результатов детекторов пуст")
        
        # Если веса не указаны, используем равные веса
        if weights is None:
            weights = [1.0] * len(detector_results)
            
        # Нормализуем веса
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Базовые результаты (копируем из первого детектора)
        base_result = detector_results[0].copy()
        n_samples = len(base_result)
        
        # Получаем оценки от всех детекторов и преобразуем их в ранги
        all_ranks = []
        
        for result in detector_results:
            scores = result['anomaly_score'].values
            # Преобразуем оценки в ранги (higher score -> higher rank)
            ranks = n_samples - np.argsort(np.argsort(scores))
            all_ranks.append(ranks)
        
        # Вычисляем взвешенные ранги
        weighted_ranks = np.zeros(n_samples)
        for i, weight in enumerate(normalized_weights):
            weighted_ranks += all_ranks[i] * weight
        
        # Нормализуем ранги для получения оценок аномальности
        min_rank = np.min(weighted_ranks)
        max_rank = np.max(weighted_ranks)
        
        if max_rank > min_rank:
            normalized_scores = (weighted_ranks - min_rank) / (max_rank - min_rank)
        else:
            normalized_scores = np.zeros(n_samples)
        
        # Определяем порог для ранговых оценок (верхние 5% считаем аномалиями)
        rank_threshold = np.percentile(weighted_ranks, 95)
        
        # Формируем предсказания на основе взвешенных рангов
        rank_predictions = (weighted_ranks > rank_threshold).astype(int)
        
        # Обновляем базовые результаты
        base_result['predicted_anomaly'] = rank_predictions
        base_result['anomaly_score'] = normalized_scores
        base_result['ensemble_method'] = 'rank_fusion'
        
        return base_result

class StackingStrategy(EnsembleStrategy):
    """Стратегия стэкинга (meta-learning)"""
    
    def __init__(self, meta_model=None):
        """Инициализация стратегии стэкинга"""
        self.meta_model = meta_model or RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'
        )
        self.is_fitted = False
        self.fallback_strategy = WeightedVotingStrategy()
    
    def fit(self, detector_results, true_labels):
        """Обучение мета-модели"""
        if not detector_results:
            raise ValueError("Список результатов детекторов пуст")
        
        # Подготовка обучающих данных для мета-модели
        X_meta = np.column_stack([result['anomaly_score'].values for result in detector_results])
        
        try:
            # Обучение мета-модели
            self.meta_model.fit(X_meta, true_labels)
            self.is_fitted = True
            print("Мета-модель успешно обучена")
        except Exception as e:
            print(f"Ошибка при обучении мета-модели: {e}")
            self.is_fitted = False
        
        return self
    
    def combine(self, detector_results, weights=None):
        """Объединение результатов детекторов методом стэкинга"""
        if not detector_results:
            raise ValueError("Список результатов детекторов пуст")
        
        # Если мета-модель не обучена, используем запасную стратегию
        if not self.is_fitted:
            print("Мета-модель не обучена. Используется стратегия взвешенного голосования.")
            return self.fallback_strategy.combine(detector_results, weights)
        
        # Базовые результаты (копируем из первого детектора)
        base_result = detector_results[0].copy()
        
        # Формирование данных для мета-модели
        try:
            X_meta = np.column_stack([result['anomaly_score'].values for result in detector_results])
            
            # Предсказания мета-модели
            predictions = self.meta_model.predict(X_meta)
            
            # Вероятности классов (для оценки аномальности)
            if hasattr(self.meta_model, 'predict_proba'):
                probas = self.meta_model.predict_proba(X_meta)
                scores = probas[:, 1]  # Вероятность положительного класса
            else:
                # Если модель не поддерживает вероятности, используем бинарные предсказания
                scores = predictions.astype(float)
            
            # Обновляем базовые результаты
            base_result['predicted_anomaly'] = predictions
            base_result['anomaly_score'] = scores
            base_result['ensemble_method'] = 'stacking'
            
            return base_result
            
        except Exception as e:
            print(f"Ошибка при применении стэкинга: {e}")
            print("Используется стратегия взвешенного голосования.")
            return self.fallback_strategy.combine(detector_results, weights)

class ImprovedEnsembleDetector:
    """
    Улучшенный ансамблевый детектор аномалий
    
    Этот детектор объединяет результаты нескольких базовых детекторов,
    используя различные стратегии ансамблирования.
    """
    
    def __init__(self, ensemble_method='weighted_voting'):
        """
        Инициализация ансамблевого детектора
        
        Args:
            ensemble_method: Метод ансамблирования
                'weighted_voting': Взвешенное голосование
                'stacking': Стэкинг (мета-обучение)
                'rank_fusion': Слияние на основе рангов
                'auto': Автоматический выбор
        """
        self.detectors = []
        self.detector_names = []
        self.weights = []
        self.ensemble_method = ensemble_method
        
        # Словарь стратегий ансамблирования
        self.strategies = {
            'weighted_voting': WeightedVotingStrategy(),
            'stacking': StackingStrategy(),
            'rank_fusion': RankFusionStrategy()
        }
        
        # Оптимальная стратегия (для автоматического выбора)
        self.optimal_strategy = None
        
        print(f"Инициализирован ансамблевый детектор с методом: {ensemble_method}")
    
    def add_detector(self, detector, name, weight=1.0):
        """
        Добавление детектора в ансамбль
        
        Args:
            detector: Объект детектора аномалий
            name: Имя детектора
            weight: Вес детектора в ансамбле
            
        Returns:
            self: Для цепочки вызовов
        """
        self.detectors.append(detector)
        self.detector_names.append(name)
        self.weights.append(weight)
        
        # Нормализуем веса
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        print(f"Добавлен детектор: {name} с весом: {weight/total_weight:.4f}")
        
        return self
    
    def train(self, data, labels=None):
        """
        Обучение всех детекторов в ансамбле
        
        Args:
            data: DataFrame с обучающими данными
            labels: Истинные метки аномалий (если есть)
            
        Returns:
            self: Обученный ансамблевый детектор
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Используйте метод add_detector().")
        
        start_time = time.time()
        print(f"Начало обучения ансамбля из {len(self.detectors)} детекторов...")
        
        # Обучаем каждый детектор
        for i, detector in enumerate(self.detectors):
            detector_name = self.detector_names[i]
            print(f"Обучение детектора {i+1}/{len(self.detectors)}: {detector_name}")
            
            try:
                # Обучаем детектор
                detector.train(data)
            except Exception as e:
                print(f"Ошибка при обучении детектора {detector_name}: {e}")
        
        # Если метод ансамблирования - стэкинг и есть истинные метки,
        # обучаем мета-модель
        if self.ensemble_method == 'stacking' and labels is not None:
            self._train_stacking_model(data, labels)
        
        # Если метод - автоматический выбор и есть истинные метки,
        # выбираем оптимальную стратегию
        elif self.ensemble_method == 'auto' and labels is not None:
            self._select_optimal_strategy(data, labels)
        
        training_time = time.time() - start_time
        print(f"Обучение ансамбля завершено за {training_time:.2f} секунд")
        
        return self
    
    def _train_stacking_model(self, data, labels):
        """
        Обучение мета-модели для стэкинга
        
        Args:
            data: DataFrame с обучающими данными
            labels: Истинные метки аномалий
        """
        print("Обучение мета-модели для стэкинга...")
        
        # Получаем предсказания от всех базовых детекторов
        detector_results = []
        
        for detector in self.detectors:
            try:
                result = detector.predict(data)
                detector_results.append(result)
            except Exception as e:
                print(f"Ошибка при получении предсказаний для мета-модели: {e}")
                return
        
        try:
            # Обучаем мета-модель с использованием стратегии стэкинга
            stacking_strategy = self.strategies['stacking']
            stacking_strategy.fit(detector_results, labels)
        except Exception as e:
            print(f"Ошибка при обучении мета-модели: {e}")
    
    def _select_optimal_strategy(self, data, labels):
        """
        Выбор оптимальной стратегии ансамблирования на основе
        перекрестной проверки
        
        Args:
            data: DataFrame с обучающими данными
            labels: Истинные метки аномалий
        """
        print("Выбор оптимальной стратегии ансамблирования...")
        
        # Получаем предсказания от всех базовых детекторов
        detector_results = []
        
        for detector in self.detectors:
            try:
                result = detector.predict(data)
                detector_results.append(result)
            except Exception as e:
                print(f"Ошибка при получении предсказаний для выбора стратегии: {e}")
                return
        
        # Оцениваем каждую стратегию
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            try:
                # Специальная обработка для стэкинга (требуется обучение)
                if name == 'stacking':
                    stacking_strategy = self.strategies['stacking']
                    stacking_strategy.fit(detector_results, labels)
                
                # Применяем стратегию
                result = strategy.combine(detector_results, self.weights)
                
                # Оцениваем производительность
                predictions = result['predicted_anomaly'].values
                scores = result['anomaly_score'].values
                
                # Расчет AUC и F1
                try:
                    auc = roc_auc_score(labels, scores)
                    # Оптимальный порог по F1
                    precision, recall, thresholds = precision_recall_curve(labels, scores)
                    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                    best_f1 = np.max(f1_scores)
                    
                    # Комбинированная оценка
                    combined_score = 0.7 * auc + 0.3 * best_f1
                    
                    strategy_scores[name] = combined_score
                    print(f"Оценка стратегии {name}: AUC={auc:.4f}, F1={best_f1:.4f}, "
                          f"Комбинированная={combined_score:.4f}")
                except Exception as e:
                    print(f"Ошибка при оценке стратегии {name}: {e}")
                    strategy_scores[name] = 0
            except Exception as e:
                print(f"Ошибка при применении стратегии {name}: {e}")
                strategy_scores[name] = 0
        
        # Выбираем стратегию с наилучшей оценкой
        if strategy_scores:
            best_strategy = max(strategy_scores, key=strategy_scores.get)
            self.optimal_strategy = best_strategy
            print(f"Выбрана оптимальная стратегия: {best_strategy} "
                  f"с оценкой {strategy_scores[best_strategy]:.4f}")
        else:
            # По умолчанию используем взвешенное голосование
            self.optimal_strategy = 'weighted_voting'
            print("Не удалось выбрать оптимальную стратегию. "
                  "Будет использовано взвешенное голосование.")
