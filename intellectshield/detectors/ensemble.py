import numpy as np
import pandas as pd
from intellectshield.detectors.base import BaseAnomalyDetector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
import time

class EnsembleAnomalyDetector:
    """
    Ансамблевый детектор аномалий, объединяющий результаты нескольких базовых детекторов.

    Подходы к ансамблированию:
    1. Majority voting - голосование большинством (hard voting)
    2. Weighted average - взвешенное среднее оценок аномальности (soft voting)
    3. Rank-based fusion - объединение на основе ранжирования
    4. Stacking - использование метамодели
    """

    def __init__(self, model_dir="models"):
        """
        Инициализация ансамблевого детектора.

        Parameters:
        -----------
        model_dir : str
            Директория для сохранения моделей
        """
        self.detectors = []  # Список базовых детекторов
        self.weights = []    # Веса детекторов
        self.model_dir = model_dir
        self.ensemble_method = "weighted_average"  # По умолчанию используем взвешенное среднее
        self.training_summary = {}
        self.scaler = MinMaxScaler()  # Для нормализации оценок аномальности

        # Создание директории для моделей, если не существует
        import os
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def add_detector(self, detector, weight=1.0):
        """
        Добавление детектора в ансамбль.

        Parameters:
        -----------
        detector : BaseAnomalyDetector
            Детектор аномалий, наследующий от BaseAnomalyDetector
        weight : float
            Вес детектора в ансамбле (по умолчанию 1.0)
        """
        self.detectors.append(detector)
        self.weights.append(weight)

        # Нормализуем веса, чтобы их сумма была равна 1
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print(f"Детектор {detector.__class__.__name__} добавлен с весом {weight / total_weight:.4f}")

        return self

    def set_ensemble_method(self, method):
        """
        Установка метода ансамблирования.

        Parameters:
        -----------
        method : str
            Метод ансамблирования ("majority_voting", "weighted_average", "rank_fusion", "stacking")
        """
        valid_methods = ["majority_voting", "weighted_average", "rank_fusion", "stacking"]

        if method not in valid_methods:
            raise ValueError(f"Недопустимый метод ансамблирования. Допустимые методы: {valid_methods}")

        self.ensemble_method = method
        print(f"Установлен метод ансамблирования: {method}")

        return self

    def train(self, data):
        """
        Обучение всех детекторов в ансамбле.

        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для обучения
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")

        print(f"Начало обучения ансамбля из {len(self.detectors)} детекторов...")
        start_time = time.time()

        # Обучаем каждый детектор
        for i, detector in enumerate(self.detectors):
            print(f"Обучение детектора {i+1}/{len(self.detectors)}: {detector.__class__.__name__}")
            detector.train(data)

        # Если используется stacking, обучаем метамодель
        if self.ensemble_method == "stacking":
            print("Обучение метамодели для stacking...")

            # Получаем предсказания от всех базовых детекторов
            predictions = self._get_all_detector_predictions(data)

            # Если есть метки аномалий, можем обучить метамодель
            if 'is_anomaly' in data.columns:
                # Проверяем на NaN значения
                feature_cols = ['anomaly_score_' + str(i) for i in range(len(self.detectors))]
                X_meta = predictions[feature_cols]

                if X_meta.isna().any().any():
                    print("Предупреждение: обнаружены NaN значения в данных для stacking.")
                    print("Метамодель не будет обучена. Будет использоваться метод взвешенного среднего.")
                    self.ensemble_method = "weighted_average"
                else:
                    try:
                        # Простая логистическая регрессия как метамодель
                        # Подготавливаем обучающий набор
                        y_meta = data['is_anomaly'].values

                        # Обучаем метамодель
                        self.metamodel = LogisticRegression(class_weight='balanced')
                        self.metamodel.fit(X_meta, y_meta)

                        print("Метамодель обучена.")
                    except Exception as e:
                        print(f"Ошибка при обучении метамодели: {e}")
                        print("Метамодель не будет обучена. Будет использоваться метод взвешенного среднего.")
                        self.ensemble_method = "weighted_average"
            else:
                print("Предупреждение: данные не содержат меток аномалий. Метамодель не будет обучена.")
                self.ensemble_method = "weighted_average"
                print(f"Метод ансамблирования изменен на: {self.ensemble_method}")

        # Запись статистики обучения
        training_time = time.time() - start_time
        self.training_summary = {
            'ensemble_method': self.ensemble_method,
            'detectors': [detector.__class__.__name__ for detector in self.detectors],
            'weights': self.weights,
            'training_time': training_time
        }

        print(f"Обучение ансамбля завершено за {training_time:.2f} секунд")

        return self

    def predict(self, data):
        """
        Обнаружение аномалий с использованием ансамбля детекторов.

        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа

        Returns:
        --------
        pandas.DataFrame
            Исходные данные с добавленными предсказаниями и аномальными оценками
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")

        # Получаем предсказания от всех детекторов
        predictions = self._get_all_detector_predictions(data)

        # Применяем выбранный метод ансамблирования
        if self.ensemble_method == "majority_voting":
            # Жесткое голосование большинством
            result = self._apply_majority_voting(predictions)
        elif self.ensemble_method == "weighted_average":
            # Взвешенное среднее оценок аномальности
            result = self._apply_weighted_average(predictions)
        elif self.ensemble_method == "rank_fusion":
            # Объединение на основе ранжирования
            result = self._apply_rank_fusion(predictions)
        elif self.ensemble_method == "stacking":
            # Использование метамодели
            result = self._apply_stacking(predictions)
        else:
            # По умолчанию используем взвешенное среднее
            result = self._apply_weighted_average(predictions)

        # Создаем итоговый результат
        result_df = data.copy()
        result_df['predicted_anomaly'] = result['predicted_anomaly']
        result_df['anomaly_score'] = result['anomaly_score']

        # Добавляем прогнозы и оценки от каждого детектора
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__
            result_df[f'anomaly_{detector_name}'] = predictions[f'predicted_anomaly_{i}']
            result_df[f'score_{detector_name}'] = predictions[f'anomaly_score_{i}']

        # Дополнительная информация о типах аномалий
        result_df = self._add_anomaly_type_info(result_df)

        return result_df

    def _get_all_detector_predictions(self, data):
        """
        Получение предсказаний от всех детекторов в ансамбле.

        Parameters:
        -----------
        data : pandas.DataFrame
            Данные для анализа

        Returns:
        --------
        pandas.DataFrame
            Датафрейм с предсказаниями и оценками от всех детекторов
        """
        all_predictions = pd.DataFrame(index=data.index)

        for i, detector in enumerate(self.detectors):
            # Получаем предсказания от текущего детектора
            detector_result = detector.predict(data)

            # Добавляем предсказания и оценки в общий датафрейм
            all_predictions[f'predicted_anomaly_{i}'] = detector_result['predicted_anomaly']
            all_predictions[f'anomaly_score_{i}'] = detector_result['anomaly_score']

        return all_predictions

    def _apply_majority_voting(self, predictions):
        """
        Применение метода голосования большинством (hard voting).

        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов

        Returns:
        --------
        dict
            Словарь с итоговыми предсказаниями и оценками
        """
        # Получаем предсказания от всех детекторов
        detector_predictions = np.array([predictions[f'predicted_anomaly_{i}'].values
                                       for i in range(len(self.detectors))]).T

        # Веса детекторов для взвешенного голосования
        weights = np.array(self.weights)

        # Считаем взвешенную сумму голосов
        weighted_votes = np.sum(detector_predictions * weights, axis=1)

        # Определяем порог для положительного предсказания (больше половины от суммы весов)
        threshold = 0.5  # Можно настроить

        # Итоговые предсказания
        final_predictions = (weighted_votes > threshold).astype(int)

        # Оценки аномальности как взвешенное среднее оценок всех детекторов
        detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                  for i in range(len(self.detectors))]).T

        final_scores = np.sum(detector_scores * weights, axis=1)

        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }

    def _apply_weighted_average(self, predictions):
        """
        Применение метода взвешенного среднего (soft voting).

        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов

        Returns:
        --------
        dict
            Словарь с итоговыми предсказаниями и оценками
        """
        # Получаем оценки аномальности от всех детекторов
        detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                  for i in range(len(self.detectors))]).T

        # Нормализуем оценки перед взвешиванием
        normalized_scores = np.zeros_like(detector_scores)

        for i in range(detector_scores.shape[1]):
            # Преобразуем оценки в диапазон [0, 1]
            min_score = np.min(detector_scores[:, i])
            max_score = np.max(detector_scores[:, i])

            if max_score > min_score:
                normalized_scores[:, i] = (detector_scores[:, i] - min_score) / (max_score - min_score)
            else:
                normalized_scores[:, i] = 0

        # Веса детекторов
        weights = np.array(self.weights)

        # Взвешенная сумма нормализованных оценок
        final_scores = np.sum(normalized_scores * weights, axis=1)

        # Определяем аномалии на основе оценок
        # Используем адаптивный порог: верхние X% рассматриваются как аномалии
        anomaly_threshold = np.percentile(final_scores, 95)  # Верхние 5%
        final_predictions = (final_scores > anomaly_threshold).astype(int)

        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }

    def _apply_rank_fusion(self, predictions):
        """
        Применение метода слияния на основе ранжирования.

        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов

        Returns:
        --------
        dict
            Словарь с итоговыми предсказаниями и оценками
        """
        n_samples = len(predictions)
        n_detectors = len(self.detectors)

        # Создаем матрицу рангов
        ranks = np.zeros((n_samples, n_detectors))

        for i in range(n_detectors):
            # Получаем оценки аномальности от текущего детектора
            scores = predictions[f'anomaly_score_{i}'].values

            # Ранжируем образцы (большие оценки -> меньшие ранги)
            # argsort возвращает индексы, которые бы отсортировали массив
            # argsort(argsort) дает ранги
            ranks[:, i] = n_samples - np.argsort(np.argsort(scores))

        # Веса детекторов
        weights = np.array(self.weights).reshape(1, -1)

        # Взвешенная сумма рангов
        weighted_ranks = np.sum(ranks * weights, axis=1)

        # Нормализуем ранги для получения оценок аномальности
        min_rank = np.min(weighted_ranks)
        max_rank = np.max(weighted_ranks)

        if max_rank > min_rank:
            final_scores = (weighted_ranks - min_rank) / (max_rank - min_rank)
        else:
            final_scores = np.zeros(n_samples)

        # Определяем аномалии на основе рангов
        # Используем адаптивный порог: верхние 5% рассматриваются как аномалии
        rank_threshold = np.percentile(weighted_ranks, 95)
        final_predictions = (weighted_ranks > rank_threshold).astype(int)

        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }

    def _apply_stacking(self, predictions):
        """
        Применение метода стекинга.

        Parameters:
        -----------
        predictions : pandas.DataFrame
            Предсказания от всех детекторов

        Returns:
        --------
        dict
            Словарь с итоговыми предсказаниями и оценками
        """
        # Проверяем, была ли обучена метамодель
        if not hasattr(self, 'metamodel'):
            print("Предупреждение: метамодель не обучена. Используется метод взвешенного среднего.")
            return self._apply_weighted_average(predictions)

        # Подготавливаем данные для метамодели
        feature_cols = ['anomaly_score_' + str(i) for i in range(len(self.detectors))]
        X_meta = predictions[feature_cols]

        # Проверяем на NaN
        if X_meta.isna().any().any():
            print("Предупреждение: обнаружены NaN значения в данных для stacking.")
            print("Используем метод взвешенного среднего вместо стекинга.")
            return self._apply_weighted_average(predictions)

        try:
            # Получаем предсказания от метамодели
            final_predictions = self.metamodel.predict(X_meta)

            # Получаем вероятности (оценки аномальности)
            if hasattr(self.metamodel, 'predict_proba'):
                probas = self.metamodel.predict_proba(X_meta)
                final_scores = probas[:, 1]  # Вероятность аномального класса
            else:
                # Если метамодель не поддерживает predict_proba, используем взвешенное среднее
                detector_scores = np.array([predictions[f'anomaly_score_{i}'].values
                                          for i in range(len(self.detectors))]).T
                final_scores = np.sum(detector_scores * np.array(self.weights), axis=1)
        except Exception as e:
            print(f"Ошибка при применении стекинга: {e}")
            print("Используем метод взвешенного среднего вместо стекинга.")
            return self._apply_weighted_average(predictions)

        return {
            'predicted_anomaly': final_predictions,
            'anomaly_score': final_scores
        }

    def _add_anomaly_type_info(self, result_df):
        """
        Добавление информации о типах аномалий.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты обнаружения аномалий

        Returns:
        --------
        pandas.DataFrame
            Результаты с добавленной информацией о типах аномалий
        """
        # Инициализируем колонку для типов аномалий
        result_df['anomaly_type'] = 'Normal'

        # Фильтруем только аномалии
        anomalies = result_df[result_df['predicted_anomaly'] == 1]

        if len(anomalies) == 0:
            return result_df

        # Определяем типы аномалий на основе признаков и детекторов

        # 1. DoS атаки
        if 'dos_attack_type' in anomalies.columns:
            dos_attacks = anomalies[anomalies['dos_attack_type'] != 'Normal']
            result_df.loc[dos_attacks.index, 'anomaly_type'] = dos_attacks['dos_attack_type']

        # 2. Анализ на основе признаков трафика

        # Объемные аномалии
        if all(col in result_df.columns for col in ['bytes', 'duration']):
            # Большой объем данных за короткое время
            volume_ratio = result_df['bytes'] / (result_df['duration'] + 0.1)  # +0.1 чтобы избежать деления на 0
            threshold = np.percentile(volume_ratio, 99)

            volume_anomalies = (result_df['predicted_anomaly'] == 1) & (volume_ratio > threshold)
            result_df.loc[volume_anomalies & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Volume Anomaly'

        # Аномалии портов
        if 'dst_port' in result_df.columns:
            unusual_ports = [6667, 31337, 4444, 9001, 1337, 8080]  # Порты, часто используемые для атак
            port_anomalies = (result_df['predicted_anomaly'] == 1) & (result_df['dst_port'].isin(unusual_ports))
            result_df.loc[port_anomalies & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Unusual Port'

        # Сканирование портов
        if all(col in result_df.columns for col in ['packets', 'duration']):
            scan_condition = (result_df['predicted_anomaly'] == 1) & (result_df['packets'] <= 3) & (result_df['duration'] < 0.5)
            result_df.loc[scan_condition & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Port Scan'

        # 3. Для временных аномалий (если есть временная метка)
        if 'timestamp' in result_df.columns:
            try:
                if result_df['timestamp'].dtype != 'datetime64[ns]':
                    result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])

                # Необычное время активности (нерабочее время)
                work_hours = (result_df['timestamp'].dt.hour >= 8) & (result_df['timestamp'].dt.hour <= 18)
                work_days = (result_df['timestamp'].dt.dayofweek < 5)  # Пн-Пт

                time_anomalies = (result_df['predicted_anomaly'] == 1) & (~(work_hours & work_days))
                result_df.loc[time_anomalies & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Unusual Time'
            except:
                # В случае проблем с обработкой времени просто продолжаем без этой информации
                pass

        # 4. Определение специфичных аномалий от разных детекторов
        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__

            if detector_name == 'SequenceAnomalyDetector':
                # Аномалии последовательностей
                sequence_condition = (result_df['predicted_anomaly'] == 1) & (result_df[f'anomaly_{detector_name}'] == 1)
                result_df.loc[sequence_condition & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Sequence Anomaly'

            elif detector_name == 'IsolationForestDetector':
                # Изолированные точки
                isolation_condition = (result_df['predicted_anomaly'] == 1) & (result_df[f'anomaly_{detector_name}'] == 1)
                if isolation_condition.sum() > 0:
                    isolation_score = result_df.loc[isolation_condition, f'score_{detector_name}']

                    # Высокие оценки аномальности от Isolation Forest
                    if len(isolation_score) > 0:
                        percentile = np.percentile(isolation_score, 75) if len(isolation_score) > 4 else isolation_score.max()
                        high_score_condition = isolation_condition & (result_df[f'score_{detector_name}'] > percentile)
                        result_df.loc[high_score_condition & (result_df['anomaly_type'] == 'Normal'), 'anomaly_type'] = 'Isolation Anomaly'

        # Если не удалось определить тип аномалии, помечаем как "Unknown"
        unknown_condition = (result_df['predicted_anomaly'] == 1) & (result_df['anomaly_type'] == 'Normal')
        result_df.loc[unknown_condition, 'anomaly_type'] = 'Unknown Anomaly'

        return result_df

    def evaluate(self, data):
        """
        Оценка производительности ансамблевого детектора.

        Parameters:
        -----------
        data : pandas.DataFrame
            Данные с истинными метками аномалий

        Returns:
        --------
        dict
            Словарь с метриками производительности
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Для оценки необходимы данные с колонкой 'is_anomaly'")

        # Получаем предсказания
        result_df = self.predict(data)

        # Вычисляем метрики
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

        y_true = data['is_anomaly'].values
        y_pred = result_df['predicted_anomaly'].values
        y_score = result_df['anomaly_score'].values

        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred)

        # Precision, Recall, F1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, y_score)
        except:
            auc_roc = None

        # Вычисляем метрики для каждого детектора
        detector_metrics = []

        for i, detector in enumerate(self.detectors):
            detector_name = detector.__class__.__name__

            detector_pred = result_df[f'anomaly_{detector_name}'].values
            detector_score = result_df[f'score_{detector_name}'].values

            # Precision, Recall, F1 для текущего детектора
            d_precision, d_recall, d_f1, _ = precision_recall_fscore_support(y_true, detector_pred, average='binary')

            # AUC-ROC для текущего детектора
            try:
                d_auc_roc = roc_auc_score(y_true, detector_score)
            except:
                d_auc_roc = None

            detector_metrics.append({
                'detector': detector_name,
                'precision': d_precision,
                'recall': d_recall,
                'f1_score': d_f1,
                'auc_roc': d_auc_roc
            })

        # Результаты
        evaluation = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'detector_metrics': detector_metrics,
            'ensemble_method': self.ensemble_method
        }

        return evaluation

    def save_model(self, filename=None):
        """
        Сохранение ансамблевого детектора в файл.

        Parameters:
        -----------
        filename : str
            Имя файла для сохранения
        """
        if not self.detectors:
            raise ValueError("Ансамбль не содержит детекторов. Добавьте детекторы с помощью метода add_detector().")

        # Если имя файла не указано, генерируем его из типа модели и времени
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"EnsembleDetector_{timestamp}.joblib"

        # Полный путь к файлу
        import os
        filepath = os.path.join(self.model_dir, filename)

        # Создаем словарь с моделью и метаданными
        model_data = {
            'detectors': self.detectors,
            'weights': self.weights,
            'ensemble_method': self.ensemble_method,
            'training_summary': self.training_summary,
            'timestamp': datetime.datetime.now().isoformat()
        }

        # Если есть метамодель, сохраняем и ее
        if hasattr(self, 'metamodel'):
            model_data['metamodel'] = self.metamodel

        # Сохраняем в файл
        import joblib
        joblib.dump(model_data, filepath)
        print(f"Ансамблевый детектор сохранен в {filepath}")

        return filepath

    def load_model(self, filepath):
        """
        Загрузка ансамблевого детектора из файла.

        Parameters:
        -----------
        filepath : str
            Путь к файлу с сохраненной моделью
        """
        # Загружаем данные из файла
        import joblib
        model_data = joblib.load(filepath)

        # Загружаем компоненты
        self.detectors = model_data['detectors']
        self.weights = model_data['weights']
        self.ensemble_method = model_data['ensemble_method']
        self.training_summary = model_data['training_summary']

        # Если есть метамодель, загружаем и ее
        if 'metamodel' in model_data:
            self.metamodel = model_data['metamodel']

        print(f"Ансамблевый детектор загружен из {filepath}")
        print(f"Метод ансамблирования: {self.ensemble_method}")
        print(f"Количество детекторов: {len(self.detectors)}")

        return self
