import numpy as np
import pandas as pd
from intellectshield.detectors.base import BaseAnomalyDetector
import time

class DoSDetector(BaseAnomalyDetector):
    """
    Специализированный детектор для обнаружения DoS-атак.
    """
    
    def __init__(self, model_dir="models"):
        """
        Инициализация детектора DoS-атак.
        """
        super().__init__(model_dir)
        self.time_window = 30  # Временное окно в секундах
        self.thresholds = {}   # Пороги для определения DoS
        
    def preprocess_data(self, data, train=False):
        """
        Предобработка данных для обнаружения DoS-атак.
        """
        # Создаем копию данных для обработки
        df = data.copy()
        
        # Проверяем наличие временной метки
        if 'timestamp' not in df.columns:
            # Если нет временной метки, создаем фиктивную на основе индекса
            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1S')
        elif df['timestamp'].dtype != 'datetime64[ns]':
            # Если временная метка есть, но не в формате datetime, преобразуем
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Создаем признаки для обнаружения DoS
        
        # 1. Группировка по IP-адресам
        if 'src_ip' in df.columns:
            # Группируем по IP-адресам источников
            df['timestamp_window'] = df['timestamp'].dt.floor(f'{self.time_window}S')
            
            # Подсчитываем количество соединений от каждого IP
            ip_counts = df.groupby(['timestamp_window', 'src_ip']).size().reset_index(name='connection_count')
            
            # Объединяем обратно с исходными данными
            df = pd.merge(df, ip_counts, on=['timestamp_window', 'src_ip'], how='left')
        else:
            # Если нет IP-адресов, используем условные идентификаторы
            df['connection_count'] = 1
        
        # 2. Подсчет количества пакетов
        if 'packets' in df.columns:
            # Уже есть количество пакетов
            pass
        else:
            # Если нет, предполагаем одно соединение = один пакет
            df['packets'] = 1
        
        # 3. Вычисляем интенсивность трафика
        df['traffic_intensity'] = df['connection_count'] * df['packets']
        
        # 4. Признаки для обнаружения SYN-флуда
        if 'flag' in df.columns:
            # Проверяем наличие SYN-флагов (характерно для KDD Cup)
            df['syn_flag'] = df['flag'].apply(lambda x: 1 if 'S0' in str(x) or 'SYN' in str(x) else 0)
        else:
            df['syn_flag'] = 0
        
        # 5. Признаки для обнаружения ICMP-флуда
        if 'protocol_type' in df.columns:
            df['icmp_protocol'] = df['protocol_type'].apply(lambda x: 1 if 'icmp' in str(x).lower() else 0)
        elif 'protocol' in df.columns:
            df['icmp_protocol'] = df['protocol'].apply(lambda x: 1 if x == 1 or 'icmp' in str(x).lower() else 0)
        else:
            df['icmp_protocol'] = 0
        
        # 6. Признаки для обнаружения UDP-флуда
        if 'protocol_type' in df.columns:
            df['udp_protocol'] = df['protocol_type'].apply(lambda x: 1 if 'udp' in str(x).lower() else 0)
        elif 'protocol' in df.columns:
            df['udp_protocol'] = df['protocol'].apply(lambda x: 1 if x == 17 or 'udp' in str(x).lower() else 0)
        else:
            df['udp_protocol'] = 0
        
        # Сохраняем признаки для последующего использования
        self.features = ['connection_count', 'traffic_intensity', 'syn_flag', 'icmp_protocol', 'udp_protocol']
        
        return df
    
    def train(self, data, percentile=99.9, time_window=30):
        """
        Обучение модели обнаружения DoS-атак.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные сетевого трафика для обучения
        percentile : float
            Перцентиль для определения порогов (по умолчанию 99.9)
        time_window : int
            Размер временного окна в секундах (по умолчанию 30)
            
        Returns:
        --------
        self
            Обученный детектор DoS-атак
        """
        print("Начало обучения специализированного детектора DoS-атак...")
        start_time = time.time()
        
        # Сохраняем параметры
        self.time_window = time_window
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=True)
        
        # Вычисляем пороги для каждого признака
        self.thresholds = {}
        
        for feature in self.features:
            if feature in preprocessed_data.columns:
                threshold = preprocessed_data[feature].quantile(percentile / 100)
                self.thresholds[feature] = threshold
                print(f"Порог для {feature}: {threshold:.4f}")
        
        # Запись статистики обучения
        training_time = time.time() - start_time
        self.training_summary = {
            'model_type': 'DoSDetector',
            'time_window': time_window,
            'percentile': percentile,
            'thresholds': self.thresholds,
            'training_samples': len(preprocessed_data),
            'training_time': training_time
        }
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return self
    
    def predict(self, data):
        """
        Обнаружение DoS-атак в данных.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Данные сетевого трафика для анализа
            
        Returns:
        --------
        pandas.DataFrame
            Исходные данные с добавленными предсказаниями и аномальными оценками
        """
        if not self.thresholds:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=False)
        
        # Вычисление оценок аномальности
        anomaly_scores = np.zeros(len(preprocessed_data))
        
        for feature in self.features:
            if feature in preprocessed_data.columns and feature in self.thresholds:
                # Вычисляем, насколько значение превышает порог
                excess = preprocessed_data[feature] / self.thresholds[feature]
                # Значения ниже порога не увеличивают оценку аномальности
                feature_scores = np.maximum(0, excess - 1)
                # Добавляем к общей оценке
                anomaly_scores += feature_scores.values
        
        # Нормализация оценок
        if np.max(anomaly_scores) > 0:
            anomaly_scores = anomaly_scores / np.max(anomaly_scores)
        
        # Определяем аномалии (значения выше нуля)
        anomalies = (anomaly_scores > 0).astype(int)
        
        # Создаем результирующую таблицу
        result_df = data.copy()
        result_df['predicted_anomaly'] = anomalies
        result_df['anomaly_score'] = anomaly_scores
        
        # Добавляем типы обнаруженных DoS-атак
        result_df['dos_attack_type'] = 'Normal'
        
        # SYN-флуд
        if 'syn_flag' in preprocessed_data.columns:
            syn_flood = (anomalies == 1) & (preprocessed_data['syn_flag'] > self.thresholds.get('syn_flag', 0))
            result_df.loc[syn_flood, 'dos_attack_type'] = 'SYN Flood'
        
        # ICMP-флуд
        if 'icmp_protocol' in preprocessed_data.columns:
            icmp_flood = (anomalies == 1) & (preprocessed_data['icmp_protocol'] > 0)
            result_df.loc[icmp_flood, 'dos_attack_type'] = 'ICMP Flood'
        
        # UDP-флуд
        if 'udp_protocol' in preprocessed_data.columns:
            udp_flood = (anomalies == 1) & (preprocessed_data['udp_protocol'] > 0)
            result_df.loc[udp_flood, 'dos_attack_type'] = 'UDP Flood'
        
        return result_df
