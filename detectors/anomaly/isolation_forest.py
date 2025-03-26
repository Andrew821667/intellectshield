import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from intellectshield.detectors.base import BaseAnomalyDetector
import time

class IsolationForestDetector(BaseAnomalyDetector):
    """
    Детектор аномалий на основе алгоритма Isolation Forest.
    """
    
    def preprocess_data(self, data, train=False):
        """
        Предобработка данных для Isolation Forest.
        """
        # Создадим копию данных для обработки
        df = data.copy()
        
        # 1. Обработка временных признаков
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df.drop('timestamp', axis=1, inplace=True)
        
        # 2. Обработка IP-адресов
        if 'src_ip' in df.columns and df['src_ip'].dtype == 'object':
            # IP-адреса уже должны иметь числовые хеши
            if 'src_ip_hash' not in df.columns:
                df['src_ip_hash'] = df['src_ip'].apply(lambda x: hash(str(x)) % 10000)
            df.drop('src_ip', axis=1, inplace=True)
        
        if 'dst_ip' in df.columns and df['dst_ip'].dtype == 'object':
            if 'dst_ip_hash' not in df.columns:
                df['dst_ip_hash'] = df['dst_ip'].apply(lambda x: hash(str(x)) % 10000)
            df.drop('dst_ip', axis=1, inplace=True)
        
        # 3. Обработка категориальных признаков
        categorical_features = ['protocol', 'src_port', 'dst_port']
        for feature in categorical_features:
            if feature in df.columns:
                # Для портов преобразуем известные порты и объединяем редкие
                if feature in ['src_port', 'dst_port']:
                    common_ports = [20, 21, 22, 23, 25, 53, 80, 123, 443, 3389]
                    df[feature] = df[feature].apply(
                        lambda x: x if x in common_ports else 0)
                # Для протоколов используем one-hot encoding
                if feature == 'protocol':
                    # Если протокол строковый, преобразуем в числовой
                    if df[feature].dtype == 'object':
                        proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1}
                        df[feature] = df[feature].map(proto_map).fillna(0).astype(int)
                    df = pd.get_dummies(df, columns=[feature], prefix=feature)
        
        # 4. Определяем признаки для анализа (исключая метки)
        self.features = [col for col in df.columns if col not in ['label', 'is_anomaly']]
        
        # 5. Заменяем бесконечности и NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 6. Масштабирование численных признаков
        if train or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(df[self.features])
        else:
            scaled_features = self.scaler.transform(df[self.features])
        
        # 7. Создаем датафрейм из масштабированных данных
        df_processed = pd.DataFrame(scaled_features, columns=self.features)
        
        # 8. Если в исходных данных есть метки, добавим их обратно
        if 'is_anomaly' in data.columns:
            df_processed['is_anomaly'] = data['is_anomaly'].values
        
        return df_processed
    
    def train(self, data, contamination=0.05, n_estimators=100, max_samples='auto', random_state=42):
        """
        Обучение модели Isolation Forest.
        """
        print("Начало обучения модели Isolation Forest...")
        start_time = time.time()
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=True)
        
        # Определение признаков для обучения
        features = [col for col in preprocessed_data.columns if col != 'is_anomaly']
        
        # Инициализация и обучение модели Isolation Forest
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Обучение модели
        self.model.fit(preprocessed_data[features])
        
        # Запись статистики обучения
        training_time = time.time() - start_time
        self.training_summary = {
            'model_type': 'IsolationForest',
            'n_estimators': n_estimators,
            'contamination': contamination,
            'max_samples': max_samples,
            'training_samples': len(preprocessed_data),
            'n_features': len(features),
            'training_time': training_time
        }
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        
        return self
    
    def predict(self, data):
        """
        Обнаружение аномалий с помощью Isolation Forest.
        """
        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=False)
        
        # Определение признаков для предсказания
        features = [col for col in preprocessed_data.columns if col != 'is_anomaly']
        
        # Получение решения модели (-1 для аномалий, 1 для нормального трафика)
        predictions = self.model.predict(preprocessed_data[features])
        
        # Получение аномальных оценок
        anomaly_scores = self.model.decision_function(preprocessed_data[features])

        # Создание результирующей таблицы
        result_df = data.copy()
        result_df['predicted_anomaly'] = (predictions == -1).astype(int)

        # Инвертируем оценки, чтобы более высокие значения соответствовали более аномальным экземплярам
        # Также обеспечиваем, чтобы все оценки были неотрицательными путем смещения всех значений
        inverted_scores = -anomaly_scores
        min_score = np.min(inverted_scores)
        if min_score < 0:
            inverted_scores = inverted_scores - min_score  # Смещение всех значений, чтобы минимум был 0

        result_df['anomaly_score'] = inverted_scores
        
        return result_df
