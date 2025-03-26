import numpy as np
import pandas as pd
from intellectshield.detectors.base import BaseAnomalyDetector
import time

class SequenceAnomalyDetector(BaseAnomalyDetector):
    """
    Детектор аномалий в последовательностях сетевого трафика.
    """
    
    def __init__(self, model_dir="models"):
        """
        Инициализация детектора последовательностей.
        """
        super().__init__(model_dir)
        self.sequence_length = 10  # Длина последовательности для анализа
        self.frequency_dict = {}   # Словарь частот переходов
        self.transition_probs = {}  # Вероятности переходов
        self.min_prob = 0.001      # Минимальная вероятность перехода
        
    def preprocess_data(self, data, train=False):
        """
        Предобработка данных для анализа последовательностей.
        """
        # Создаем копию данных для обработки
        df = data.copy()
        
        # Выбираем ключевые признаки для создания "состояний"
        if 'protocol_type' in df.columns:
            state_features = ['protocol_type']
        elif 'protocol' in df.columns:
            state_features = ['protocol']
        else:
            state_features = []
        
        if 'service' in df.columns:
            state_features.append('service')
        
        if 'dst_port' in df.columns:
            state_features.append('dst_port')
        elif 'dst_port' in df.columns:
            state_features.append('dst_port')
        
        # Если нет ключевых признаков, используем числовые признаки
        if not state_features:
            # Выбираем числовые признаки
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            # Исключаем метки и id
            exclude_cols = ['label', 'is_anomaly', 'predicted_anomaly', 'anomaly_score', 'id']
            state_features = [col for col in numeric_cols if col not in exclude_cols]
            
            # Если числовых признаков много, выбираем первые 3
            if len(state_features) > 3:
                state_features = state_features[:3]
            
            # Для числовых признаков создаем категории
            for feature in state_features:
                if feature in df.columns:
                    # Разбиваем на 5 категорий
                    df[feature + '_cat'] = pd.qcut(df[feature], 5, labels=False, duplicates='drop')
                    state_features[state_features.index(feature)] = feature + '_cat'
        
        # Сохраняем признаки для последующего использования
        self.features = state_features
        
        # Создаем состояния как комбинацию признаков
        if all(feature in df.columns for feature in state_features):
            df['state'] = df[state_features].astype(str).agg('_'.join, axis=1)
        else:
            # Если не все признаки доступны, используем доступные
            available_features = [f for f in state_features if f in df.columns]
            if available_features:
                df['state'] = df[available_features].astype(str).agg('_'.join, axis=1)
            else:
                # Создаем искусственные состояния на основе индекса
                df['state'] = df.index.astype(str)
        
        return df
    
    def train(self, data, min_freq=5, sequence_length=10):
        """
        Обучение модели для обнаружения аномальных последовательностей.
        """
        print("Начало обучения модели анализа последовательностей...")
        start_time = time.time()
        
        # Сохраняем параметры
        self.sequence_length = sequence_length
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=True)
        
        # Извлечение последовательности состояний
        states = preprocessed_data['state'].tolist()
        
        # Подсчет частот переходов
        self.frequency_dict = {}
        
        # Для одиночных состояний
        for state in states:
            if state not in self.frequency_dict:
                self.frequency_dict[state] = 0
            self.frequency_dict[state] += 1
        
        # Для пар состояний (переходов)
        self.transition_freq = {}
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            if current_state not in self.transition_freq:
                self.transition_freq[current_state] = {}
            
            if next_state not in self.transition_freq[current_state]:
                self.transition_freq[current_state][next_state] = 0
            
            self.transition_freq[current_state][next_state] += 1
        
        # Расчет вероятностей переходов
        self.transition_probs = {}
        for current_state, transitions in self.transition_freq.items():
            self.transition_probs[current_state] = {}
            total_transitions = sum(transitions.values())
            
            for next_state, count in transitions.items():
                if count >= min_freq:
                    self.transition_probs[current_state][next_state] = count / total_transitions
                else:
                    # Игнорируем редкие переходы
                    self.transition_probs[current_state][next_state] = 0
        
        # Определение минимальной вероятности перехода
        all_probs = []
        for current_state, transitions in self.transition_probs.items():
            for next_state, prob in transitions.items():
                if prob > 0:
                    all_probs.append(prob)
        
        if all_probs:
            self.min_prob = min(all_probs)
        else:
            self.min_prob = 0.001
        
        # Запись статистики обучения
        training_time = time.time() - start_time
        self.training_summary = {
            'model_type': 'SequenceAnomalyDetector',
            'sequence_length': sequence_length,
            'min_freq': min_freq,
            'unique_states': len(self.frequency_dict),
            'min_probability': self.min_prob,
            'training_samples': len(preprocessed_data),
            'training_time': training_time
        }
        
        print(f"Обучение завершено за {training_time:.2f} секунд")
        print(f"Обнаружено {len(self.frequency_dict)} уникальных состояний")
        
        return self
    
    def predict(self, data):
        """
        Обнаружение аномалий в последовательностях.
        """
        if not self.transition_probs:
            raise ValueError("Модель не обучена. Сначала вызовите метод train().")
        
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=False)
        
        # Извлечение последовательности состояний
        states = preprocessed_data['state'].tolist()
        
        # Анализ последовательностей
        anomaly_scores = []
        
        for i in range(len(states)):
            # Получаем последовательность состояний
            end_idx = min(i + self.sequence_length, len(states))
            sequence = states[i:end_idx]
            
            # Оценка аномальности последовательности
            sequence_score = 0
            
            for j in range(len(sequence) - 1):
                current_state = sequence[j]
                next_state = sequence[j + 1]
                
                # Проверка наличия информации о переходе
                if current_state in self.transition_probs and next_state in self.transition_probs[current_state]:
                    prob = self.transition_probs[current_state][next_state]
                    if prob > 0:
                        # Преобразуем вероятность в оценку аномальности (редкие переходы -> высокая оценка)
                        transition_score = -np.log(prob / self.min_prob)
                        sequence_score += transition_score
                    else:
                        # Если переход не наблюдался в обучающих данных
                        sequence_score += 10  # Высокая оценка аномальности
                else:
                    # Если состояние не наблюдалось в обучающих данных
                    sequence_score += 10  # Высокая оценка аномальности
            
            # Нормализация оценки
            if len(sequence) > 1:
                sequence_score /= (len(sequence) - 1)
            else:
                sequence_score = 0
            
            anomaly_scores.append(sequence_score)
        
        # Нормализация оценок аномалий
        if anomaly_scores:
            max_score = max(anomaly_scores)
            if max_score > 0:
                anomaly_scores = [score / max_score for score in anomaly_scores]
        
        # Определение аномалий (верхние 5%)
        threshold = np.percentile(anomaly_scores, 95)
        anomalies = [1 if score >= threshold else 0 for score in anomaly_scores]
        
        # Создание результирующей таблицы
        result_df = data.copy()
        result_df['predicted_anomaly'] = anomalies
        result_df['anomaly_score'] = anomaly_scores
        
        return result_df
