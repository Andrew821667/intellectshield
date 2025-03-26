
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time
import matplotlib.pyplot as plt
import seaborn as sns

from improved_sql_injection_detector import ImprovedSQLInjectionDetector
from improved_ensemble_detector import ImprovedEnsembleDetector

# Создаем простые детекторы для тестирования ансамблевого детектора
class SimpleIsolationForestDetector:
    def train(self, data):
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        features = list(data.columns)
        scaled_data = self.scaler.fit_transform(data[features])
        
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(scaled_data)
        return self
    
    def predict(self, data):
        features = [col for col in data.columns if col != 'is_anomaly']
        scaled_data = self.scaler.transform(data[features])
        
        predictions = self.model.predict(scaled_data)
        scores = self.model.decision_function(scaled_data)
        
        result = data.copy()
        result['predicted_anomaly'] = (predictions == -1).astype(int)
        result['anomaly_score'] = -scores  # Инвертируем оценки
        
        return result

class SimpleLOFDetector:
    def train(self, data):
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        features = list(data.columns)
        scaled_data = self.scaler.fit_transform(data[features])
        
        self.model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        self.model.fit(scaled_data)
        return self
    
    def predict(self, data):
        features = [col for col in data.columns if col != 'is_anomaly']
        scaled_data = self.scaler.transform(data[features])
        
        predictions = self.model.predict(scaled_data)
        scores = -self.model.decision_function(scaled_data)
        
        result = data.copy()
        result['predicted_anomaly'] = (predictions == -1).astype(int)
        result['anomaly_score'] = scores
        
        return result

def test_improved_detectors():
    print("=== ТЕСТИРОВАНИЕ УЛУЧШЕННЫХ ДЕТЕКТОРОВ ===")
    
    # Тестирование улучшенного ансамблевого детектора аномалий
    print("\n1. Тестирование улучшенного ансамблевого детектора")
    
    # Генерация синтетических данных для аномалий
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    # Создаем нормальные данные
    normal_data = np.random.normal(0, 1, size=(int(n_samples * 0.95), n_features))
    
    # Создаем аномалии
    anomaly_data = np.random.normal(5, 1, size=(int(n_samples * 0.05), n_features))
    
    # Объединяем данные
    X = np.vstack([normal_data, anomaly_data])
    y = np.zeros(n_samples)
    y[int(n_samples * 0.95):] = 1
    
    # Перемешиваем данные
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Создаем DataFrame
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['is_anomaly'] = y
    
    print(f"Создано {len(df)} образцов с {int(df['is_anomaly'].sum())} аномалиями")
    
    # Разделение на обучающую и тестовую выборки
    train_data = df[df['is_anomaly'] == 0].drop('is_anomaly', axis=1)
    test_data = df
    
    # Создаем и тренируем базовые детекторы
    isolation_forest = SimpleIsolationForestDetector()
    isolation_forest.train(train_data)
    
    lof = SimpleLOFDetector()
    lof.train(train_data)
    
    # Создаем улучшенный ансамблевый детектор
    ensemble_detector = ImprovedEnsembleDetector(ensemble_method='weighted_voting')
    ensemble_detector.add_detector(isolation_forest, "IsolationForest", weight=1.5)
    ensemble_detector.add_detector(lof, "LOF", weight=1.0)
    
    # Обучаем ансамблевый детектор
    ensemble_detector.train(train_data, test_data['is_anomaly'])
    
    # Тестируем ансамблевый детектор
    ensemble_results = ensemble_detector.predict(test_data)
    
    # Оцениваем результаты
    metrics = {}
    y_true = test_data['is_anomaly']
    y_pred = ensemble_results['predicted_anomaly']
    scores = ensemble_results['anomaly_score']
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['auc'] = roc_auc_score(y_true, scores)
    
    print("
Результаты улучшенного ансамблевого детектора:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    
    # Тестирование улучшенного детектора SQL-инъекций
    print("
2. Тестирование улучшенного детектора SQL-инъекций")
    
    # Создаем тестовые данные SQL-запросов
    normal_queries = [
        "SELECT * FROM users WHERE id = 123",
        "INSERT INTO products (name, price) VALUES ('Apple', 2.5)",
        "UPDATE customers SET email = 'john@example.com' WHERE id = 456",
        "DELETE FROM orders WHERE status = 'cancelled'",
        "SELECT p.name, c.name FROM products p JOIN categories c ON p.category_id = c.id"
    ]

    injection_queries = [
        "SELECT * FROM users WHERE username = '' OR 1=1 --",
        "1' UNION SELECT username, password FROM users--",
        "1'; EXEC xp_cmdshell('net user hacker password123 /ADD'); --",
        "SELECT * FROM users WHERE username = '' OR/**/1=1--",  # Обфускация
        "' + 'OR' + '1' + '=' + '1"  # Конкатенация строк
    ]

    # Создание IP-адресов
    normal_ips = ["192.168.1.1", "192.168.1.2"]
    attack_ips = ["10.0.0.1", "10.0.0.2"]

    # Создание тестовых данных
    queries = normal_queries + injection_queries
    ips = normal_ips * (len(normal_queries) // 2 + 1) + attack_ips * (len(injection_queries) // 2 + 1)
    ips = ips[:len(queries)]
    labels = [0] * len(normal_queries) + [1] * len(injection_queries)

    sql_df = pd.DataFrame({
        'query': queries,
        'ip_address': ips,
        'is_anomaly': labels
    })

    # Создание тренировочных данных (только нормальные запросы)
    sql_train = pd.DataFrame({'query': normal_queries})

    # Создание улучшенного детектора SQL-инъекций
    sql_detector = ImprovedSQLInjectionDetector(
        text_column='query',
        threshold=0.3,
        verbose=True,
        use_ml=True,
        adaptive_threshold=True,
        context_aware=True
    )

    # Обучение детектора
    sql_detector.fit(sql_train)

    # Тестирование детектора
    sql_results = sql_detector.detect_and_explain(sql_df, sql_df['ip_address'])

    # Оценка результатов
    sql_y_true = sql_df['is_anomaly']
    sql_y_pred = sql_results['predicted_anomaly']

    sql_accuracy = accuracy_score(sql_y_true, sql_y_pred)
    sql_precision = precision_score(sql_y_true, sql_y_pred, zero_division=0)
    sql_recall = recall_score(sql_y_true, sql_y_pred, zero_division=0)
    sql_f1 = f1_score(sql_y_true, sql_y_pred, zero_division=0)

    print("
Результаты улучшенного детектора SQL-инъекций:")
    print(f"Accuracy: {sql_accuracy:.4f}")
    print(f"Precision: {sql_precision:.4f}")
    print(f"Recall: {sql_recall:.4f}")
    print(f"F1 Score: {sql_f1:.4f}")

    # Примеры обнаруженных инъекций
    print("
Примеры обнаруженных SQL-инъекций:")
    detected = sql_df[sql_results['predicted_anomaly'] == 1]
    for _, row in detected.iterrows():
        print(f"
Запрос: {row['query']}")
        idx = sql_results[sql_results['query'] == row['query']].index[0]
        print(f"Оценка аномальности: {sql_results.loc[idx, 'anomaly_score']:.4f}")
        print(f"Тип атаки: {sql_results.loc[idx, 'attack_type']}")
        print(f"Уровень риска: {sql_results.loc[idx, 'risk_level']}")

    print("
Тестирование улучшенных детекторов завершено!")
    
    return {
        'ensemble_metrics': metrics,
        'sql_metrics': {
            'accuracy': sql_accuracy,
            'precision': sql_precision,
            'recall': sql_recall,
            'f1': sql_f1
        }
    }

if __name__ == "__main__":
    test_improved_detectors()
