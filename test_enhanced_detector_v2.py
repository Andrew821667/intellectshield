import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.abspath('..'))

# Импортируем оригинальный и улучшенные детекторы
from detectors.sql_injection_detector import SQLInjectionDetector
from detectors.enhanced_sql_injection_detector import EnhancedSQLInjectionDetector
from detectors.enhanced_sql_injection_detector_v2 import EnhancedSQLInjectionDetectorV2

# Создаем тестовый набор данных
normal_queries = [
    "SELECT * FROM users WHERE id = 123",
    "INSERT INTO products (name, price) VALUES ('Apple', 2.5)",
    "UPDATE customers SET email = 'john@example.com' WHERE id = 456",
    "DELETE FROM orders WHERE status = 'cancelled'",
    "SELECT p.name, c.name FROM products p JOIN categories c ON p.category_id = c.id",
    "SELECT COUNT(*) FROM users WHERE login_attempts > 3",
    "SELECT * FROM transactions WHERE amount > 1000 AND status = 'pending'",
    "SELECT id, username FROM users ORDER BY created_at DESC LIMIT 10",
    "INSERT INTO logs (event, message, timestamp) VALUES ('login', 'User logged in', NOW())",
    "UPDATE products SET stock = stock - 1 WHERE id = 789"
]

suspicious_queries = [
    "SELECT * FROM users WHERE username = '' OR 1=1 --",
    "SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1'",
    "SELECT * FROM users; DROP TABLE users; --",
    "1' UNION SELECT username, password FROM users--",
    "1'; EXEC xp_cmdshell('net user hacker password123 /ADD'); --"
]

# Создаем DataFrame с разметкой
df_test = pd.DataFrame({
    'query': normal_queries + suspicious_queries,
    'is_anomaly': [0] * len(normal_queries) + [1] * len(suspicious_queries)
})

# Создаем обучающий набор только из нормальных запросов
df_train = pd.DataFrame({'query': normal_queries})

# Функция для вычисления метрик
def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }

print("=== Тестирование оригинального детектора SQL-инъекций ===")
# Инициализируем оригинальный детектор
original_detector = SQLInjectionDetector(text_column='query', threshold=0.5, verbose=True)
original_detector.fit(df_train)

# Анализируем тестовые данные
original_results = original_detector.detect_and_explain(df_test)

# Вычисляем метрики для оригинального детектора
y_true = df_test['is_anomaly']
y_pred_original = original_results['predicted_anomaly']
original_metrics = calculate_metrics(y_true, y_pred_original)

print("\nМетрики оригинального детектора:")
print(f"Точность (Accuracy): {original_metrics['accuracy']:.4f}")
print(f"Полнота (Recall): {original_metrics['recall']:.4f}")
print(f"Precision: {original_metrics['precision']:.4f}")
print(f"F1 Score: {original_metrics['f1_score']:.4f}")

print("\n=== Тестирование улучшенного детектора SQL-инъекций V1 ===")
# Инициализируем улучшенный детектор V1
enhanced_detector = EnhancedSQLInjectionDetector(
    text_column='query', 
    threshold=0.6,  # Повышенный порог
    verbose=True
)
enhanced_detector.fit(df_train)

# Анализируем тестовые данные
enhanced_results = enhanced_detector.predict(df_test)

# Добавляем объяснения для улучшенного детектора
explanations = []
for idx, row in enhanced_results.iterrows():
    if row['predicted_anomaly'] == 1:
        text = df_test.loc[idx, 'query']
        explanation = enhanced_detector._generate_explanation(text)
        explanations.append(explanation)
    else:
        explanations.append(None)

enhanced_results['explanation'] = explanations

# Вычисляем метрики для улучшенного детектора V1
y_pred_enhanced = enhanced_results['predicted_anomaly']
enhanced_metrics = calculate_metrics(y_true, y_pred_enhanced)

print("\nМетрики улучшенного детектора V1:")
print(f"Точность (Accuracy): {enhanced_metrics['accuracy']:.4f}")
print(f"Полнота (Recall): {enhanced_metrics['recall']:.4f}")
print(f"Precision: {enhanced_metrics['precision']:.4f}")
print(f"F1 Score: {enhanced_metrics['f1_score']:.4f}")

print("\n=== Тестирование улучшенного детектора SQL-инъекций V2 ===")
# Инициализируем улучшенный детектор V2
enhanced_detector_v2 = EnhancedSQLInjectionDetectorV2(
    text_column='query', 
    threshold=0.55,  # Оптимизированный порог
    verbose=True
)
enhanced_detector_v2.fit(df_train)

# Анализируем тестовые данные
enhanced_results_v2 = enhanced_detector_v2.detect_and_explain(df_test)

# Вычисляем метрики для улучшенного детектора V2
y_pred_enhanced_v2 = enhanced_results_v2['predicted_anomaly']
enhanced_metrics_v2 = calculate_metrics(y_true, y_pred_enhanced_v2)

print("\nМетрики улучшенного детектора V2:")
print(f"Точность (Accuracy): {enhanced_metrics_v2['accuracy']:.4f}")
print(f"Полнота (Recall): {enhanced_metrics_v2['recall']:.4f}")
print(f"Precision: {enhanced_metrics_v2['precision']:.4f}")
print(f"F1 Score: {enhanced_metrics_v2['f1_score']:.4f}")

# Сравниваем результаты детекторов
print("\n=== Сравнение всех трех детекторов ===")
print("1. Оригинальный детектор")
print("2. Улучшенный детектор V1")
print("3. Улучшенный детектор V2")

# Анализируем ложные срабатывания
original_fp = ((original_results['predicted_anomaly'] == 1) & (df_test['is_anomaly'] == 0)).sum()
enhanced_fp = ((enhanced_results['predicted_anomaly'] == 1) & (df_test['is_anomaly'] == 0)).sum()
enhanced_v2_fp = ((enhanced_results_v2['predicted_anomaly'] == 1) & (df_test['is_anomaly'] == 0)).sum()

print("\nЛожные срабатывания:")
print(f"Оригинальный: {original_fp} из {len(normal_queries)} нормальных запросов ({original_fp/len(normal_queries)*100:.1f}%)")
print(f"Улучшенный V1: {enhanced_fp} из {len(normal_queries)} нормальных запросов ({enhanced_fp/len(normal_queries)*100:.1f}%)")
print(f"Улучшенный V2: {enhanced_v2_fp} из {len(normal_queries)} нормальных запросов ({enhanced_v2_fp/len(normal_queries)*100:.1f}%)")

# Анализируем пропущенные аномалии
original_fn = ((original_results['predicted_anomaly'] == 0) & (df_test['is_anomaly'] == 1)).sum()
enhanced_fn = ((enhanced_results['predicted_anomaly'] == 0) & (df_test['is_anomaly'] == 1)).sum()
enhanced_v2_fn = ((enhanced_results_v2['predicted_anomaly'] == 0) & (df_test['is_anomaly'] == 1)).sum()

print("\nПропущенные аномалии:")
print(f"Оригинальный: {original_fn} из {len(suspicious_queries)} вредоносных запросов ({original_fn/len(suspicious_queries)*100:.1f}%)")
print(f"Улучшенный V1: {enhanced_fn} из {len(suspicious_queries)} вредоносных запросов ({enhanced_fn/len(suspicious_queries)*100:.1f}%)")
print(f"Улучшенный V2: {enhanced_v2_fn} из {len(suspicious_queries)} вредоносных запросов ({enhanced_v2_fn/len(suspicious_queries)*100:.1f}%)")

print("\n=== Детальный анализ обнаружения вредоносных запросов ===")
print("Тестовые вредоносные запросы и их обнаружение (1 - обнаружено, 0 - пропущено):")
for i, query in enumerate(suspicious_queries):
    idx = len(normal_queries) + i
    print(f"\nЗапрос: {query}")
    print(f"Оригинальный: {original_results.iloc[idx]['predicted_anomaly']} (оценка: {original_results.iloc[idx]['anomaly_score']:.4f})")
    print(f"Улучшенный V1: {enhanced_results.iloc[idx]['predicted_anomaly']} (оценка: {enhanced_results.iloc[idx]['anomaly_score']:.4f})")
    print(f"Улучшенный V2: {enhanced_results_v2.iloc[idx]['predicted_anomaly']} (оценка: {enhanced_results_v2.iloc[idx]['anomaly_score']:.4f})")
    
    if enhanced_results_v2.iloc[idx]['predicted_anomaly'] == 1:
        print("\nОбъяснение (Улучшенный V2):")
        print(enhanced_results_v2.iloc[idx]['explanation'])
    elif enhanced_results.iloc[idx]['predicted_anomaly'] == 1:
        print("\nОбъяснение (Улучшенный V1):")
        print(enhanced_results.iloc[idx]['explanation'])
