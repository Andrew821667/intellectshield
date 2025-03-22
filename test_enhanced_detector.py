import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.abspath('..'))

# Импортируем оригинальный и улучшенный детекторы
from detectors.sql_injection_detector import SQLInjectionDetector
from detectors.enhanced_sql_injection_detector import EnhancedSQLInjectionDetector

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

print("=== Тестирование оригинального детектора SQL-инъекций ===")
# Инициализируем оригинальный детектор
original_detector = SQLInjectionDetector(text_column='query', threshold=0.5, verbose=True)
original_detector.fit(df_train)

# Анализируем тестовые данные
original_results = original_detector.detect_and_explain(df_test)

# Вычисляем метрики для оригинального детектора
y_true = df_test['is_anomaly']
y_pred = original_results['predicted_anomaly']

original_metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred, zero_division=0),
    'recall': recall_score(y_true, y_pred, zero_division=0),
    'f1_score': f1_score(y_true, y_pred, zero_division=0)
}

print("\nМетрики оригинального детектора:")
print(f"Точность (Accuracy): {original_metrics['accuracy']:.4f}")
print(f"Полнота (Recall): {original_metrics['recall']:.4f}")
print(f"Precision: {original_metrics['precision']:.4f}")
print(f"F1 Score: {original_metrics['f1_score']:.4f}")

print("\n=== Тестирование улучшенного детектора SQL-инъекций ===")
# Инициализируем улучшенный детектор
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

# Вычисляем метрики для улучшенного детектора
y_pred_enhanced = enhanced_results['predicted_anomaly']

enhanced_metrics = {
    'accuracy': accuracy_score(y_true, y_pred_enhanced),
    'precision': precision_score(y_true, y_pred_enhanced, zero_division=0),
    'recall': recall_score(y_true, y_pred_enhanced, zero_division=0),
    'f1_score': f1_score(y_true, y_pred_enhanced, zero_division=0)
}

print("\nМетрики улучшенного детектора:")
print(f"Точность (Accuracy): {enhanced_metrics['accuracy']:.4f}")
print(f"Полнота (Recall): {enhanced_metrics['recall']:.4f}")
print(f"Precision: {enhanced_metrics['precision']:.4f}")
print(f"F1 Score: {enhanced_metrics['f1_score']:.4f}")

# Сравниваем результаты детекторов
print("\n=== Сравнение оригинального и улучшенного детекторов ===")

# Анализируем различия в обнаружении
diff_detection = (original_results['predicted_anomaly'] != enhanced_results['predicted_anomaly']).sum()
print(f"Количество запросов с разными предсказаниями: {diff_detection}")

# Анализируем ложные срабатывания
original_fp = ((original_results['predicted_anomaly'] == 1) & (df_test['is_anomaly'] == 0)).sum()
enhanced_fp = ((enhanced_results['predicted_anomaly'] == 1) & (df_test['is_anomaly'] == 0)).sum()

print(f"Ложные срабатывания (оригинальный): {original_fp}")
print(f"Ложные срабатывания (улучшенный): {enhanced_fp}")
if original_fp > 0:
    print(f"Снижение ложных срабатываний: {original_fp - enhanced_fp} ({(1 - enhanced_fp/original_fp)*100:.1f}%)")
else:
    print("Ложных срабатываний не обнаружено")

# Анализируем пропущенные аномалии
original_fn = ((original_results['predicted_anomaly'] == 0) & (df_test['is_anomaly'] == 1)).sum()
enhanced_fn = ((enhanced_results['predicted_anomaly'] == 0) & (df_test['is_anomaly'] == 1)).sum()

print(f"Пропущенные аномалии (оригинальный): {original_fn}")
print(f"Пропущенные аномалии (улучшенный): {enhanced_fn}")

# Выводим детали для нескольких примеров
print("\n=== Примеры результатов анализа ===")
for i in range(min(3, len(suspicious_queries))):
    idx = len(normal_queries) + i
    query = df_test.iloc[idx]['query']
    print(f"\nЗапрос: {query}")
    print(f"Оценка (оригинальный): {original_results.iloc[idx]['anomaly_score']:.4f}")
    print(f"Оценка (улучшенный): {enhanced_results.iloc[idx]['anomaly_score']:.4f}")
    if enhanced_results.iloc[idx]['predicted_anomaly'] == 1:
        print("Объяснение (улучшенный):")
        print(enhanced_results.iloc[idx]['explanation'])
