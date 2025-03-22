
import sys
import os
import pandas as pd

# Добавляем корневую директорию проекта в путь для импорта
sys.path.append(os.path.abspath('..'))

from detectors.sql_injection_detector import SQLInjectionDetector

# Создаем тестовый набор данных
normal_queries = [
    "SELECT * FROM users WHERE id = 123",
    "INSERT INTO products (name, price) VALUES ('Apple', 2.5)",
    "UPDATE customers SET email = 'john@example.com' WHERE id = 456",
    "DELETE FROM orders WHERE status = 'cancelled'",
    "SELECT p.name, c.name FROM products p JOIN categories c ON p.category_id = c.id"
]

suspicious_queries = [
    "SELECT * FROM users WHERE username = '' OR 1=1 --",
    "SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1'",
    "SELECT * FROM users; DROP TABLE users; --",
    "1' UNION SELECT username, password FROM users--",
    "1'; EXEC xp_cmdshell('net user hacker password123 /ADD'); --"
]

# Создаем DataFrame
df_train = pd.DataFrame({'query': normal_queries})
df_test = pd.DataFrame({'query': normal_queries + suspicious_queries})

# Инициализируем и обучаем детектор
detector = SQLInjectionDetector(text_column='query', threshold=0.5, verbose=True)
detector.fit(df_train)

# Анализируем тестовые данные
results = detector.detect_and_explain(df_test)

# Выводим результаты
print("\nРезультаты анализа SQL-инъекций:\n")
print(f"Всего запросов: {len(df_test)}")
print(f"Обнаружено аномалий: {results['predicted_anomaly'].sum()} ({results['predicted_anomaly'].sum()/len(df_test)*100:.1f}%)\n")

for i, (query, is_anomaly, anomaly_type, explanation) in enumerate(
    zip(df_test['query'], results['predicted_anomaly'], results['anomaly_type'], results['explanation'])
):
    print(f"{i+1}. Запрос: {query}")
    print(f"   Аномалия: {'Да' if is_anomaly else 'Нет'}")
    if is_anomaly:
        print(f"   Тип аномалии: {anomaly_type}")
        print(f"   Объяснение:\n{explanation}\n")
    print("-" * 80)
