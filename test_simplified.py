import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from simplified_detector import AdvancedSQLDetector

def test_detector():
    """Тестирование улучшенного детектора SQL-инъекций."""
    
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
    
    # Тестирование детектора
    print("=== Тестирование улучшенного детектора SQL-инъекций V3 ===")
    detector = AdvancedSQLDetector(text_column='query', threshold=0.5, verbose=True)
    detector.fit(df_train)
    
    print("Выполняем тестирование...")
    
    results = detector.detect_and_explain(df_test)
    y_true = df_test['is_anomaly']
    y_pred = results['predicted_anomaly']
    
    # Вычисляем метрики
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print("\nМетрики детектора V3:")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Анализ ложных срабатываний и пропущенных аномалий
    false_positives = ((results['predicted_anomaly'] == 1) & (y_true == 0)).sum()
    false_negatives = ((results['predicted_anomaly'] == 0) & (y_true == 1)).sum()
    
    print("\nЛожные срабатывания:")
    print(f"V3: {false_positives} из {len(normal_queries)} ({false_positives/len(normal_queries)*100:.1f}%)")
    
    print("\nПропущенные аномалии:")
    print(f"V3: {false_negatives} из {len(suspicious_queries)} ({false_negatives/len(suspicious_queries)*100:.1f}%)")
    
    # Подробный анализ аномалий
    print("\n=== Детальный анализ обнаруженных аномалий ===")
    for i, query in enumerate(suspicious_queries):
        idx = len(normal_queries) + i
        print(f"\nЗапрос: {query}")
        print(f"Оценка аномалии: {results.iloc[idx]['anomaly_score']:.4f}")
        
        if results.iloc[idx]['predicted_anomaly'] == 1:
            print(f"Уровень угрозы: {results.iloc[idx]['threat_level']}")
            print("Статус: Обнаружен как аномальный")
            
            if results.iloc[idx]['explanation'] is not None:
                print("\nОбъяснение:")
                print(results.iloc[idx]['explanation'])
            
            if results.iloc[idx]['recommendation'] is not None:
                print("\nРекомендации:")
                print(results.iloc[idx]['recommendation'])
        else:
            print("Статус: НЕ обнаружен как аномальный (ложноотрицательный)")
    
    return detector, results

if __name__ == "__main__":
    test_detector()
