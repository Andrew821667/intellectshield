import sys
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import os
sys.path.append('.')

try:
    # Импортируем наш новый улучшенный детектор
    from intellectshield.detectors.enhanced_sql_injection_detector_v4 import EnhancedSQLInjectionDetectorV4
    # Импортируем предыдущую версию для сравнения
    from intellectshield.detectors.enhanced_sql_injection_detector_enhanced import EnhancedSQLInjectionDetector
    
    print("=== СРАВНИТЕЛЬНОЕ ТЕСТИРОВАНИЕ УЛУЧШЕННОГО SQL-ДЕТЕКТОРА V4 ===\n")
    
    # ПОДГОТОВКА ТЕСТОВЫХ ДАННЫХ
    print("Подготовка тестовых данных...")
    
    # 1. Нормальные SQL-запросы
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
    
    # 2. Базовые SQL-инъекции
    basic_injections = [
        "SELECT * FROM users WHERE username = '' OR 1=1 --",
        "SELECT * FROM users WHERE username = 'admin' AND password = '' OR '1'='1'",
        "SELECT * FROM users; DROP TABLE users; --",
        "1' UNION SELECT username, password FROM users--",
        "1'; EXEC xp_cmdshell('net user hacker password123 /ADD'); --"
    ]
    
    # 3. Обфусцированные SQL-инъекции (различные техники обфускации)
    obfuscated_injections = [
        "SELECT * FROM users WHERE username = '' OR/**/1=1--",  # Комментарии
        "SELECT * FROM users WHERE username = '' %4fR 1=1--",   # URL-кодирование
        "SELECT * FROM users WHERE username = '' UnIoN SeLeCt username, password FROM users--",  # Смешанный регистр
        "SELECT * FROM users WHERE username = '' OR 0x313d31--",  # Шестнадцатеричное кодирование
        "SELECT * FROM users WHERE username = CHAR(39) + CHAR(39) + CHAR(32) + CHAR(79) + CHAR(82) + CHAR(32) + CHAR(49) + CHAR(61) + CHAR(49)--",  # Кодирование символов
        "SELECT * FROM users WHERE username = '' OR'1'='1'--",  # Без пробелов
        "SELECT * FROM users WHERE username = '' OR/**//**/1=1--",  # Множественные комментарии
        "' + 'OR' + '1' + '=' + '1",  # Конкатенация строк
        "SELECT * FROM users WHERE username = '' OR true--",  # Логические значения
        "SELECT * FROM users WHERE username = '' OR 0.1=0.1--"  # Числа с плавающей точкой
    ]
    
    # 4. Продвинутые SQL-инъекции
    advanced_injections = [
        "SELECT * FROM users WHERE username = '' AND 5=(SELECT COUNT(*) FROM information_schema.tables)--",  # Подзапросы
        "SELECT * FROM users WHERE username = '' UNION SELECT 1, table_name FROM information_schema.tables--",  # Получение схемы БД
        "SELECT * FROM users WHERE username = '' WAITFOR DELAY '0:0:5'--",  # Time-based атака
        "SELECT * FROM users WHERE username = '' AND (SELECT SUBSTRING(username,1,1) FROM users WHERE id=1)='a'--",  # Blind SQL Injection
        "SELECT * FROM users WHERE username = '' UNION SELECT 1, @@version--"  # Получение версии БД
    ]
    
    # Создаем IP-адреса для контекстно-зависимого анализа
    ips = [f"192.168.1.{i}" for i in range(1, 11)]
    
    # Подготовка данных для тестирования
    all_queries = []
    all_ips = []
    all_labels = []
    
    # Нормальные запросы - распределены по разным IP
    for i, query in enumerate(normal_queries):
        all_queries.append(query)
        all_ips.append(ips[i % len(ips)])
        all_labels.append(0)
    
    # Базовые инъекции - все с одного IP (имитация атаки)
    for query in basic_injections:
        all_queries.append(query)
        all_ips.append(ips[0])
        all_labels.append(1)
    
    # Обфусцированные инъекции - все с другого IP
    for query in obfuscated_injections:
        all_queries.append(query)
        all_ips.append(ips[1])
        all_labels.append(1)
    
    # Продвинутые инъекции - с третьего IP
    for query in advanced_injections:
        all_queries.append(query)
        all_ips.append(ips[2])
        all_labels.append(1)
    
    # Создаем DataFrame для тестирования
    df_test = pd.DataFrame({
        'query': all_queries,
        'ip_address': all_ips,
        'is_anomaly': all_labels
    })
    
    # Создаем обучающий набор только из нормальных запросов
    df_train = pd.DataFrame({'query': normal_queries})
    
    print(f"Подготовлено тестовых запросов: {len(df_test)}")
    print(f" - Нормальных запросов: {len(normal_queries)}")
    print(f" - Базовых SQL-инъекций: {len(basic_injections)}")
    print(f" - Обфусцированных SQL-инъекций: {len(obfuscated_injections)}")
    print(f" - Продвинутых SQL-инъекций: {len(advanced_injections)}")
    
    # ТЕСТИРОВАНИЕ ПРЕДЫДУЩЕЙ ВЕРСИИ ДЕТЕКТОРА
    print("\n=== ТЕСТ 1: Предыдущая версия детектора ===")
    print("Инициализация предыдущей версии детектора...")
    
    prev_detector = EnhancedSQLInjectionDetector(
        text_column='query',
        threshold=0.5,
        verbose=True,
        use_ml=True,
        adaptive_threshold=True,
        context_aware=True
    )
    
    # Обучение и тестирование
    prev_detector.fit(df_train)
    
    start_time = time.time()
    prev_results = prev_detector.detect_and_explain(df_test, df_test['ip_address'])
    prev_time = time.time() - start_time
    
    # Вычисление метрик
    y_true = df_test['is_anomaly']
    y_pred_prev = prev_results['predicted_anomaly']
    
    prev_accuracy = accuracy_score(y_true, y_pred_prev)
    prev_precision = precision_score(y_true, y_pred_prev, zero_division=0)
    prev_recall = recall_score(y_true, y_pred_prev, zero_division=0)
    prev_f1 = f1_score(y_true, y_pred_prev, zero_division=0)
    
    print(f"Результаты тестирования предыдущей версии:")
    print(f"Время выполнения: {prev_time:.4f} секунд")
    print(f"Точность (Accuracy): {prev_accuracy:.4f}")
    print(f"Полнота (Recall): {prev_recall:.4f}")
    print(f"Precision: {prev_precision:.4f}")
    print(f"F1 Score: {prev_f1:.4f}")
    
    # ТЕСТИРОВАНИЕ НОВОЙ ВЕРСИИ ДЕТЕКТОРА
    print("\n=== ТЕСТ 2: Новая версия детектора (V4) ===")
    print("Инициализация новой версии детектора...")
    
    new_detector = EnhancedSQLInjectionDetectorV4(
        text_column='query',
        threshold=0.5,
        verbose=True,
        use_ml=True,
        adaptive_threshold=True,
        context_aware=True
    )
    
    # Обучение и тестирование
    new_detector.fit(df_train)
    
    start_time = time.time()
    new_results = new_detector.detect_and_explain(df_test, df_test['ip_address'])
    new_time = time.time() - start_time
    
    # Вычисление метрик
    y_pred_new = new_results['predicted_anomaly']
    
    new_accuracy = accuracy_score(y_true, y_pred_new)
    new_precision = precision_score(y_true, y_pred_new, zero_division=0)
    new_recall = recall_score(y_true, y_pred_new, zero_division=0)
    new_f1 = f1_score(y_true, y_pred_new, zero_division=0)
    
    print(f"Результаты тестирования новой версии:")
    print(f"Время выполнения: {new_time:.4f} секунд")
    print(f"Точность (Accuracy): {new_accuracy:.4f}")
    print(f"Полнота (Recall): {new_recall:.4f}")
    print(f"Precision: {new_precision:.4f}")
    print(f"F1 Score: {new_f1:.4f}")
    
    # Сравнение результатов
    print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
    print(f"Изменение точности: {(new_accuracy - prev_accuracy) * 100:.2f}%")
    print(f"Изменение полноты: {(new_recall - prev_recall) * 100:.2f}%")
    print(f"Изменение precision: {(new_precision - prev_precision) * 100:.2f}%")
    print(f"Изменение F1: {(new_f1 - prev_f1) * 100:.2f}%")
    print(f"Изменение времени выполнения: {(new_time - prev_time) / prev_time * 100:.2f}%")
    
    # Функция для расчета метрик по диапазону
    def calculate_category_metrics(results, start_idx, end_idx, category_name):
        category_indices = range(start_idx, end_idx)
        detected = sum(results.iloc[idx]['predicted_anomaly'] for idx in category_indices)
        total = end_idx - start_idx
        
        print(f"\n{category_name}:")
        print(f" - Обнаружено: {detected} из {total} ({detected/total*100:.1f}%)")
        
        # Примеры обнаруженных и пропущенных
        if detected < total:
            missed_indices = [idx for idx in category_indices if results.iloc[idx]['predicted_anomaly'] == 0]
            print(f" - Пример пропущенной инъекции: {df_test.iloc[missed_indices[0]]['query']}")
        
        if detected > 0:
            detected_indices = [idx for idx in category_indices if results.iloc[idx]['predicted_anomaly'] == 1]
            print(f" - Пример обнаруженной инъекции: {df_test.iloc[detected_indices[0]]['query']}")
            print(f"   Оценка аномалии: {results.iloc[detected_indices[0]]['anomaly_score']:.2f}")
    
    # Анализ результатов по категориям
    normal_start, normal_end = 0, len(normal_queries)
    basic_start, basic_end = normal_end, normal_end + len(basic_injections)
    obfuscated_start, obfuscated_end = basic_end, basic_end + len(obfuscated_injections)
    advanced_start, advanced_end = obfuscated_end, obfuscated_end + len(advanced_injections)
    
    print("\n=== АНАЛИЗ РЕЗУЛЬТАТОВ ПО КАТЕГОРИЯМ ДЛЯ НОВОГО ДЕТЕКТОРА ===")
    calculate_category_metrics(new_results, normal_start, normal_end, "Нормальные запросы (должны быть 0 обнаружений)")
    calculate_category_metrics(new_results, basic_start, basic_end, "Базовые SQL-инъекции")
    calculate_category_metrics(new_results, obfuscated_start, obfuscated_end, "Обфусцированные SQL-инъекции")
    calculate_category_metrics(new_results, advanced_start, advanced_end, "Продвинутые SQL-инъекции")
    
    # Особое внимание к конкатенированным строкам
    concat_idx = df_test[df_test['query'] == "' + 'OR' + '1' + '=' + '1"].index
    if len(concat_idx) > 0:
        concat_idx = concat_idx[0]
        prev_detected = prev_results.iloc[concat_idx]['predicted_anomaly'] == 1
        new_detected = new_results.iloc[concat_idx]['predicted_anomaly'] == 1
        
        print("\n=== ТЕСТ ОБНАРУЖЕНИЯ КОНКАТЕНИРОВАННЫХ СТРОК ===")
        print(f"Запрос: ' + 'OR' + '1' + '=' + '1")
        print(f"Предыдущая версия обнаружила: {prev_detected}")
        print(f"Новая версия обнаружила: {new_detected}")
        if new_detected:
            print(f"Оценка аномалии: {new_results.iloc[concat_idx]['anomaly_score']:.2f}")
            if 'obfuscation_details' in new_results and new_results.iloc[concat_idx]['obfuscation_details']:
                print("Информация об обфускации:")
                print(new_results.iloc[concat_idx]['obfuscation_details'])
    
    # Сохраняем результаты в файл
    result_file = os.path.join(os.getcwd(), "detection_comparison_results.csv")
    comparison_df = pd.DataFrame({
        'query': df_test['query'],
        'is_anomaly': df_test['is_anomaly'],
        'prev_predicted': prev_results['predicted_anomaly'],
        'prev_score': prev_results['anomaly_score'],
        'new_predicted': new_results['predicted_anomaly'],
        'new_score': new_results['anomaly_score']
    })
    comparison_df.to_csv(result_file, index=False)
    print(f"\nРезультаты сравнения сохранены в файл: {result_file}")
    
    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")
    print("Улучшенный детектор SQL-инъекций V4 успешно протестирован и сравнен с предыдущей версией.")
    
except Exception as e:
    print(f"Ошибка при тестировании детектора: {e}")
    import traceback
    traceback.print_exc()
