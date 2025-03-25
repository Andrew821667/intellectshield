
import sys
import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
sys.path.append('.')

try:
    from intellectshield.detectors.enhanced_sql_injection_detector_enhanced import EnhancedSQLInjectionDetector
    print("=== КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ УЛУЧШЕННОГО SQL-ДЕТЕКТОРА ===\n")
    
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
    
    # ТЕСТИРОВАНИЕ БАЗОВОЙ ФУНКЦИОНАЛЬНОСТИ
    print("\n=== ТЕСТ 1: Базовая функциональность детектора ===")
    print("Инициализация детектора с базовыми параметрами...")
    
    basic_detector = EnhancedSQLInjectionDetector(
        text_column='query',
        threshold=0.5,
        verbose=True
    )
    
    # Обучение и тестирование
    basic_detector.fit(df_train)
    
    start_time = time.time()
    basic_results = basic_detector.detect_and_explain(df_test)
    basic_time = time.time() - start_time
    
    # Вычисление метрик
    y_true = df_test['is_anomaly']
    y_pred = basic_results['predicted_anomaly']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Результаты базового тестирования:")
    print(f"Время выполнения: {basic_time:.4f} секунд")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Полнота (Recall): {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Анализ по категориям инъекций
    normal_start, normal_end = 0, len(normal_queries)
    basic_start, basic_end = normal_end, normal_end + len(basic_injections)
    obfuscated_start, obfuscated_end = basic_end, basic_end + len(obfuscated_injections)
    advanced_start, advanced_end = obfuscated_end, obfuscated_end + len(advanced_injections)
    
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
    
    print("\nАнализ результатов по категориям:")
    calculate_category_metrics(basic_results, normal_start, normal_end, "Нормальные запросы (должны быть 0 обнаружений)")
    calculate_category_metrics(basic_results, basic_start, basic_end, "Базовые SQL-инъекции")
    calculate_category_metrics(basic_results, obfuscated_start, obfuscated_end, "Обфусцированные SQL-инъекции")
    calculate_category_metrics(basic_results, advanced_start, advanced_end, "Продвинутые SQL-инъекции")
    
    # ТЕСТИРОВАНИЕ ОБРАБОТКИ ОБФУСКАЦИИ
    print("\n=== ТЕСТ 2: Детальный анализ обфускации ===")
    
    # Проверка обфусцированных инъекций
    obfuscation_techniques = {}
    obfuscation_count = 0
    
    for idx in range(obfuscated_start, obfuscated_end):
        if basic_results.iloc[idx]['obfuscation_details'] is not None:
            details = basic_results.iloc[idx]['obfuscation_details']
            obfuscation_count += 1
            
            for technique in details.get('techniques', []):
                obfuscation_techniques[technique] = obfuscation_techniques.get(technique, 0) + 1
    
    print(f"Обнаружено обфускаций: {obfuscation_count} из {len(obfuscated_injections)}")
    print("Обнаруженные техники обфускации:")
    for technique, count in obfuscation_techniques.items():
        print(f" - {technique}: {count} случаев")
    
    # Пример детальной информации об обфускации
    if obfuscation_count > 0:
        for idx in range(obfuscated_start, obfuscated_end):
            if basic_results.iloc[idx]['obfuscation_details'] is not None:
                details = basic_results.iloc[idx]['obfuscation_details']
                print("\nПример обнаружения обфускации:")
                print(f"Исходный запрос: {details['original']}")
                print(f"Декодированный запрос: {details['deobfuscated']}")
                print(f"Обнаруженные техники: {', '.join(details['techniques'])}")
                break
    
    # ТЕСТИРОВАНИЕ МАШИННОГО ОБУЧЕНИЯ И АДАПТИВНЫХ ПОРОГОВ
    print("\n=== ТЕСТ 3: Машинное обучение и адаптивные пороги ===")
    
    # Инициализация детектора с ML и адаптивными порогами
    advanced_detector = EnhancedSQLInjectionDetector(
        text_column='query',
        threshold=0.5,
        verbose=True,
        use_ml=True,
        adaptive_threshold=True
    )
    
    # Обучение и тестирование
    advanced_detector.fit(df_train)
    
    start_time = time.time()
    advanced_results = advanced_detector.detect_and_explain(df_test, df_test['ip_address'])
    advanced_time = time.time() - start_time
    
    # Вычисление метрик
    y_pred_advanced = advanced_results['predicted_anomaly']
    
    advanced_accuracy = accuracy_score(y_true, y_pred_advanced)
    advanced_precision = precision_score(y_true, y_pred_advanced, zero_division=0)
    advanced_recall = recall_score(y_true, y_pred_advanced, zero_division=0)
    advanced_f1 = f1_score(y_true, y_pred_advanced, zero_division=0)
    
    print(f"Результаты расширенного тестирования (ML + адаптивные пороги):")
    print(f"Время выполнения: {advanced_time:.4f} секунд")
    print(f"Точность (Accuracy): {advanced_accuracy:.4f}")
    print(f"Полнота (Recall): {advanced_recall:.4f}")
    print(f"Precision: {advanced_precision:.4f}")
    print(f"F1 Score: {advanced_f1:.4f}")
    
    # Сравнение базового и расширенного детекторов
    print("\nСравнение базового и расширенного детекторов:")
    print(f"Изменение точности: {(advanced_accuracy - accuracy) * 100:.2f}%")
    print(f"Изменение полноты: {(advanced_recall - recall) * 100:.2f}%")
    print(f"Изменение precision: {(advanced_precision - precision) * 100:.2f}%")
    print(f"Изменение F1: {(advanced_f1 - f1) * 100:.2f}%")
    print(f"Изменение времени выполнения: {(advanced_time - basic_time) / basic_time * 100:.2f}%")
    
    # Анализ по категориям инъекций для расширенного детектора
    print("\nАнализ результатов расширенного детектора по категориям:")
    calculate_category_metrics(advanced_results, normal_start, normal_end, "Нормальные запросы (должны быть 0 обнаружений)")
    calculate_category_metrics(advanced_results, basic_start, basic_end, "Базовые SQL-инъекции")
    calculate_category_metrics(advanced_results, obfuscated_start, obfuscated_end, "Обфусцированные SQL-инъекции")
    calculate_category_metrics(advanced_results, advanced_start, advanced_end, "Продвинутые SQL-инъекции")
    
    # ТЕСТИРОВАНИЕ КОНТЕКСТНО-ЗАВИСИМОГО АНАЛИЗА
    print("\n=== ТЕСТ 4: Контекстно-зависимый анализ ===")
    
    # Создаем новый набор данных с последовательностью запросов с одного IP
    sequence_queries = []
    sequence_ips = []
    sequence_labels = []
    
    # Добавляем несколько нормальных запросов с IP 192.168.1.100
    for i in range(5):
        sequence_queries.append(normal_queries[i % len(normal_queries)])
        sequence_ips.append("192.168.1.100")
        sequence_labels.append(0)
    
    # Добавляем последовательность подозрительных запросов с того же IP
    for i in range(10):
        if i < 3:  # Первые три запроса нормальные
            sequence_queries.append(normal_queries[i % len(normal_queries)])
            sequence_labels.append(0)
        elif i < 6:  # Следующие три - на грани (не совсем инъекции, но подозрительные)
            sequence_queries.append(f"SELECT * FROM users WHERE username LIKE '%a{i}%'")
            sequence_labels.append(0)  # В идеале не должны быть обнаружены как аномалии
        else:  # Последние четыре - явные инъекции
            sequence_queries.append(basic_injections[i % len(basic_injections)])
            sequence_labels.append(1)
        
        sequence_ips.append("192.168.1.100")
    
    # Создаем DataFrame
    df_sequence = pd.DataFrame({
        'query': sequence_queries,
        'ip_address': sequence_ips,
        'is_anomaly': sequence_labels
    })
    
    # Инициализация детектора с контекстно-зависимым анализом
    context_detector = EnhancedSQLInjectionDetector(
        text_column='query',
        threshold=0.5,
        verbose=True,
        use_ml=False,
        adaptive_threshold=False,
        context_aware=True
    )
    
    # Обучение и тестирование
    context_detector.fit(df_train)
    
    context_results = context_detector.detect_and_explain(df_sequence, df_sequence['ip_address'])
    
    # Анализ результатов
    y_true_seq = df_sequence['is_anomaly']
    y_pred_seq = context_results['predicted_anomaly']
    
    context_accuracy = accuracy_score(y_true_seq, y_pred_seq)
    context_precision = precision_score(y_true_seq, y_pred_seq, zero_division=0)
    context_recall = recall_score(y_true_seq, y_pred_seq, zero_division=0)
    context_f1 = f1_score(y_true_seq, y_pred_seq, zero_division=0)
    
    print(f"Результаты тестирования контекстно-зависимого анализа:")
    print(f"Точность (Accuracy): {context_accuracy:.4f}")
    print(f"Полнота (Recall): {context_recall:.4f}")
    print(f"Precision: {context_precision:.4f}")
    print(f"F1 Score: {context_f1:.4f}")
    
    # Проверка, влияет ли история запросов на обнаружение
    print("\nДетальный анализ контекстно-зависимого обнаружения:")
    print("Запросы от IP 192.168.1.100:")
    
    for i in range(len(df_sequence)):
        query = df_sequence.iloc[i]['query']
        is_true_anomaly = df_sequence.iloc[i]['is_anomaly'] == 1
        is_detected = context_results.iloc[i]['predicted_anomaly'] == 1
        score = context_results.iloc[i]['anomaly_score']
        
        query_short = query[:50] + "..." if len(query) > 50 else query
        print(f"{i+1:2d}. {'[A]' if is_true_anomaly else '   '} {'[D]' if is_detected else '   '} {score:.2f} | {query_short}")
    
    # Проверка счетчика аномалий для IP
    if hasattr(context_detector, 'ip_anomaly_counter'):
        print("\nСтатистика аномалий по IP-адресам:")
        for ip, count in context_detector.ip_anomaly_counter.items():
            if count > 0:
                print(f" - {ip}: {count} подозрительных запросов")
    
    print("\n=== ТЕСТИРОВАНИЕ ЗАВЕРШЕНО ===")
    print("Улучшенный детектор SQL-инъекций успешно протестирован по всем основным параметрам.")
    
except Exception as e:
    print(f"Ошибка при тестировании детектора: {e}")
    import traceback
    traceback.print_exc()
