
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

class ImprovedSQLInjectionDetector:
    """
    Улучшенная версия детектора SQL-инъекций с расширенными паттернами 
    и более гибкой системой оценки аномалий
    """
    
    def __init__(self, 
                text_column='query', 
                threshold=0.3,
                verbose=False,
                use_ml=True,
                adaptive_threshold=True,
                context_aware=True):
        """
        Инициализация улучшенного детектора SQL-инъекций
        """
        self.text_column = text_column
        self.threshold = threshold
        self.verbose = verbose
        self.use_ml = use_ml
        self.adaptive_threshold = adaptive_threshold
        self.context_aware = context_aware
        
        # Увеличенный набор паттернов для обнаружения SQL-инъекций
        self.patterns = self._initialize_patterns()
        
        # Веса для различных категорий паттернов
        self.pattern_weights = {
            "Boolean-based": 0.25,
            "UNION-based": 0.35,
            "Destructive": 0.4,
            "Command Execution": 0.45,
            "Obfuscation": 0.2,
            "Time-based": 0.3,
            "Blind": 0.25,
            "Error-based": 0.3
        }
        
        # Сохраняем нормальные запросы и их характеристики
        self.normal_queries = []
        self.normal_query_stats = {}
        self.ip_behavior = {}
        
        if verbose:
            print(f"Детектор инициализирован с {sum(len(patterns) for patterns in self.patterns.values())} паттернами обнаружения.")
    
    def _initialize_patterns(self):
        """
        Расширенный набор паттернов для обнаружения SQL-инъекций,
        сгруппированный по категориям
        """
        return {
            # Boolean-based инъекции (простые условия)
            "Boolean-based": [
                (r"'\s*OR\s+'1'\s*=\s*'1", "OR '1'='1' condition"),
                (r"'\s*OR\s+[0-9]+\s*=\s*[0-9]+", "OR numeric condition"),
                (r"'\s*OR\s+'[a-z0-9]+'\s*=\s*'[a-z0-9]+'", "OR string condition"),
                (r"'\s*OR\s+1\s*>\s*0", "OR 1>0 condition"),
                (r"'\s*OR\s+[a-z]+\s*([<>=]|LIKE)\s*[a-z]+", "OR comparison"),
                (r"'\s*OR\s+TRUE\s*--", "OR TRUE condition"),
                (r"'\s*OR\s+[0-9]+\s*<>\s*[0-9]+", "OR not equal condition"),
                (r"'\s*OR\s+'[^']+'\s+LIKE\s+'%[^']*%'", "OR LIKE condition")
            ],
            
            # UNION-based инъекции
            "UNION-based": [
                (r"UNION\s+(?:ALL\s+)?SELECT", "UNION SELECT"),
                (r"UNION\s+SELECT\s+NULL", "UNION SELECT NULL"),
                (r"UNION\s+SELECT\s+[0-9]+", "UNION SELECT numbers"),
                (r"\d+\s+UNION\s+SELECT", "numeric UNION"),
                (r"'\s+UNION\s+SELECT", "string UNION"),
                (r"UNION\s+SELECT\s+CONCAT\(", "UNION SELECT CONCAT")
            ],
            
            # Destructive операции
            "Destructive": [
                (r";\s*DROP\s+TABLE", "DROP TABLE"),
                (r";\s*TRUNCATE\s+TABLE", "TRUNCATE TABLE"),
                (r";\s*DELETE\s+FROM", "DELETE FROM"),
                (r";\s*ALTER\s+TABLE", "ALTER TABLE"),
                (r";\s*UPDATE\s+[a-z_]+\s+SET", "UPDATE TABLE")
            ],
            
            # Выполнение команд
            "Command Execution": [
                (r"EXEC\s+xp_cmdshell", "xp_cmdshell execution"),
                (r"EXECUTE\s+sp_", "stored procedure execution"),
                (r"EXEC\s+sp_", "stored procedure execution"),
                (r"CALL\s+", "procedure call"),
                (r"EXECUTE\s+IMMEDIATE", "dynamic execution"),
                (r"INTO\s+OUTFILE", "file writing"),
                (r"INTO\s+DUMPFILE", "file dumping"),
                (r"LOAD_FILE\s*\(", "file reading")
            ],
            
            # Техники обфускации
            "Obfuscation": [
                (r"/\*.*?\*/", "inline comment"),
                (r"--", "single-line comment"),
                (r"#", "MySQL comment"),
                (r"%[0-9A-Fa-f]{2}", "URL encoding"),
                (r"CHAR\s*\(\s*\d+\s*\)", "CHAR function"),
                (r"0x[0-9A-Fa-f]+", "hex encoding"),
                (r"CONCAT\s*\(", "string concatenation"),
                (r"\|\|", "string concatenation operator"),
                (r"'\s*\+\s*'", "string concatenation with +"),
                (r"'?\s*\|\|\s*'?", "Oracle string concatenation")
            ],
            
            # Time-based атаки
            "Time-based": [
                (r"SLEEP\s*\(\s*\d+\s*\)", "SLEEP function"),
                (r"BENCHMARK\s*\(", "BENCHMARK function"),
                (r"WAITFOR\s+DELAY", "WAITFOR DELAY"),
                (r"pg_sleep", "PostgreSQL sleep function")
            ],
            
            # Blind SQL инъекции
            "Blind": [
                (r"SUBSTR\s*\(", "SUBSTR function"),
                (r"SUBSTRING\s*\(", "SUBSTRING function"),
                (r"ASCII\s*\(", "ASCII function"),
                (r"ORD\s*\(", "ORD function"),
                (r"MID\s*\(", "MID function"),
                (r"LIKE\s+BINARY", "binary comparison")
            ],
            
            # Error-based инъекции
            "Error-based": [
                (r"EXTRACTVALUE\s*\(", "XML function exploitation"),
                (r"UPDATEXML\s*\(", "XML function exploitation"),
                (r"exp\s*\(~\s*\(", "Error-producing expression"),
                (r"JSON_EXTRACT", "JSON function exploitation"),
                (r"POLYGON\s*\(", "Geometric function exploitation")
            ]
        }
    
    def _extract_query_features(self, query):
        """
        Извлечение статистических характеристик запроса
        """
        features = {}
        
        # Общая длина запроса
        features['length'] = len(query)
        
        # Количество ключевых слов SQL
        sql_keywords = ['SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 
                       'JOIN', 'GROUP', 'HAVING', 'ORDER', 'LIMIT', 'UNION', 'DROP']
        features['keyword_count'] = sum(1 for keyword in sql_keywords if keyword in query.upper())
        
        # Соотношение специальных символов
        special_chars = "';"=<>()/*-+,"
        features['special_char_ratio'] = sum(1 for char in query if char in special_chars) / max(1, len(query))
        
        # Количество пробелов
        features['space_count'] = query.count(' ')
        
        # Количество комментариев
        features['comment_count'] = len(re.findall(r'(--|#|/\*.*?\*/)', query))
        
        # Соотношение цифр
        features['digit_ratio'] = sum(1 for char in query if char.isdigit()) / max(1, len(query))
        
        return features
    
    def fit(self, data):
        """
        Обучение детектора на нормальных запросах
        """
        if self.text_column not in data.columns:
            raise ValueError(f"Колонка {self.text_column} не найдена в данных")
        
        # Сохраняем нормальные запросы
        self.normal_queries = data[self.text_column].tolist()
        
        # Извлекаем статистические характеристики нормальных запросов
        all_features = []
        for query in self.normal_queries:
            features = self._extract_query_features(query)
            all_features.append(features)
        
        # Рассчитываем среднее и стандартное отклонение для каждой характеристики
        if all_features:
            feature_names = all_features[0].keys()
            for name in feature_names:
                values = [f[name] for f in all_features]
                self.normal_query_stats[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values) or 1.0  # Избегаем деления на ноль
                }
        
        if self.verbose:
            print(f"Детектор обучен на {len(self.normal_queries)} нормальных запросах.")
        
        return self
    
    def detect(self, data, ip_addresses=None):
        """
        Обнаружение SQL-инъекций в запросах
        """
        if self.text_column not in data.columns:
            raise ValueError(f"Колонка {self.text_column} не найдена в данных")
        
        # Копируем входные данные
        results = data.copy()
        
        # Вычисляем оценки и предсказания
        scores = []
        pattern_matches = []
        
        for query in data[self.text_column]:
            # Вычисляем оценку аномальности
            score, matches = self._compute_anomaly_score(query)
            scores.append(score)
            pattern_matches.append(matches)
        
        # Добавляем результаты
        results['anomaly_score'] = scores
        results['pattern_matches'] = pattern_matches
        
        # Применяем порог для классификации
        threshold = self.threshold
        
        # Если включен адаптивный порог, корректируем его на основе распределения оценок
        if self.adaptive_threshold and len(scores) > 5:
            sorted_scores = sorted(scores)
            # Ищем естественный разрыв в распределении
            score_diffs = [sorted_scores[i+1] - sorted_scores[i] for i in range(len(sorted_scores)-1)]
            if score_diffs:
                max_diff_idx = np.argmax(score_diffs)
                # Если найден значительный разрыв, используем его как порог
                if score_diffs[max_diff_idx] > 0.1:
                    adaptive_threshold = sorted_scores[max_diff_idx + 1]
                    # Используем адаптивный порог, если он выше базового и не слишком высокий
                    if self.threshold <= adaptive_threshold <= 0.7:
                        threshold = adaptive_threshold
                        if self.verbose:
                            print(f"Установлен адаптивный порог: {threshold:.3f}")
        
        # Классификация
        results['predicted_anomaly'] = (results['anomaly_score'] >= threshold).astype(int)
        
        # Если включен контекстный анализ и предоставлены IP-адреса
        if self.context_aware and ip_addresses is not None:
            self._adjust_with_context(results, ip_addresses)
        
        return results
    
    def detect_and_explain(self, data, ip_addresses=None):
        """
        Обнаружение SQL-инъекций с детальными объяснениями
        """
        # Получаем базовые результаты обнаружения
        results = self.detect(data, ip_addresses)
        
        # Добавляем объяснения и рекомендации
        explanations = []
        risk_levels = []
        attack_types = []
        recommendations = []
        
        for i, row in results.iterrows():
            query = data.iloc[i][self.text_column]
            score = row['anomaly_score']
            matches = row['pattern_matches']
            is_anomaly = row['predicted_anomaly'] == 1
            
            if is_anomaly:
                explanation, risk, attack, recommendation = self._explain_anomaly(
                    query, score, matches
                )
            else:
                explanation = "Нормальный запрос без признаков SQL-инъекции."
                risk = "Low"
                attack = "None"
                recommendation = "Нет специальных рекомендаций."
            
            explanations.append(explanation)
            risk_levels.append(risk)
            attack_types.append(attack)
            recommendations.append(recommendation)
        
        # Добавляем новые колонки
        results['explanation'] = explanations
        results['risk_level'] = risk_levels
        results['attack_type'] = attack_types
        results['recommendations'] = recommendations
        
        # Удаляем промежуточные результаты
        results.drop('pattern_matches', axis=1, inplace=True)
        
        return results
    
    def _compute_anomaly_score(self, query):
        """
        Вычисление оценки аномальности для запроса
        """
        # Инициализация
        category_scores = {}
        matches = {}
        
        # Поиск совпадений по всем категориям паттернов
        for category, patterns in self.patterns.items():
            category_matches = []
            
            for pattern, description in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    category_matches.append(description)
            
            if category_matches:
                # Сохраняем найденные совпадения
                matches[category] = category_matches
                
                # Вычисляем оценку для этой категории с учетом веса категории
                weight = self.pattern_weights.get(category, 0.2)
                # Корректируем вес в зависимости от количества совпадений
                adjusted_weight = min(weight * len(category_matches), 0.7)
                category_scores[category] = adjusted_weight
# Продолжение содержимого improved_sql_injection_detector.py
        
        # Если есть совпадения хотя бы по одной категории
        if category_scores:
            # Комбинированная оценка - взвешенная сумма всех категорий
            combined_score = sum(category_scores.values())
            
            # Нормализуем итоговую оценку до диапазона [0, 1]
            final_score = min(combined_score, 1.0)
            
            # Регулируем оценку с учётом дополнительных характеристик
            if self.use_ml and self.normal_query_stats:
                ml_adjustment = self._compute_ml_adjustment(query)
                final_score = min(final_score + ml_adjustment, 1.0)
            
            return final_score, matches
        
        # Если не найдено ни одного паттерна
        return 0.0, {}
    
    def _compute_ml_adjustment(self, query):
        """
        Вычисление корректировки оценки на основе статистических характеристик
        """
        # Извлекаем характеристики запроса
        features = self._extract_query_features(query)
        
        # Вычисляем Z-оценки для характеристик
        z_scores = {}
        for name, value in features.items():
            if name in self.normal_query_stats:
                stats = self.normal_query_stats[name]
                z_scores[name] = abs((value - stats['mean']) / stats['std'])
        
        # Если нет Z-оценок, не корректируем
        if not z_scores:
            return 0.0
        
        # Вычисляем корректировку на основе максимальной Z-оценки
        max_z = max(z_scores.values())
        
        # Нелинейная функция для преобразования Z-оценки в корректировку
        if max_z < 2.0:
            # Небольшие отклонения - незначительная корректировка
            return 0.0
        elif max_z < 3.0:
            # Средние отклонения - небольшая корректировка
            return 0.05
        elif max_z < 5.0:
            # Большие отклонения - средняя корректировка
            return 0.1
        else:
            # Очень большие отклонения - значительная корректировка
            return 0.2
    
    def _adjust_with_context(self, results, ip_addresses):
        """
        Корректировка результатов с учетом контекстной информации (IP-адреса)
        """
        # Обновляем информацию о поведении IP-адресов
        for i, ip in enumerate(ip_addresses):
            score = results['anomaly_score'].iloc[i]
            prediction = results['predicted_anomaly'].iloc[i]
            
            if ip not in self.ip_behavior:
                self.ip_behavior[ip] = {
                    'query_count': 1,
                    'anomaly_count': prediction,
                    'avg_score': score,
                    'last_queries': [results.iloc[i][self.text_column]]
                }
            else:
                behavior = self.ip_behavior[ip]
                behavior['query_count'] += 1
                behavior['anomaly_count'] += prediction
                behavior['avg_score'] = (behavior['avg_score'] * (behavior['query_count'] - 1) + score) / behavior['query_count']
                
                # Сохраняем последние запросы (максимум 5)
                behavior['last_queries'].append(results.iloc[i][self.text_column])
                if len(behavior['last_queries']) > 5:
                    behavior['last_queries'].pop(0)
        
        # Корректируем оценки с учетом поведения IP
        for i, ip in enumerate(ip_addresses):
            behavior = self.ip_behavior[ip]
            
            # Частота аномалий от данного IP
            anomaly_ratio = behavior['anomaly_count'] / behavior['query_count']
            
            # Текущая оценка аномальности
            current_score = results.iloc[i]['anomaly_score']
            
            # Поправка на основе частоты аномалий
            if anomaly_ratio > 0.5 and behavior['query_count'] > 3:
                # IP с высокой частотой аномалий - повышаем оценку
                adjustment = 0.15
            elif anomaly_ratio > 0.3 and behavior['query_count'] > 2:
                # IP с средней частотой аномалий - немного повышаем оценку
                adjustment = 0.1
            elif current_score > 0 and current_score < self.threshold:
                # Пограничные случаи - более внимательная проверка
                if behavior['avg_score'] > self.threshold:
                    adjustment = 0.05
                else:
                    adjustment = 0
            else:
                adjustment = 0
            
            # Применяем поправку
            if adjustment > 0:
                new_score = min(current_score + adjustment, 1.0)
                results.at[results.index[i], 'anomaly_score'] = new_score
                
                # Обновляем предсказание, если оценка превысила порог
                if new_score >= self.threshold and results.iloc[i]['predicted_anomaly'] == 0:
                    results.at[results.index[i], 'predicted_anomaly'] = 1
    
    def _explain_anomaly(self, query, score, matches):
        """
        Формирование объяснения для обнаруженной аномалии
        """
        # Определение уровня риска
        if score >= 0.7:
            risk_level = "High"
        elif score >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Определение типа атаки
        if "UNION-based" in matches:
            attack_type = "UNION-based SQL Injection"
        elif "Destructive" in matches:
            attack_type = "Destructive SQL Command"
        elif "Command Execution" in matches:
            attack_type = "OS Command Execution"
        elif "Time-based" in matches:
            attack_type = "Time-based SQL Injection"
        elif "Blind" in matches:
            attack_type = "Blind SQL Injection"
        elif "Error-based" in matches:
            attack_type = "Error-based SQL Injection"
        elif "Boolean-based" in matches:
            attack_type = "Boolean-based SQL Injection"
        else:
            attack_type = "Unknown SQL Injection"
        
        # Формирование объяснения
        explanation = f"[{risk_level} Risk] Обнаружены признаки SQL-инъекции в запросе.
"
        explanation += f"Тип атаки: {attack_type}
"
        explanation += "Обнаруженные подозрительные паттерны:
"
        
        # Добавляем детали по каждой категории
        for category, category_matches in matches.items():
            explanation += f"- {category}:
"
            for match in category_matches:
                explanation += f"  ◦ {match}
"
        
        # Формирование рекомендаций
        recommendation = "Рекомендации по защите:
"
        
        # Общие рекомендации
        recommendation += "• Используйте параметризованные запросы вместо прямой конкатенации строк.
"
        recommendation += "• Используйте принцип наименьших привилегий для пользователей БД.
"
        recommendation += "• Реализуйте строгую валидацию пользовательских входных данных.
"
        
        # Специфичные рекомендации по типу атаки
        if "Command Execution" in matches:
            recommendation += "• Отключите опасные хранимые процедуры, такие как xp_cmdshell.
"
            recommendation += "• Реализуйте строгое разделение доступа к системным функциям.
"
        
        if "UNION-based" in matches:
            recommendation += "• Ограничьте возможные поля для выборки и контролируйте запросы с UNION.
"
            recommendation += "• Используйте ORM-фреймворки для абстрагирования от прямых SQL-запросов.
"
        
        if "Destructive" in matches:
            recommendation += "• Настройте резервное копирование данных.
"
            recommendation += "• Реализуйте механизм аудита для всех деструктивных операций.
"
        
        if "Obfuscation" in matches:
            recommendation += "• Используйте специализированные WAF для обнаружения обфусцированных атак.
"
            recommendation += "• Внедрите многоуровневую защиту с несколькими методами обнаружения.
"
        
        return explanation, risk_level, attack_type, recommendation
