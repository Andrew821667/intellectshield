
import re
import pandas as pd
import numpy as np
import urllib.parse
import html
import base64
from typing import List, Dict, Tuple, Optional, Union, Any
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
import pickle
import os
import json
import time

class EnhancedSQLInjectionDetector:
    """
    Улучшенный детектор SQL-инъекций с расширенными возможностями обнаружения,
    машинным обучением и обработкой обфускации.
    """
    
    def __init__(self, 
                 text_column='query', 
                 threshold=0.5, 
                 verbose=False, 
                 use_ml=True,
                 adaptive_threshold=True,
                 context_aware=True,
                 vulnerabilities_db_path=None):
        """
        Инициализация детектора SQL-инъекций с расширенными параметрами.
        
        Args:
            text_column (str): Название столбца с текстом запроса
            threshold (float): Пороговое значение для классификации аномалий
            verbose (bool): Подробный вывод информации
            use_ml (bool): Использовать машинное обучение для улучшения обнаружения
            adaptive_threshold (bool): Использовать адаптивные пороги
            context_aware (bool): Учитывать контекст запросов
            vulnerabilities_db_path (str): Путь к базе данных уязвимостей
        """
        self.text_column = text_column
        self.threshold = threshold
        self.base_threshold = threshold  # Сохраняем исходный порог для адаптивного подхода
        self.verbose = verbose
        self.use_ml = use_ml
        self.adaptive_threshold = adaptive_threshold
        self.context_aware = context_aware
        
        # История запросов для контекстного анализа
        self.request_history = []
        self.history_size = 100  # Количество запросов для хранения в истории
        
        # Статистика для адаптивных порогов
        self.score_history = []
        self.adaptive_window_size = 100
        
        # Инициализация ML модели
        self.ml_model = None
        self.vectorizer = None
        
        # Счетчик обнаруженных аномалий для каждого IP-адреса
        self.ip_anomaly_counter = defaultdict(int)
        
        # Время последнего запроса для каждого IP
        self.last_request_time = {}
        
        # Загрузка базы данных уязвимостей, если путь указан
        self.vulnerabilities_db = self._load_vulnerabilities_db(vulnerabilities_db_path)
        self.last_db_update = time.time()
        self.db_update_interval = 86400  # 24 часа
        
        # Критические паттерны SQL-инъекций (расширенные)
        self.critical_patterns = [
            # Базовые паттерны
            r"OR 1=1",
            r"OR '1'='1'",
            r"--",
            r"#",
            r"UNION SELECT",
            r"; DROP",
            r"xp_cmdshell",
            r"WAITFOR DELAY",
            
            # Дополнительные критические паттерны
            r"OR \d+=\d+",                      # Числовые сравнения
            r"OR ['\"].*?['\"]=['\"].*?['\"]",  # Строковые сравнения
            r"UNION ALL SELECT",                # UNION ALL
            r"@@version",                       # Информация о версии
            r"sys\.databases",                  # Системные таблицы
            r"information_schema\.",            # Схема метаданных
            r"CAST\s*\(",                       # Преобразование типов
            r"EXEC\s*\(",                       # Выполнение динамического SQL
            r"sp_executesql",                   # Выполнение динамического SQL
            r"BULK INSERT",                     # Массовая вставка
            r"INTO OUTFILE",                    # Запись в файл
            r"INTO DUMPFILE",                   # Запись дампа
            r"LOAD_FILE",                       # Загрузка файла
            r"BENCHMARK\s*\(",                  # Тестирование производительности
            r"SLEEP\s*\(",                      # Функция задержки MySQL
            r"PG_SLEEP\s*\(",                   # Функция задержки PostgreSQL
            r"DBMS_PIPE\.RECEIVE_MESSAGE",      # Функция задержки Oracle
        ]
        
        # Вторичные паттерны (расширенные)
        self.secondary_patterns = [
            # Существующие паттерны
            r"OR 1>",
            r"LIKE '%",
            r"; ALTER",
            r"ORDER BY \d+",
            r"GROUP BY .* HAVING",
            
            # Дополнительные вторичные паттерны
            r"SELECT\s+@@",                    # Переменные сервера
            r"SUBSTRING\s*\(",                 # Извлечение подстроки
            r"CONCAT\s*\(",                    # Конкатенация строк
            r"CONVERT\s*\(",                   # Конвертация типов
            r"AND\s+\d+",                      # Потенциальные проверки условий
            r"CASE\s+WHEN",                    # CASE выражения
            r"IF\s*\(",                        # IF выражения
            r"CHAR\s*\(",                      # Функция CHAR для кодирования символов
            r"ASCII\s*\(",                     # Функция ASCII
            r"UPDATE.+SET",                    # Обновление данных
            r"DELETE.+FROM",                   # Удаление данных
            r"INSERT.+INTO",                   # Вставка данных
            r"REPLACE.+INTO",                  # Замена данных
            r"LIMIT\s+\d+\s*,\s*\d+",          # LIMIT с двумя параметрами
            r"PROCEDURE\s+ANALYSE",            # Процедура анализа
            r"IS\s+NULL",                      # Проверка на NULL
            r"IS\s+NOT\s+NULL",                # Проверка на NOT NULL
        ]
        
        # Категории атак (расширенные)
        self.attack_categories = {
            'boolean_based': "Boolean-based SQL Injection",
            'union_based': "UNION-based SQL Injection",
            'time_based': "Time-based Blind SQL Injection",
            'error_based': "Error-based SQL Injection", 
            'destructive': "Destructive SQL Command",
            'command_exec': "OS Command Execution",
            'stacked_queries': "Stacked Queries",
            'out_of_band': "Out-of-Band SQL Injection"
        }
        
        # Рекомендации (расширенные)
        self.security_recommendations = {
            'boolean_based': "Используйте параметризованные запросы вместо прямой конкатенации строк.",
            'union_based': "Применяйте ORM-фреймворки и строгую валидацию входных данных.",
            'time_based': "Установите тайм-ауты для SQL-запросов.",
            'error_based': "Скрывайте детали ошибок БД от пользователей, используйте обобщенные сообщения об ошибках.",
            'destructive': "Ограничьте права доступа пользователя БД.",
            'command_exec': "Никогда не используйте xp_cmdshell в продакшене.",
            'stacked_queries': "Используйте параметризованные запросы и проверяйте, что драйвер БД не выполняет несколько запросов.",
            'out_of_band': "Настройте файрвол для блокировки исходящих соединений от сервера БД."
        }
        
        self.default_recommendation = "Используйте параметризованные запросы и валидацию ввода."
        
        # Техники обфускации для декодирования
        self.obfuscation_techniques = [
            self._decode_url,
            self._decode_hex,
            self._decode_unicode,
            self._decode_base64,
            self._decode_comment_blocks,
            self._normalize_whitespace,
            self._decode_case_variations,
            self._decode_concatenation,
            self._decode_char_encoding
        ]
    
    def fit(self, data):
        """
        Обучение детектора на данных.
        
        Args:
            data (pd.DataFrame): Данные для обучения
            
        Returns:
            self: Обученный детектор
        """
        if self.verbose:
            print(f"Детектор инициализирован с {len(self.critical_patterns)} критическими "
                  f"и {len(self.secondary_patterns)} вторичными паттернами.")
        
        if self.use_ml and len(data) >= 10:  # Достаточно данных для ML
            if self.verbose:
                print("Обучение модели машинного обучения...")
            
            # Извлечение текстов запросов
            queries = data[self.text_column].fillna('').astype(str).tolist()
            
            # Создание TF-IDF векторизатора для преобразования текста в числовые признаки
            self.vectorizer = TfidfVectorizer(
                max_features=100,     # Количество признаков
                ngram_range=(1, 3),   # Униграммы, биграммы и триграммы
                analyzer='char',      # Анализируем на уровне символов
                lowercase=True,
                min_df=2,
                max_df=0.9
            )
            
            # Обучение векторизатора и преобразование данных
            X = self.vectorizer.fit_transform(queries)
            
            # Создание и обучение модели Isolation Forest для обнаружения аномалий
            self.ml_model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination='auto',
                random_state=42
            )
            
            self.ml_model.fit(X)
            
            if self.verbose:
                print(f"Модель машинного обучения обучена на {len(queries)} запросах.")
        
        return self
    
    def predict(self, data, ip_addresses=None):
        """
        Предсказание SQL-инъекций.
        
        Args:
            data (pd.DataFrame): Данные для анализа
            ip_addresses (pd.Series, optional): IP-адреса для контекстного анализа
            
        Returns:
            pd.DataFrame: Результаты анализа
        """
        # Проверка обновления базы данных уязвимостей
        self._check_db_update()
        
        # Копирование данных для результатов
        result = data.copy()
        result['anomaly_score'] = 0.0
        result['predicted_anomaly'] = 0
        result['threat_level'] = None
        result['attack_type'] = None
        result['detected_by'] = None
        
        # Если предоставлены IP-адреса, добавляем их в результаты
        if ip_addresses is not None and len(ip_addresses) == len(data):
            result['ip_address'] = ip_addresses
        
        # Используем ML модель, если она обучена
        ml_scores = None
        if self.use_ml and self.ml_model is not None and self.vectorizer is not None:
            # Извлечение текстов запросов
            queries = data[self.text_column].fillna('').astype(str).tolist()
            
            # Преобразование текста в числовые признаки
            X = self.vectorizer.transform(queries)
            
            # Предсказания аномалий (-1 для аномалий, 1 для нормальных)
            # Преобразуем в оценки от 0 до 1
            raw_scores = self.ml_model.decision_function(X)
            ml_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        current_time = time.time()
        
        # Анализ каждого запроса
        for idx, row in data.iterrows():
            text = row[self.text_column]
            if not isinstance(text, str):
                continue
            
            # Получение IP-адреса, если доступно
            ip_address = ip_addresses.iloc[idx] if ip_addresses is not None else None
            
            # Проверка частоты запросов с одного IP
            if ip_address and ip_address in self.last_request_time:
                time_diff = current_time - self.last_request_time[ip_address]
                # Если запросы от IP приходят слишком часто (менее 0.5 секунды между запросами)
                if time_diff < 0.5:
                    self.ip_anomaly_counter[ip_address] += 1
            
            if ip_address:
                self.last_request_time[ip_address] = current_time
            
            # Декодирование обфускации
            deobfuscated_text = self._deobfuscate(text)
            
            # Оригинальный текст и декодированный текст для поиска
            texts_to_check = [text.lower(), deobfuscated_text.lower()]
            
            # Проверка критических паттернов
            pattern_result = self._check_patterns(texts_to_check)
            
            # Оценка аномалии на основе паттернов
            pattern_score = pattern_result['score']
            matched_patterns = pattern_result['matched_patterns']
            
            # Добавление ML оценки, если доступна
            ml_score = ml_scores[idx] if ml_scores is not None else 0
            
            # Объединение оценок
            if self.use_ml and self.ml_model is not None:
                # Комбинирование оценок (придаем больший вес паттернам)
                combined_score = 0.7 * pattern_score + 0.3 * ml_score
            else:
                combined_score = pattern_score
            
            # Контекстная информация
            context_modifier = 0
            
            # Проверка контекста, если включена
            if self.context_aware and ip_address:
                # Штраф за подозрительную активность с IP
                if self.ip_anomaly_counter[ip_address] > 5:
                    context_modifier = min(0.3, 0.05 * self.ip_anomaly_counter[ip_address])
            
            # Применение контекстного модификатора
            final_score = min(1.0, combined_score + context_modifier)
            
            # Вычисление текущего порога
            current_threshold = self.threshold
            
            # Адаптивные пороги
            if self.adaptive_threshold and len(self.score_history) > 10:
                # Вычисление адаптивного порога
                current_threshold = self._calculate_adaptive_threshold()
            
            # Проверка на аномалию и определение типа атаки
            is_anomaly = final_score >= current_threshold
            
            # Определение источника обнаружения
            detected_by = []
            if pattern_score >= current_threshold:
                detected_by.append("pattern")
            if ml_score >= current_threshold:
                detected_by.append("ml")
            if context_modifier > 0 and final_score >= current_threshold:
                detected_by.append("context")
            
            # Определение типа атаки
            attack_type = None
            if is_anomaly:
                attack_type = self._determine_attack_type(text, deobfuscated_text, matched_patterns)
            
            # Определение уровня угрозы
            threat_level = None
            if is_anomaly:
                if final_score >= 0.9:
                    threat_level = 'Critical'
                elif final_score >= 0.7:
                    threat_level = 'High'
                elif final_score >= 0.5:
                    threat_level = 'Medium'
                else:
                    threat_level = 'Low'
            
            # Обновление результатов
            result.loc[idx, 'anomaly_score'] = final_score
            result.loc[idx, 'predicted_anomaly'] = 1 if is_anomaly else 0
            result.loc[idx, 'threat_level'] = threat_level
            result.loc[idx, 'attack_type'] = attack_type
            result.loc[idx, 'detected_by'] = ','.join(detected_by) if detected_by else None
            
            # Сохранение в историю для адаптивных порогов
            self.score_history.append(final_score)
            if len(self.score_history) > self.adaptive_window_size:
                self.score_history.pop(0)
            
            # Сохранение в историю запросов для контекстного анализа
            if self.context_aware:
                self.request_history.append({
                    'text': text,
                    'deobfuscated': deobfuscated_text,
                    'ip': ip_address,
                    'timestamp': current_time,
                    'is_anomaly': is_anomaly,
                    'score': final_score
                })
                
                # Ограничение размера истории
                if len(self.request_history) > self.history_size:
                    self.request_history.pop(0)
        
        return result
    
    def detect_and_explain(self, data, ip_addresses=None):
        """
        Обнаружение и объяснение SQL-инъекций.
        
        Args:
            data (pd.DataFrame): Данные для анализа
            ip_addresses (pd.Series, optional): IP-адреса для контекстного анализа
            
        Returns:
            pd.DataFrame: Результаты анализа с объяснениями
        """
        # Получение предсказаний
        result = self.predict(data, ip_addresses)
        
        # Добавление объяснений и рекомендаций
        explanations = []
        recommendations = []
        obfuscation_details = []
        
        for idx, row in result.iterrows():
            if row['predicted_anomaly'] == 1:
                text = data.loc[idx, self.text_column]
                deobfuscated_text = self._deobfuscate(text)
                
                # Сохранение информации об обфускации
                obfuscation_info = self._get_obfuscation_info(text, deobfuscated_text)
                obfuscation_details.append(obfuscation_info if obfuscation_info else None)
                
                explanation = self._generate_explanation(
                    text, 
                    deobfuscated_text,
                    row['threat_level'], 
                    row['attack_type'],
                    row['detected_by']
                )
                recommendation = self._generate_recommendation(row['attack_type'])
                
                explanations.append(explanation)
                recommendations.append(recommendation)
            else:
                explanations.append(None)
                recommendations.append(None)
                obfuscation_details.append(None)
        
        result['explanation'] = explanations
        result['recommendation'] = recommendations
        result['obfuscation_details'] = obfuscation_details
        
        return result
    
    def _deobfuscate(self, text: str) -> str:
        """
        Декодирование различных техник обфускации в SQL-запросе.
        
        Args:
            text (str): Исходный SQL-запрос
            
        Returns:
            str: Декодированный SQL-запрос
        """
        if not isinstance(text, str):
            return ""
        
        # Применение всех техник декодирования
        decoded = text
        
        # Итеративная обработка: применяем все техники, пока запрос не перестанет меняться
        prev_decoded = ""
        iteration = 0
        max_iterations = 5  # Ограничение для предотвращения бесконечного цикла
        
        while prev_decoded != decoded and iteration < max_iterations:
            prev_decoded = decoded
            
            for technique in self.obfuscation_techniques:
                decoded = technique(decoded)
            
            iteration += 1
        
        return decoded
    
    def _decode_url(self, text: str) -> str:
        """Декодирование URL-кодирования."""
        try:
            # Поиск URL-кодированных последовательностей
            if '%' in text:
                return urllib.parse.unquote(text)
            return text
        except Exception:
            return text
    
    def _decode_hex(self, text: str) -> str:
        """Декодирование шестнадцатеричного кодирования."""
        try:
            # Поиск шестнадцатеричных последовательностей в формате 0x...
            pattern = r'0x([0-9a-fA-F]+)'
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    hex_val = match
                    # Проверка, что это четное количество символов для шестнадцатеричного представления
                    if len(hex_val) % 2 == 0:
                        # Преобразование шестнадцатеричной строки в байты
                        bytes_val = bytes.fromhex(hex_val)
                        # Преобразование байтов в ASCII
                        ascii_val = bytes_val.decode('latin-1')  # Используем latin-1 для всех байтов
                        text = text.replace(f"0x{hex_val}", ascii_val)
                except Exception:
                    continue
            
            return text
        except Exception:
            return text
    
    def _decode_unicode(self, text: str) -> str:
        """Декодирование Unicode-escape последовательностей."""
        try:
            # Замена Unicode-escape последовательностей
            pattern = r'(\\u[0-9a-fA-F]{4})'
            matches = re.findall(pattern, text)
            
            for match in matches:
                try:
                    # Преобразование Unicode-escape в символ
                    unicode_char = bytes(match, 'utf-8').decode('unicode_escape')
                    text = text.replace(match, unicode_char)
                except Exception:
                    continue
            
            return text
        except Exception:
            return text
    
    def _decode_base64(self, text: str) -> str:
        """Попытка декодирования Base64."""
        # Поиск потенциальных Base64 последовательностей
        pattern = r'([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?'
        matches = re.findall(pattern, text)
        
        for match_tuple in matches:
            match = ''.join(match_tuple)
            
            # Проверка, что длина соответствует Base64
            if len(match) >= 4 and len(match) % 4 == 0:
                try:
                    # Попытка декодирования Base64
                    decoded = base64.b64decode(match).decode('latin-1')
                    # Проверка, что результат содержит только печатаемые ASCII символы
                    if all(32 <= ord(c) <= 126 for c in decoded):
                        text = text.replace(match, decoded)
                except Exception:
                    continue
        
        return text
    
    def _decode_comment_blocks(self, text: str) -> str:
        """Удаление SQL-комментариев."""
        # Удаление однострочных комментариев --
        text = re.sub(r'--.*?(\n|$)', ' ', text)
        
        # Удаление комментариев #
        text = re.sub(r'#.*?(\n|$)', ' ', text)
        
        # Удаление многострочных комментариев /* ... */
        text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Нормализация пробелов и невидимых символов."""
        # Замена последовательностей пробелов, табуляций, переносов строк на одиночный пробел
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _decode_case_variations(self, text: str) -> str:
        """Нормализация регистра для ключевых SQL-слов."""
        # Список ключевых SQL слов для нормализации
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 
            'DROP', 'CREATE', 'ALTER', 'UNION', 'JOIN', 'HAVING', 
            'GROUP', 'ORDER', 'BY', 'LIMIT', 'PROCEDURE', 'EXEC', 
            'EXECUTE', 'WAITFOR', 'DELAY', 'BENCHMARK'
        ]
        
        # Нормализация регистра ключевых слов
        for keyword in sql_keywords:
            # Регулярное выражение с учетом различных вариантов регистра
            pattern = r'\b' + r''.join(f'[{c.lower()}{c.upper()}]' for c in keyword) + r'\b'
            
            # Замена на ключевое слово в верхнем регистре
            text = re.sub(pattern, keyword, text)
        
        return text
    
    def _decode_concatenation(self, text: str) -> str:
        """Объединение строк в конкатенациях."""
        # Замена конкатенаций строк в формате 'a'+'b' на 'ab'
        pattern = r"'([^']*?)'\s*\+\s*'([^']*?)'"
        
        while re.search(pattern, text):
            text = re.sub(pattern, r"'\1\2'", text)
        
        # Замена конкатенаций в формате CONCAT('a','b') на 'ab'
        pattern = r"CONCAT\s*\(\s*'([^']*?)'\s*,\s*'([^']*?)'\s*\)"
        
        while re.search(pattern, text):
            text = re.sub(pattern, r"'\1\2'", text)
        
        return text
    
    def _decode_char_encoding(self, text: str) -> str:
        """Декодирование функций CHAR() для кодирования символов."""
        # Замена CHAR(n) на соответствующий символ
        pattern = r"CHAR\s*\(\s*(\d+)\s*\)"
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                # Преобразование числового кода в символ
                char_code = int(match)
                if 0 <= char_code <= 127:  # Ограничиваем ASCII
                    char = chr(char_code)
                    text = re.sub(f"CHAR\\s*\\(\\s*{match}\\s*\\)", char, text, flags=re.IGNORECASE)
            except ValueError:
                continue
        
        # Декодирование последовательностей в формате CHAR(65)+CHAR(66)
        pattern = r"CHAR\s*\(\s*(\d+)\s*\)\s*\+\s*CHAR\s*\(\s*(\d+)\s*\)"
        
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for match in matches:
            try:
                char1 = chr(int(match[0]))
                char2 = chr(int(match[1]))
                combined = char1 + char2
                text = re.sub(f"CHAR\\s*\\(\\s*{match[0]}\\s*\\)\\s*\\+\\s*CHAR\\s*\\(\\s*{match[1]}\\s*\\)", 
                              combined, text, flags=re.IGNORECASE)
            except ValueError:
                continue
        
        return text
    
    def _get_obfuscation_info(self, original: str, deobfuscated: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации об использованных техниках обфускации.
        
        Args:
            original (str): Исходный запрос
            deobfuscated (str): Декодированный запрос
            
        Returns:
            Optional[Dict[str, Any]]: Информация об обфускации
        """
        if original == deobfuscated:
            return None
        
        techniques = []
        
        # Проверка различных техник обфускации
        if '%' in original and original != self._decode_url(original):
            techniques.append("URL-кодирование")
        
        if re.search(r'0x[0-9a-fA-F]+', original):
            techniques.append("Шестнадцатеричное кодирование")
        
        if re.search(r'\\u[0-9a-fA-F]{4}', original):
            techniques.append("Unicode-escape последовательности")
        
        if re.search(r'--.*?(\n|$)|#.*?(\n|$)|/\*.*?\*/', original, flags=re.DOTALL):
            techniques.append("SQL-комментарии")
        
        if re.search(r"'\s*\+\s*'", original) or re.search(r"CONCAT\s*\(", original, re.IGNORECASE):
            techniques.append("Конкатенация строк")
        
        if re.search(r"CHAR\s*\(\s*\d+\s*\)", original, re.IGNORECASE):
            techniques.append("Кодирование символов (CHAR)")
        
        # Проверка на регистр ключевых слов
        sql_keywords = ['SELECT', 'UNION', 'FROM', 'WHERE']
        for keyword in sql_keywords:
            pattern = r'\b' + r''.join(f'[{c.lower()}{c.upper()}]' for c in keyword) + r'\b'
            if re.search(pattern, original) and not re.search(r'\b' + keyword + r'\b', original):
                techniques.append("Вариации регистра")
                break
        
        if techniques:
            return {
                "techniques": techniques,
                "original": original,
                "deobfuscated": deobfuscated
            }
        
        return None
    
    def _check_patterns(self, texts_to_check: List[str]) -> Dict[str, Any]:
        """
        Проверка на наличие подозрительных паттернов в текстах.
        
        Args:
            texts_to_check (List[str]): Список текстов для проверки
            
        Returns:
            Dict[str, Any]: Результаты проверки
        """
        score = 0.0
        matched_patterns = []
        
        for text in texts_to_check:
            # Проверка критических паттернов
            for pattern in self.critical_patterns:
                if re.search(pattern.lower(), text, re.IGNORECASE):
                    score = 1.0
                    matched_patterns.append(pattern)
            
            # Если уже найден критический паттерн, прекращаем проверку
            if score >= 1.0:
                break
            
            # Проверка вторичных паттернов
            for pattern in self.secondary_patterns:
                if re.search(pattern.lower(), text, re.IGNORECASE):
                    matched_patterns.append(pattern)
            
            # Вычисление оценки на основе вторичных паттернов
            if matched_patterns and score < 1.0:
                score = min(0.9, len(matched_patterns) * 0.15)
            
            # Проверка несбалансированных кавычек
            single_quotes = text.count("'")
            double_quotes = text.count('"')
            
            if (single_quotes % 2 != 0) or (double_quotes % 2 != 0):
                score = max(score, 0.6)
                matched_patterns.append("Несбалансированные кавычки")
        
        return {
            'score': score,
            'matched_patterns': matched_patterns
        }
    
    def _determine_attack_type(self, text: str, deobfuscated_text: str, matched_patterns: List[str]) -> str:
        """
        Определение типа атаки на основе паттернов и декодированного текста.
        
        Args:
            text (str): Исходный текст
            deobfuscated_text (str): Декодированный текст
            matched_patterns (List[str]): Обнаруженные паттерны
            
        Returns:
            str: Тип атаки
        """
        text_lower = text.lower()
        deobfuscated_lower = deobfuscated_text.lower()
        
        # Поиск характерных признаков различных типов атак
        if 'or 1=1' in deobfuscated_lower or "or '1'='1'" in deobfuscated_lower:
            return 'boolean_based'
        elif 'union select' in deobfuscated_lower or 'union all select' in deobfuscated_lower:
            return 'union_based'
        elif 'waitfor delay' in deobfuscated_lower or 'sleep(' in deobfuscated_lower or 'benchmark(' in deobfuscated_lower:
            return 'time_based'
        elif 'drop table' in deobfuscated_lower or 'drop database' in deobfuscated_lower or 'truncate' in deobfuscated_lower:
            return 'destructive'
        elif 'xp_cmdshell' in deobfuscated_lower or 'exec master.dbo' in deobfuscated_lower:
            return 'command_exec'
        elif ';' in deobfuscated_lower and any(cmd in deobfuscated_lower for cmd in ['select', 'insert', 'update', 'delete']):
            return 'stacked_queries'
        elif 'into outfile' in deobfuscated_lower or 'into dumpfile' in deobfuscated_lower:
            return 'out_of_band'
        elif 'error' in ' '.join(matched_patterns).lower() or 'convert(' in deobfuscated_lower or 'cast(' in deobfuscated_lower:
            return 'error_based'
        else:
            return 'generic_injection'
    
    def _calculate_adaptive_threshold(self) -> float:
        """
        Расчет адаптивного порога на основе истории оценок.
        
        Returns:
            float: Адаптивный порог
        """
        if not self.score_history:
            return self.base_threshold
        
        # Отсортированные оценки аномалий
        sorted_scores = sorted(self.score_history)
        n = len(sorted_scores)
        
        # Если есть явное разделение (гэп) в оценках, используем его как порог
        for i in range(n - 1):
            gap = sorted_scores[i+1] - sorted_scores[i]
            if gap > 0.2 and sorted_scores[i] > 0.3:  # Значительный разрыв
                return sorted_scores[i] + (gap / 2)
        
        # Другой вариант - использовать статистический подход
        # Вычисляем среднее и стандартное отклонение
        mean = sum(self.score_history) / n
        std_dev = (sum((x - mean) ** 2 for x in self.score_history) / n) ** 0.5
        
        # Порог как среднее плюс 2 стандартных отклонения
        adaptive_threshold = mean + 2 * std_dev
        
        # Ограничиваем порог разумными пределами
        return max(0.3, min(0.9, adaptive_threshold))
    
    def _load_vulnerabilities_db(self, db_path: Optional[str]) -> Dict[str, Any]:
        """
        Загрузка базы данных известных уязвимостей.
        
        Args:
            db_path (Optional[str]): Путь к файлу базы данных
            
        Returns:
            Dict[str, Any]: База данных уязвимостей
        """
        if not db_path:
            # Возвращаем пустую базу данных с временной меткой
            return {"last_updated": time.time(), "patterns": {}}
        
        try:
            if os.path.exists(db_path):
                with open(db_path, 'r', encoding='utf-8') as f:
                    db = json.load(f)
                
                if self.verbose:
                    print(f"Загружена база данных уязвимостей с {len(db.get('patterns', {}))} паттернами.")
                
                return db
        except Exception as e:
            if self.verbose:
                print(f"Ошибка при загрузке базы данных уязвимостей: {e}")
        
        # В случае ошибки возвращаем пустую базу данных
        return {"last_updated": time.time(), "patterns": {}}
    
    def _check_db_update(self) -> None:
        """Проверка и обновление базы данных уязвимостей."""
        current_time = time.time()
        
        # Проверяем, нужно ли обновлять базу данных
        if (current_time - self.last_db_update) >= self.db_update_interval:
            # Здесь может быть логика для загрузки обновлений из внешнего источника
            self.last_db_update = current_time
    
    def _generate_explanation(self, text: str, deobfuscated_text: str, threat_level: str, 
                              attack_type: str, detected_by: Optional[str]) -> str:
        """
        Генерация объяснения для аномалии.
        
        Args:
            text (str): Исходный текст
            deobfuscated_text (str): Декодированный текст
            threat_level (str): Уровень угрозы
            attack_type (str): Тип атаки
            detected_by (Optional[str]): Способ обнаружения
            
        Returns:
            str: Объяснение
        """
        explanation_parts = []
        
        # Добавляем информацию об уровне угрозы
        explanation_parts.append(f"[{threat_level} Risk] Обнаружены признаки SQL-инъекции в запросе.")
        
        # Добавляем категорию атаки
        if attack_type in self.attack_categories:
            explanation_parts.append(f"Тип атаки: {self.attack_categories[attack_type]}")
        else:
            explanation_parts.append("Тип атаки: Неклассифицированная SQL-инъекция")
        
        # Добавляем информацию о способе обнаружения
        if detected_by:
            detection_methods = {
                "pattern": "Обнаружены подозрительные паттерны",
                "ml": "Обнаружено машинным обучением",
                "context": "Обнаружено на основе контекстного анализа"
            }
            
            detection_info = []
            for method in detected_by.split(','):
                if method in detection_methods:
                    detection_info.append(detection_methods[method])
            
            if detection_info:
                explanation_parts.append("Метод обнаружения: " + ", ".join(detection_info))
        
        # Находим все совпадающие паттерны
        matching_patterns = []
        
        for pattern in self.critical_patterns + self.secondary_patterns:
            if re.search(pattern.lower(), text.lower(), re.IGNORECASE) or \
               re.search(pattern.lower(), deobfuscated_text.lower(), re.IGNORECASE):
                matching_patterns.append(pattern)
        
        if matching_patterns:
            explanation_parts.append("Обнаруженные подозрительные паттерны:")
            for i, pattern in enumerate(matching_patterns[:3]):
                explanation_parts.append(f" - Паттерн {i+1}: {pattern}")
            
            if len(matching_patterns) > 3:
                explanation_parts.append(f" - ... и еще {len(matching_patterns) - 3} паттернов")
        
        # Если обнаружена обфускация
        if text != deobfuscated_text:
            explanation_parts.append("\nОбнаружена обфускация кода! Декодированный запрос:")
            explanation_parts.append(f"```\n{deobfuscated_text}\n```")
        
        return "\n".join(explanation_parts)
    
    def _generate_recommendation(self, attack_type: Optional[str]) -> str:
        """
        Генерация рекомендаций по безопасности.
        
        Args:
            attack_type (Optional[str]): Тип атаки
            
        Returns:
            str: Рекомендации
        """
        recommendation_parts = ["Рекомендации по защите:"]
        
        # Добавляем рекомендации по типу атаки
        if attack_type and attack_type in self.security_recommendations:
            recommendation_parts.append(f"• {self.security_recommendations[attack_type]}")
        else:
            recommendation_parts.append(f"• {self.default_recommendation}")
        
        # Общие рекомендации
        recommendation_parts.append("• Используйте принцип наименьших привилегий для пользователей БД.")
        recommendation_parts.append("• Реализуйте строгую валидацию пользовательских входных данных.")
        recommendation_parts.append("• Внедрите мониторинг и логирование SQL-запросов для выявления аномалий.")
        recommendation_parts.append("• Рассмотрите использование WAF (Web Application Firewall) для дополнительной защиты.")
        
        return "\n".join(recommendation_parts)
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели в файл.
        
        Args:
            path (str): Путь для сохранения
        """
        model_data = {
            'text_column': self.text_column,
            'threshold': self.threshold,
            'base_threshold': self.base_threshold,
            'verbose': self.verbose,
            'use_ml': self.use_ml,
            'adaptive_threshold': self.adaptive_threshold,
            'context_aware': self.context_aware,
            'vectorizer': self.vectorizer,
            'ml_model': self.ml_model,
            'critical_patterns': self.critical_patterns,
            'secondary_patterns': self.secondary_patterns,
            'attack_categories': self.attack_categories,
            'security_recommendations': self.security_recommendations,
            'default_recommendation': self.default_recommendation
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if self.verbose:
            print(f"Модель сохранена в {path}")
    
    def load_model(self, path: str) -> None:
        """
        Загрузка модели из файла.
        
        Args:
            path (str): Путь к файлу модели
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.text_column = model_data['text_column']
        self.threshold = model_data['threshold']
        self.base_threshold = model_data['base_threshold']
        self.verbose = model_data['verbose']
        self.use_ml = model_data['use_ml']
        self.adaptive_threshold = model_data['adaptive_threshold']
        self.context_aware = model_data['context_aware']
        self.vectorizer = model_data['vectorizer']
        self.ml_model = model_data['ml_model']
        self.critical_patterns = model_data['critical_patterns']
        self.secondary_patterns = model_data['secondary_patterns']
        self.attack_categories = model_data['attack_categories']
        self.security_recommendations = model_data['security_recommendations']
        self.default_recommendation = model_data['default_recommendation']
        
        if self.verbose:
            print(f"Модель загружена из {path}")

    def add_custom_patterns(self, critical_patterns=None, secondary_patterns=None) -> None:
        """
        Добавление пользовательских паттернов для обнаружения.
        
        Args:
            critical_patterns (List[str], optional): Критические паттерны
            secondary_patterns (List[str], optional): Вторичные паттерны
        """
        if critical_patterns:
            self.critical_patterns.extend(critical_patterns)
        
        if secondary_patterns:
            self.secondary_patterns.extend(secondary_patterns)
    
    def performance_metrics(self, data: pd.DataFrame, true_labels_column: str) -> Dict[str, float]:
        """
        Расчет метрик производительности детектора.
        
        Args:
            data (pd.DataFrame): Данные для анализа
            true_labels_column (str): Столбец с истинными метками
            
        Returns:
            Dict[str, float]: Метрики производительности
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Получение предсказаний
        results = self.predict(data)
        
        # Истинные метки
        y_true = data[true_labels_column]
        
        # Предсказанные метки
        y_pred = results['predicted_anomaly']
        
        # Оценки аномалий
        y_scores = results['anomaly_score']
        
        # Расчет метрик
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUC-ROC, если есть хотя бы один положительный и один отрицательный класс
        if len(set(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        
        return metrics
