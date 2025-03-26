import re
import pandas as pd
import numpy as np
import urllib.parse
import html
import base64
from typing import List, Dict, Tuple, Optional, Union, Any, Set
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import json
import time
import warnings
from difflib import SequenceMatcher

class EnhancedSQLInjectionDetectorV4:
    """
    Улучшенная версия 4 детектора SQL-инъекций с расширенными возможностями обнаружения,
    улучшенным обнаружением обфусцированных запросов и оптимизированными адаптивными порогами.
    
    Ключевые улучшения:
    1. Улучшенный механизм обнаружения разделенных строк и обфускации
    2. Оптимизированный алгоритм ML для продвинутых инъекций
    3. Улучшенные адаптивные пороги
    4. Оптимизированный контекстно-зависимый анализ
    """
    
    def __init__(self, 
                 text_column='query', 
                 threshold=0.5, 
                 verbose=False, 
                 use_ml=True,
                 adaptive_threshold=True,
                 context_aware=True,
                 min_threshold=0.4,
                 max_threshold=0.8,
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
            min_threshold (float): Минимальное пороговое значение
            max_threshold (float): Максимальное пороговое значение
            vulnerabilities_db_path (str): Путь к базе данных уязвимостей
        """
        self.text_column = text_column
        self.threshold = threshold
        self.base_threshold = threshold  # Сохраняем исходный порог для адаптивного подхода
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
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
        
        # Инициализация ML моделей
        self.ml_model = None
        self.vectorizer = None
        self.advanced_ml_model = None  # Специализированная модель для продвинутых инъекций
        self.advanced_vectorizer = None
        
        # Счетчик обнаруженных аномалий для каждого IP-адреса
        self.ip_anomaly_counter = defaultdict(int)
        
        # Счетчик последовательных запросов для каждого IP
        self.ip_sequence_counter = defaultdict(int)
        
        # Время последнего запроса для каждого IP
        self.last_request_time = {}
        
        # Последние запросы для каждого IP
        self.ip_last_queries = defaultdict(list)
        self.max_ip_queries = 10  # Хранить последние N запросов
        
        # Загрузка базы данных уязвимостей, если путь указан
        self.vulnerabilities_db = self._load_vulnerabilities_db(vulnerabilities_db_path)
        self.last_db_update = time.time()
        self.db_update_interval = 86400  # 24 часа
        
        # Инициализация паттернов
        self._init_patterns()
        
        # Инициализация техник обфускации
        self._init_obfuscation_techniques()
        
        # Списки известных кодовых слов для нормализации
        self._init_known_keywords()

    def _init_patterns(self):
        """Инициализация паттернов для обнаружения SQL-инъекций"""
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
            
            # Новые критические паттерны
            r"OR true",                        # Логическое условие
            r"OR 1",                           # Сокращенное условие
            r"';\s*--",                        # Конец запроса с комментарием
            r"\)\s*when\s*\d+\s*then",         # CASE WHEN инъекция
            r"admin'\s*--",                    # Admin комментарий
            r"'\s*or\s*'x'='x",                # OR с произвольными символами
            r"'\s*or\s*0=0\s*--",              # Еще одно OR условие
            r"'\s*or\s*0=0\s*#",               # OR условие с # комментарием
            r"'\s*GROUP BY\s*[a-z0-9]+\s*HAVING\s*\d+=\d+", # GROUP BY HAVING инъекция
            r"ORDER BY\s+\d+",                 # Blind SQL инъекция с ORDER BY
            r"HAVING\s+\d+=\d+",               # HAVING условие
        ]
        
        # Шаблоны для обнаружения токенов SQL-запросов
        self.sql_tokens = [
            r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b", r"\bINSERT\b", r"\bUPDATE\b", 
            r"\bDELETE\b", r"\bDROP\b", r"\bCREATE\b", r"\bALTER\b", r"\bJOIN\b",
            r"\bGROUP BY\b", r"\bORDER BY\b", r"\bHAVING\b", r"\bIN\b", r"\bEXISTS\b",
            r"\bLIKE\b", r"\bBETWEEN\b", r"\bAND\b", r"\bOR\b", r"\bNOT\b",
            r"\bNULL\b", r"\bINNER\b", r"\bOUTER\b", r"\bLEFT\b", r"\bRIGHT\b",
            r"\bFULL\b", r"\bTRUE\b", r"\bFALSE\b", r"\bDESC\b", r"\bASC\b",
            r"\bDISTINCT\b", r"\bCOUNT\b", r"\bMAX\b", r"\bMIN\b", r"\bAVG\b",
            r"\bSUM\b", r"\bGROUP_CONCAT\b", r"\bCONCAT\b", r"\bSUBSTRING\b",
            r"\bLIMIT\b", r"\bOFFSET\b", r"\bUNION\b", r"\bALL\b", r"\bINTO\b"
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
            
            # Новые вторичные паттерны
            r"ISNULL\s*\(",                    # ISNULL функция
            r"COALESCE\s*\(",                  # COALESCE функция
            r"OR\s+\d+",                       # Простое OR условие
            r"SELECT\s+\*",                    # SELECT всех колонок
            r"TABLE_NAME",                     # Имя таблицы
            r"COLUMN_NAME",                    # Имя колонки
            r"DATABASE\s*\(\)",                # Функция DATABASE
            r"USER\s*\(\)",                    # Функция USER
            r"VERSION\s*\(\)",                 # Функция VERSION
            r"\|\|",                           # Оператор конкатенации
            r"AND\s+\('.*?'\s*=\s*'.*?'\)",    # Сложное условие AND
            r"XOR",                            # Оператор XOR
            r"RLIKE",                          # Оператор RLIKE (RegExp)
            r"REGEXP",                         # Оператор REGEXP
            r"&&",                             # Логический оператор AND
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

    def _init_obfuscation_techniques(self):
        """Инициализация техник декодирования обфускации"""
        # Техники обфускации для декодирования
        self.obfuscation_techniques = [
            self._decode_url,
            self._decode_hex,
            self._decode_unicode,
            self._decode_base64,
            self._decode_comment_blocks,
            self._normalize_whitespace,
            self._decode_case_variations,
            self._enhanced_decode_concatenation,  # Улучшенная версия
            self._decode_char_encoding,
            self._decode_multiple_encodings,      # Новый метод
            self._reconstruct_fragmented_patterns # Новый метод
        ]

    def _init_known_keywords(self):
        """Инициализация списков известных ключевых слов SQL для нормализации"""
        # Список SQL ключевых слов
        self.sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE', 
            'DROP', 'CREATE', 'ALTER', 'UNION', 'JOIN', 'HAVING', 
            'GROUP', 'ORDER', 'BY', 'LIMIT', 'PROCEDURE', 'EXEC', 
            'EXECUTE', 'WAITFOR', 'DELAY', 'BENCHMARK', 'INTO',
            'VALUES', 'TABLE', 'DATABASE', 'COLUMN', 'INDEX',
            'CONSTRAINT', 'VIEW', 'FUNCTION', 'TRIGGER', 'SCHEMA',
            'GRANT', 'REVOKE', 'COMMIT', 'ROLLBACK', 'SAVEPOINT',
            'TRANSACTION', 'SET', 'DECLARE', 'CASE', 'WHEN', 'THEN',
            'ELSE', 'END', 'AND', 'OR', 'NOT', 'EXISTS', 'BETWEEN',
            'LIKE', 'IN', 'IS', 'NULL', 'TRUE', 'FALSE', 'PRIMARY',
            'FOREIGN', 'KEY', 'REFERENCES', 'INNER', 'OUTER', 'LEFT',
            'RIGHT', 'FULL', 'CROSS', 'NATURAL', 'ON', 'USING',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'ALL',
            'UNION', 'EXCEPT', 'INTERSECT', 'ANY', 'SOME', 'FOR',
            'OFFSET', 'FETCH', 'FIRST', 'NEXT', 'ONLY', 'WITH'
        ]
        
        # Операторы для нормализации
        self.sql_operators = [
            '=', '!=', '<>', '<', '>', '<=', '>=', '+', '-', '*', '/',
            '%', '^', '&', '|', '~', '!', '||', '&&', '<<', '>>'
        ]
        
        # Токены для обнаружения обфускации
        self.sql_tokens_map = {
            'OR': ['||', '0R', '0|2', 'OR', 'oR', 'Or', 'or', '||R'],
            'AND': ['&&', 'AnD', '&ND', 'AND', 'aNd', 'AnD', 'anD', 'and'],
            'SELECT': ['SEL/**/ECT', 'SE%4cECT', 'S%45LECT', 'SELECT', 'select', 'SeLeCt'],
            'UNION': ['UN/**/ION', 'UN%49ON', 'UN%69ON', 'UNION', 'union', 'UnIoN'],
            'FROM': ['FR/**/OM', 'F%52OM', 'F%72OM', 'FROM', 'from', 'FrOm'],
            '=': ['<>', '!=', 'is', 'like', '=']
        }

    def _enhanced_decode_concatenation(self, text: str) -> str:
        """
        Улучшенная версия декодирования строковых конкатенаций.
        Обнаруживает и обрабатывает различные паттерны конкатенации.
        """
        # Первоначальное декодирование стандартных конкатенаций
        decoded = text
        
        # Паттерны конкатенации строк
        concat_patterns = [
            # 'a'+'b' -> 'ab'
            (r"'([^']*?)'\s*\+\s*'([^']*?)'", r"'\1\2'"),
            # "a"+"b" -> "ab"
            (r'"([^"]*?)"\s*\+\s*"([^"]*?)"', r'"\1\2"'),
            # CONCAT('a','b') -> 'ab'
            (r"CONCAT\s*\(\s*'([^']*?)'\s*,\s*'([^']*?)'\s*\)", r"'\1\2'"),
            # CONCAT("a","b") -> "ab"
            (r'CONCAT\s*\(\s*"([^"]*?)"\s*,\s*"([^"]*?)"\s*\)', r'"\1\2"'),
            # 'a' || 'b' -> 'ab' (PostgreSQL конкатенация)
            (r"'([^']*?)'\s*\|\|\s*'([^']*?)'", r"'\1\2'"),
            # "a" || "b" -> "ab"
            (r'"([^"]*?)"\s*\|\|\s*"([^"]*?)"', r'"\1\2"'),
            # 'a'.concat('b') -> 'ab' (JavaScript-подобная конкатенация)
            (r"'([^']*?)'\s*\.\s*concat\s*\(\s*'([^']*?)'\s*\)", r"'\1\2'"),
            # Сложные вложенные конкатенации (например, 'a'+'b'+'c' -> 'abc')
            (r"'([^']*?)'\s*\+\s*'([^']*?)'\s*\+\s*'([^']*?)'", r"'\1\2\3'"),
        ]
        
        # Итеративное применение всех паттернов, пока текст не перестанет меняться
        prev_decoded = ""
        iterations = 0
        max_iterations = 10  # Ограничение для предотвращения зацикливания
        
        while prev_decoded != decoded and iterations < max_iterations:
            prev_decoded = decoded
            
            for pattern, replacement in concat_patterns:
                decoded = re.sub(pattern, replacement, decoded, flags=re.IGNORECASE)
            
            iterations += 1
        
        # Специальная обработка для JavaScript-подобных конкатенаций
        # Такие как: 'foo' + 'bar' или "foo" + "bar"
        js_concat_pattern = r"(?:'[^']*'|\"[^\"]*\")\s*\+\s*(?:'[^']*'|\"[^\"]*\")"
        js_concat_matches = re.findall(js_concat_pattern, decoded)
        
        for match in js_concat_matches:
            try:
                # Извлекаем части внутри кавычек
                parts = re.findall(r"['\"]([^'\"]*)['\"]", match)
                if parts:
                    # Объединяем части
                    combined = ''.join(parts)
                    # Заменяем всю конкатенацию на объединенную строку
                    delimiter = match[0]  # Используем первый символ (кавычка) как разделитель
                    replacement = f"{delimiter}{combined}{delimiter}"
                    decoded = decoded.replace(match, replacement)
            except Exception:
                continue
        
        # Специальная обработка для ' + 'OR' + '1' + '=' + '1
        fragments_pattern = r"'\s*\+\s*'([^']+)'\s*\+\s*'([^']+)'\s*\+\s*'([^']+)'\s*(?:\+\s*'([^']+)')?"
        fragments_matches = re.findall(fragments_pattern, decoded)
        
        for match in fragments_matches:
            try:
                # Объединяем все части
                combined = ''.join(part for part in match if part)
                # Создаем полный шаблон для поиска этой конкретной конкатенации
                parts_pattern = r"'\s*\+\s*'" + r"'\s*\+\s*'".join(re.escape(part) for part in match if part)
                if parts_pattern.endswith(r"\s*\+\s*'"):
                    parts_pattern = parts_pattern[:-len(r"\s*\+\s*'")]
                pattern_to_find = r"'" + parts_pattern
                replacement = f"'{combined}'"
                decoded = re.sub(pattern_to_find, replacement, decoded)
            except Exception:
                continue
        
        return decoded
    
    def _decode_multiple_encodings(self, text: str) -> str:
        """
        Обнаружение и декодирование множественных кодировок.
        Например, URL-кодирование внутри шестнадцатеричного кодирования и т.д.
        """
        # Применяем все возможные комбинации декодирования
        decoded = text
        
        # Пробуем различные комбинации декодирования
        for technique1 in [self._decode_url, self._decode_hex, self._decode_unicode]:
            temp1 = technique1(decoded)
            if temp1 != decoded:
                # Если первое декодирование успешно, пробуем второе
                for technique2 in [self._decode_url, self._decode_hex, self._decode_unicode]:
                    if technique1 != technique2:
                        temp2 = technique2(temp1)
                        if temp2 != temp1:
                            # Нашли множественное кодирование
                            decoded = temp2
        
        return decoded
    
    def _reconstruct_fragmented_patterns(self, text: str) -> str:
        """
        Восстановление фрагментированных паттернов SQL-инъекций.
        Например, 'O' + 'R' + ' ' + '1' + '=' + '1'
        """
        # Обнаружение фрагментированных паттернов для известных ключевых слов
        for keyword, variations in self.sql_tokens_map.items():
            # Создаем регулярное выражение для поиска фрагментированных вариантов
            # Например, 'O' + 'R' или "O" + "R" для ключевого слова OR
            for variation in variations:
                # Проверяем, разделен ли паттерн на отдельные символы
                if len(variation) > 1:
                    # Создаем паттерн для поиска фрагментированных строк
                    # Например, 'O' + 'R' для OR
                    fragments = []
                    for char in variation:
                        # Ищем как одиночные, так и двойные кавычки
                        fragments.append(f"(?:'{re.escape(char)}'|\"{re.escape(char)}\")")
                    
                    # Соединяем с оператором конкатенации
                    pattern = r"\s*\+\s*".join(fragments)
                    
                    # Ищем и заменяем в тексте
                    matches = re.findall(pattern, text)
                    for match in matches:
                        full_match = match
                        replacement = f"'{variation}'"
                        text = text.replace(full_match, replacement)
        
        return text
    
    def _tokenize_sql_query(self, query: str) -> List[Dict[str, str]]:
        """
        Токенизация SQL-запроса для анализа.
        
        Args:
            query (str): SQL-запрос
            
        Returns:
            List[Dict[str, str]]: Список токенов
        """
        tokens = []
        current_pos = 0
        query_length = len(query)
        
        while current_pos < query_length:
            token_found = False
            
            # Пропуск пробелов
            match = re.match(r'\s+', query[current_pos:])
            if match:
                current_pos += match.end()
                continue
            
            # Проверка на строки в кавычках
            match = re.match(r"'([^']*)'|\"([^\"]*)\"|`([^`]*)`", query[current_pos:])
            if match:
                value = match.group(0)
                tokens.append({'type': 'string', 'value': value})
                current_pos += match.end()
                token_found = True
                continue
            
            # Проверка на комментарии
            match = re.match(r'--[^\n]*|#[^\n]*|/\*.*?\*/', query[current_pos:], re.DOTALL)
            if match:
                value = match.group(0)
                tokens.append({'type': 'comment', 'value': value})
                current_pos += match.end()
                token_found = True
                continue
            
            # Проверка на ключевые слова SQL
            for keyword in self.sql_keywords:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                match = re.match(pattern, query[current_pos:], re.IGNORECASE)
                if match:
                    value = match.group(0)
                    tokens.append({'type': 'keyword', 'value': value})
                    current_pos += match.end()
                    token_found = True
                    break
            
            if token_found:
                continue
            
            # Проверка на операторы
            for operator in sorted(self.sql_operators, key=len, reverse=True):
                pattern = re.escape(operator)
                match = re.match(pattern, query[current_pos:])
                if match:
                    value = match.group(0)
                    tokens.append({'type': 'operator', 'value': value})
                    current_pos += match.end()
                    token_found = True
                    break
            
            if token_found:
                continue
            
            # Проверка на числа
            match = re.match(r'\b\d+(?:\.\d+)?\b', query[current_pos:])
            if match:
                value = match.group(0)
                tokens.append({'type': 'number', 'value': value})
                current_pos += match.end()
                token_found = True
                continue
            
            # Проверка на идентификаторы
            match = re.match(r'\b[a-zA-Z_]\w*\b', query[current_pos:])
            if match:
                value = match.group(0)
                tokens.append({'type': 'identifier', 'value': value})
                current_pos += match.end()
                token_found = True
                continue
            
            # Другие символы
            tokens.append({'type': 'other', 'value': query[current_pos]})
            current_pos += 1
        
        return tokens

    def _analyze_tokens_for_suspicious_patterns(self, tokens: List[Dict[str, str]]) -> List[str]:
        """
        Анализ токенов для обнаружения подозрительных паттернов.
        
        Args:
            tokens (List[Dict[str, str]]): Список токенов
            
        Returns:
            List[str]: Список обнаруженных подозрительных паттернов
        """
        suspicious_patterns = []
        
        # Анализ последовательностей токенов
        for i in range(len(tokens) - 2):
            # Проверка на OR 1=1, OR true, OR 'a'='a'
            if (tokens[i]['type'] == 'keyword' and tokens[i]['value'].upper() == 'OR' and
                ((tokens[i+1]['type'] == 'number' and tokens[i+2]['type'] == 'operator' and tokens[i+2]['value'] == '=') or
                 (tokens[i+1]['type'] == 'keyword' and tokens[i+1]['value'].upper() == 'TRUE') or
                 (tokens[i+1]['type'] == 'string' and i+3 < len(tokens) and tokens[i+2]['type'] == 'operator' and 
                  tokens[i+2]['value'] == '=' and tokens[i+3]['type'] == 'string'))):
                suspicious_patterns.append('boolean_based')
            
            # Проверка на UNION SELECT
            if (tokens[i]['type'] == 'keyword' and tokens[i]['value'].upper() == 'UNION' and
                i+2 < len(tokens) and tokens[i+1]['type'] == 'keyword' and tokens[i+1]['value'].upper() == 'SELECT'):
                suspicious_patterns.append('union_based')
            
            # Проверка на SLEEP(), WAITFOR DELAY, BENCHMARK()
            if ((tokens[i]['type'] == 'keyword' and tokens[i]['value'].upper() in ('SLEEP', 'WAITFOR', 'BENCHMARK')) or
                (tokens[i]['type'] == 'identifier' and tokens[i]['value'].upper() in ('SLEEP', 'WAITFOR', 'BENCHMARK'))):
                suspicious_patterns.append('time_based')
            
            # Проверка на DROP, TRUNCATE
            if (tokens[i]['type'] == 'keyword' and tokens[i]['value'].upper() in ('DROP', 'TRUNCATE')):
                suspicious_patterns.append('destructive')
            
            # Проверка на xp_cmdshell
            if (tokens[i]['type'] == 'identifier' and 'cmdshell' in tokens[i]['value'].lower()):
                suspicious_patterns.append('command_exec')
        
        return list(set(suspicious_patterns))

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
        
        # Проверка на фрагментированные строки (' + 'OR' + '1' + '=' + '1)
        if re.search(r"'\s*\+\s*'[^']+'\s*\+\s*'", original):
            techniques.append("Фрагментированные строки")
        
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
            
            # Обучение основной ML-модели
            self._train_basic_ml_model(queries)
            
            # Обучение продвинутой ML-модели для сложных инъекций
            self._train_advanced_ml_model(queries)
            
            if self.verbose:
                print(f"Модель машинного обучения обучена на {len(queries)} запросах.")
        
        return self

    def _train_basic_ml_model(self, queries):
        """
        Обучение основной ML-модели для обнаружения аномалий.
        
        Args:
            queries (List[str]): Список текстов запросов
        """
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

    def _train_advanced_ml_model(self, queries):
        """
        Обучение продвинутой ML-модели для обнаружения сложных инъекций.
        Создает синтетические примеры инъекций на основе обычных запросов.
        
        Args:
            queries (List[str]): Список обычных запросов
        """
        # Создаем синтетические примеры инъекций на основе обычных запросов
        synthetic_injections = self._generate_synthetic_injections(queries)
        
        if not synthetic_injections:
            # Если не удалось создать синтетические примеры, не обучаем продвинутую модель
            return
        
        # Создаем обучающий набор с метками (0 - нормальные, 1 - инъекции)
        X_train = queries + synthetic_injections
        y_train = [0] * len(queries) + [1] * len(synthetic_injections)
        
        # Создаем TF-IDF векторизатор с более продвинутыми параметрами
        self.advanced_vectorizer = TfidfVectorizer(
            max_features=200,     # Больше признаков
            ngram_range=(1, 4),   # Больше n-грамм для захвата продвинутых паттернов
            analyzer='char',      # Анализируем на уровне символов
            lowercase=True,
            min_df=1,             # Включаем редкие n-граммы
            max_df=1.0
        )
        
        # Преобразуем тексты в числовые признаки
        X_vectors = self.advanced_vectorizer.fit_transform(X_train)
        
        # Обучаем RandomForestClassifier для задачи классификации
        self.advanced_ml_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            random_state=42,
            class_weight='balanced'  # Учитываем дисбаланс классов
        )
        
        self.advanced_ml_model.fit(X_vectors, y_train)

    def _generate_synthetic_injections(self, normal_queries):
        """
        Генерация синтетических примеров инъекций на основе обычных запросов.
        
        Args:
            normal_queries (List[str]): Список обычных запросов
        
        Returns:
            List[str]: Список синтетических инъекций
        """
        synthetic_injections = []
        
        # Шаблоны для продвинутых инъекций
        advanced_injection_templates = [
            "' OR 1=1 --",
            "' UNION SELECT 1,2,3 --",
            "'; DROP TABLE users --",
            "' OR '1'='1",
            "' AND (SELECT COUNT(*) FROM users) > 10 --",
            "' WAITFOR DELAY '0:0:5' --",
            "' OR SLEEP(5) --",
            "' OR EXISTS(SELECT 1 FROM users) --",
            "' OR (SELECT ascii(substring(database(),1,1))) > 90 --",
            "' AND (SELECT 1 FROM information_schema.tables LIMIT 1) --",
            "' UNION ALL SELECT 1,2,@@version --",
            "' ORDER BY 10 --",
            "' GROUP BY 1 HAVING 1=1 --",
            "' INSERT INTO users VALUES ('hacker','hacked') --"
        ]
        
        # Обфусцированные шаблоны
        obfuscated_templates = [
            "' OR/**/ 1=1 --",
            "' UNION/**/SELECT 1,2,3 --",
            "' OR '1'/**/='1",
            "' + 'OR' + '1' + '=' + '1",
            "' OR 0x31=0x31 --",
            "' OR CHAR(49)=CHAR(49) --",
            "' || 1=1 --",
            "' UnIoN SeLeCt 1,2,3 --",
            "' OR (1)/**/=/**/(1) --"
        ]
        
        # Для каждого шаблона создаем инъекции
        for template in advanced_injection_templates + obfuscated_templates:
            synthetic_injections.append(template)
        
        # Для каждого нормального запроса пытаемся внедрить инъекцию
        for query in normal_queries:
            # Извлекаем части запроса с помощью регулярных выражений
            select_match = re.search(r'\bSELECT\b.*?\bFROM\b', query, re.IGNORECASE)
            where_match = re.search(r'\bWHERE\b.*?(?:\bORDER BY\b|\bGROUP BY\b|\bHAVING\b|$)', query, re.IGNORECASE)
            
            if select_match and where_match:
                # Создаем инъекцию в WHERE условии
                where_part = where_match.group(0)
                injected_where = re.sub(r'(\b\w+\s*=\s*)([\'"][\w@.]+[\'"]|\d+)', r"\1' OR 1=1 --", where_part, count=1)
                injected_query = query.replace(where_part, injected_where)
                synthetic_injections.append(injected_query)
                
                # Создаем UNION-инъекцию
                select_part = select_match.group(0)
                column_count = len(re.findall(r',', select_part)) + 1
                union_injection = f" UNION SELECT {','.join(['1'] * column_count)} --"
                injected_query = query + union_injection
                synthetic_injections.append(injected_query)
        
        return synthetic_injections

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
        advanced_ml_scores = None
        
        if self.use_ml:
            if self.ml_model is not None and self.vectorizer is not None:
                # Извлечение текстов запросов
                queries = data[self.text_column].fillna('').astype(str).tolist()
                
                # Предсказания основной модели
                try:
                    # Преобразование текста в числовые признаки
                    X = self.vectorizer.transform(queries)
                    
                    # Предсказания аномалий (-1 для аномалий, 1 для нормальных)
                    # Преобразуем в оценки от 0 до 1
                    raw_scores = self.ml_model.decision_function(X)
                    ml_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
                except Exception as e:
                    if self.verbose:
                        print(f"Ошибка при использовании основной ML-модели: {e}")
                    ml_scores = np.zeros(len(queries))
            
            # Используем продвинутую ML-модель, если она обучена
            if self.advanced_ml_model is not None and self.advanced_vectorizer is not None:
                try:
                    # Извлечение текстов запросов
                    queries = data[self.text_column].fillna('').astype(str).tolist()
                    
                    # Преобразование текста в числовые признаки
                    X_advanced = self.advanced_vectorizer.transform(queries)
                    
                    # Предсказания вероятностей классов (индекс 1 соответствует классу инъекций)
                    advanced_ml_scores = self.advanced_ml_model.predict_proba(X_advanced)[:, 1]
                except Exception as e:
                    if self.verbose:
                        print(f"Ошибка при использовании продвинутой ML-модели: {e}")
                    advanced_ml_scores = np.zeros(len(queries))
        
        current_time = time.time()
        
        # Анализ каждого запроса
        for idx, row in data.iterrows():
            text = row[self.text_column]
            if not isinstance(text, str):
                continue
            
            # Получение IP-адреса, если доступно
            ip_address = ip_addresses.iloc[idx] if ip_addresses is not None else None
            
            # Улучшенный контекстный анализ
            context_modifier = self._analyze_context(text, ip_address, current_time)
            
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
            advanced_score = advanced_ml_scores[idx] if advanced_ml_scores is not None else 0
            
            # Объединение оценок (улучшенный алгоритм)
            combined_score = self._combine_scores(pattern_score, ml_score, advanced_score)
            
            # Применение контекстного модификатора
            final_score = min(1.0, combined_score + context_modifier)
            
            # Вычисление текущего порога
            current_threshold = self._calculate_threshold(text, deobfuscated_text, pattern_score)
            
            # Проверка на аномалию и определение типа атаки
            is_anomaly = final_score >= current_threshold
            
            # Определение источника обнаружения
            detected_by = self._determine_detection_source(pattern_score, ml_score, advanced_score, 
                                                          context_modifier, current_threshold)
            
            # Определение типа атаки
            attack_type = None
            if is_anomaly:
                attack_type = self._determine_attack_type(text, deobfuscated_text, matched_patterns)
            
            # Определение уровня угрозы
            threat_level = self._determine_threat_level(final_score, is_anomaly)
            
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
                self._update_request_history(text, deobfuscated_text, ip_address, current_time, is_anomaly, final_score)
        
        return result

    def _combine_scores(self, pattern_score, ml_score, advanced_score):
        """
        Улучшенный алгоритм комбинирования оценок.
        
        Args:
            pattern_score (float): Оценка на основе паттернов
            ml_score (float): Оценка основной ML-модели
            advanced_score (float): Оценка продвинутой ML-модели
            
        Returns:
            float: Комбинированная оценка
        """
        # Если паттерн-оценка высокая, отдаем предпочтение ей
        if pattern_score >= 0.9:
            return pattern_score
        
        # Если продвинутая ML-оценка высокая, учитываем её с большим весом
        if advanced_score >= 0.8:
            return max(pattern_score, 0.7 * advanced_score + 0.3 * ml_score)
        
        # Базовая комбинация всех трех оценок
        weights_sum = 3.0
        weighted_score = (0.6 * pattern_score + 0.2 * ml_score + 0.2 * advanced_score) / (weights_sum / 3)
        
        return weighted_score

    def _calculate_threshold(self, text, deobfuscated_text, pattern_score):
        """
        Расчет индивидуального порога для конкретного запроса.
        
        Args:
            text (str): Исходный текст запроса
            deobfuscated_text (str): Декодированный текст
            pattern_score (float): Оценка на основе паттернов
            
        Returns:
            float: Вычисленный порог
        """
        # Базовый порог
        threshold = self.threshold
        
        # Если адаптивные пороги включены
        if self.adaptive_threshold and len(self.score_history) > 10:
            threshold = self._calculate_adaptive_threshold()
        
        # Нижняя граница порога для критических признаков
        if pattern_score >= 0.7:
            return min(threshold, 0.6)  # Снижаем порог для потенциально опасных запросов
        
        # Повышаем порог для запросов без подозрительных паттернов
        if pattern_score < 0.1:
            return max(threshold, 0.65)  # Повышаем порог для безопасных запросов
        
        # Для продвинутых и обфусцированных инъекций оставляем порог низким
        if text != deobfuscated_text:
            return min(threshold, 0.55)  # Снижаем порог для обфусцированных запросов
        
        # Дополнительные проверки для специфических случаев
        if any(keyword in text.upper() for keyword in ['UNION', 'DROP', 'EXEC', 'WAITFOR']):
            return min(threshold, 0.5)  # Ещё сильнее снижаем порог для явно опасных запросов
        
        return threshold

    def _calculate_adaptive_threshold(self) -> float:
        """
        Усовершенствованный расчет адаптивного порога на основе истории оценок.
        
        Returns:
            float: Адаптивный порог
        """
        if not self.score_history:
            return self.base_threshold
        
        # Отсортированные оценки аномалий
        sorted_scores = sorted(self.score_history)
        n = len(sorted_scores)
        
        # Находим естественные кластеры в данных
        clusters = self._find_score_clusters(sorted_scores)
        
        # Если найдено хотя бы 2 кластера, используем разрыв между ними как порог
        if len(clusters) >= 2:
            # Выбираем разрыв между самым высоким кластером нормальных запросов 
            # и самым низким кластером аномальных запросов
            normal_upper = max(clusters[0])
            anomaly_lower = min(clusters[-1])
            
            # Используем середину разрыва как порог
            gap_threshold = (normal_upper + anomaly_lower) / 2
            
            # Ограничиваем порог разумными пределами
            return max(self.min_threshold, min(self.max_threshold, gap_threshold))
        
        # Если кластеризация не дала результатов, используем статистический подход
        # Вычисляем среднее и стандартное отклонение
        mean = sum(self.score_history) / n
        std_dev = (sum((x - mean) ** 2 for x in self.score_history) / n) ** 0.5
        
        # Порог как среднее плюс 2 стандартных отклонения
        statistical_threshold = mean + 2 * std_dev
        
        # Ограничиваем порог разумными пределами
        return max(self.min_threshold, min(self.max_threshold, statistical_threshold))

    def _find_score_clusters(self, sorted_scores):
        """
        Находит естественные кластеры в отсортированном списке оценок.
        
        Args:
            sorted_scores (List[float]): Отсортированный список оценок
            
        Returns:
            List[List[float]]: Список кластеров оценок
        """
        if not sorted_scores:
            return []
        
        clusters = []
        current_cluster = [sorted_scores[0]]
        
        # Порог для разделения кластеров
        gap_threshold = 0.15
        
        for i in range(1, len(sorted_scores)):
            # Если разрыв между текущим и предыдущим значением больше порога,
            # начинаем новый кластер
            if sorted_scores[i] - sorted_scores[i-1] > gap_threshold:
                clusters.append(current_cluster)
                current_cluster = [sorted_scores[i]]
            else:
                current_cluster.append(sorted_scores[i])
        
        # Добавляем последний кластер
        if current_cluster:
            clusters.append(current_cluster)
        
        return clusters

    def _analyze_context(self, text, ip_address, current_time):
        """
        Улучшенный контекстный анализ запроса.
        
        Args:
            text (str): Текст запроса
            ip_address (str): IP-адрес
            current_time (float): Текущее время
            
        Returns:
            float: Модификатор оценки на основе контекста
        """
        if not self.context_aware or not ip_address:
            return 0
        
        context_modifier = 0
        
        # Проверка частоты запросов с одного IP
        if ip_address in self.last_request_time:
            time_diff = current_time - self.last_request_time[ip_address]
            
            # Если запросы от IP приходят слишком часто (менее 0.5 секунды между запросами)
            if time_diff < 0.5:
                self.ip_anomaly_counter[ip_address] += 1
                context_modifier += 0.05  # Увеличиваем модификатор за каждый быстрый запрос
            elif time_diff < 1.0:
                self.ip_anomaly_counter[ip_address] += 0.5
                context_modifier += 0.02  # Меньший модификатор для менее быстрых запросов
        
        # Обновляем время последнего запроса
        self.last_request_time[ip_address] = current_time
        
        # Штраф за подозрительную активность с IP
        if self.ip_anomaly_counter[ip_address] > 5:
            context_modifier += min(0.3, 0.05 * self.ip_anomaly_counter[ip_address])
        
        # Анализ последовательности запросов от одного IP
        if ip_address in self.ip_last_queries and len(self.ip_last_queries[ip_address]) > 0:
            # Собираем последние запросы от этого IP
            last_queries = self.ip_last_queries[ip_address]
            
            # Проверяем признаки последовательного сканирования
            if self._detect_sequential_scanning(last_queries, text):
                context_modifier += 0.15
                if self.verbose:
                    print(f"Обнаружено последовательное сканирование от {ip_address}")
            
            # Проверяем на постепенную эскалацию сложности запросов
            if self._detect_complexity_escalation(last_queries, text):
                context_modifier += 0.2
                if self.verbose:
                    print(f"Обнаружена эскалация сложности запросов от {ip_address}")
            
            # Проверяем на паттерны разведки
            if self._detect_reconnaissance_pattern(last_queries, text):
                context_modifier += 0.25
                if self.verbose:
                    print(f"Обнаружен паттерн разведки от {ip_address}")
        
        # Ограничиваем максимальный контекстный модификатор
        return min(0.6, context_modifier)

    def _update_request_history(self, text, deobfuscated_text, ip_address, timestamp, is_anomaly, score):
        """
        Обновление истории запросов для контекстного анализа.
        
        Args:
            text (str): Исходный текст запроса
            deobfuscated_text (str): Декодированный текст
            ip_address (str): IP-адрес
            timestamp (float): Временная метка
            is_anomaly (bool): Является ли запрос аномальным
            score (float): Оценка аномальности
        """
        # Сохраняем в общую историю запросов
        self.request_history.append({
            'text': text,
            'deobfuscated': deobfuscated_text,
            'ip': ip_address,
            'timestamp': timestamp,
            'is_anomaly': is_anomaly,
            'score': score
        })
        
        # Ограничиваем размер общей истории
        if len(self.request_history) > self.history_size:
            self.request_history.pop(0)
        
        # Сохраняем в историю запросов для каждого IP
        if ip_address:
            # Добавляем новый запрос в историю для данного IP
            self.ip_last_queries[ip_address].append({
                'text': text,
                'deobfuscated': deobfuscated_text,
                'timestamp': timestamp,
                'is_anomaly': is_anomaly,
                'score': score
            })
            
            # Ограничиваем размер истории для каждого IP
            if len(self.ip_last_queries[ip_address]) > self.max_ip_queries:
                self.ip_last_queries[ip_address].pop(0)

    def _detect_sequential_scanning(self, last_queries, current_query):
        """
        Обнаружение последовательного сканирования.
        
        Args:
            last_queries (List[Dict]): Последние запросы от IP
            current_query (str): Текущий запрос
            
        Returns:
            bool: Обнаружено ли последовательное сканирование
        """
        # Проверка на последовательное изменение числовых параметров в WHERE условиях
        where_values = []
        
        # Извлекаем числовые значения из WHERE условий для последних запросов
        for query_info in last_queries:
            query = query_info['text']
            where_match = re.search(r'\bWHERE\b.*?(\b\d+\b)', query, re.IGNORECASE)
            if where_match:
                where_values.append(int(where_match.group(1)))
        
        # Извлекаем числовое значение из текущего запроса
        current_where_match = re.search(r'\bWHERE\b.*?(\b\d+\b)', current_query, re.IGNORECASE)
        if current_where_match:
            current_value = int(current_where_match.group(1))
            where_values.append(current_value)
        
        # Проверяем на последовательное увеличение или уменьшение
        if len(where_values) >= 3:
            # Проверка на арифметическую прогрессию
            diffs = [where_values[i+1] - where_values[i] for i in range(len(where_values)-1)]
            if len(set(diffs)) == 1:  # Все разности одинаковые
                return True
            
            # Проверка на последовательное увеличение или уменьшение с небольшими вариациями
            increasing = all(where_values[i] <= where_values[i+1] for i in range(len(where_values)-1))
            decreasing = all(where_values[i] >= where_values[i+1] for i in range(len(where_values)-1))
            if increasing or decreasing:
                return True
        
        return False

    def _detect_complexity_escalation(self, last_queries, current_query):
        """
        Обнаружение постепенной эскалации сложности запросов.
        
        Args:
            last_queries (List[Dict]): Последние запросы от IP
            current_query (str): Текущий запрос
            
        Returns:
            bool: Обнаружена ли эскалация сложности
        """
        # Функция для оценки сложности запроса
        def query_complexity(q):
            complexity = 0
            # Основные SQL-операции
            if re.search(r'\bSELECT\b', q, re.IGNORECASE): complexity += 1
            if re.search(r'\bFROM\b', q, re.IGNORECASE): complexity += 1
            if re.search(r'\bWHERE\b', q, re.IGNORECASE): complexity += 2
            if re.search(r'\bJOIN\b', q, re.IGNORECASE): complexity += 3
            if re.search(r'\bGROUP BY\b', q, re.IGNORECASE): complexity += 3
            if re.search(r'\bORDER BY\b', q, re.IGNORECASE): complexity += 2
            if re.search(r'\bHAVING\b', q, re.IGNORECASE): complexity += 3
            # Подзапросы и функции
            if re.search(r'\([^()]*SELECT[^()]*\)', q, re.IGNORECASE): complexity += 5
            if re.search(r'\bCASE\b', q, re.IGNORECASE): complexity += 4
            # Количество условий
            complexity += len(re.findall(r'\bAND\b|\bOR\b', q, re.IGNORECASE)) * 2
            # Количество функций
            complexity += len(re.findall(r'\b\w+\s*\(', q)) * 2
            # Общая длина запроса
            complexity += len(q) / 50
            return complexity
        
        # Вычисляем сложность последних запросов
        complexities = [query_complexity(q['text']) for q in last_queries]
        current_complexity = query_complexity(current_query)
        
        # Проверяем тренд на увеличение сложности
        if len(complexities) >= 2:
            # Добавляем текущую сложность
            all_complexities = complexities + [current_complexity]
            
            # Проверка на общий тренд увеличения сложности
            is_increasing = True
            for i in range(len(all_complexities) - 2):
                # Допускаем небольшие снижения сложности
                if all_complexities[i+1] < all_complexities[i] * 0.8:
                    is_increasing = False
                    break
            
            # Проверяем, что последний запрос значительно сложнее первого
            significant_increase = all_complexities[-1] > all_complexities[0] * 1.5
            
            return is_increasing and significant_increase
        
        return False

    def _detect_reconnaissance_pattern(self, last_queries, current_query):
        """
        Обнаружение паттернов разведки.
        
        Args:
            last_queries (List[Dict]): Последние запросы от IP
            current_query (str): Текущий запрос
            
        Returns:
            bool: Обнаружен ли паттерн разведки
        """
        # Признаки разведки: запросы к системным таблицам, информации о схеме БД, тестовые инъекции
        reconnaissance_indicators = [
            # Запросы к системным таблицам
            r'information_schema\.',
            r'sys\.',
            r'pg_catalog\.',
            r'sqlite_master',
            # Запросы версии и конфигурации
            r'@@version',
            r'version\(\)',
            r'sqlite_version\(\)',
            r'sys\.configurations',
            # Тестовые инъекции
            r"'--",
            r'"--',
            r"' OR '1'='1"
        ]
        
        # Считаем количество индикаторов разведки в последних запросах
        reconnaissance_count = 0
        all_queries = [q['text'] for q in last_queries] + [current_query]
        
        for query in all_queries:
            for indicator in reconnaissance_indicators:
                if re.search(indicator, query, re.IGNORECASE):
                    reconnaissance_count += 1
        
        # Если обнаружено достаточное количество индикаторов, считаем это разведкой
        return reconnaissance_count >= 2

    def _determine_detection_source(self, pattern_score, ml_score, advanced_score, context_modifier, threshold):
        """
        Определение источника обнаружения аномалии.
        
        Args:
            pattern_score (float): Оценка на основе паттернов
            ml_score (float): Оценка основной ML-модели
            advanced_score (float): Оценка продвинутой ML-модели
            context_modifier (float): Контекстный модификатор
            threshold (float): Пороговое значение
            
        Returns:
            List[str]: Список источников обнаружения
        """
        detected_by = []
        if pattern_score >= threshold:
            detected_by.append("pattern")
        if ml_score >= threshold:
            detected_by.append("ml")
        if advanced_score >= threshold:
            detected_by.append("advanced_ml")
        if context_modifier > 0 and context_modifier + max(pattern_score, ml_score, advanced_score) >= threshold:
            detected_by.append("context")
        
        return detected_by

    def _determine_threat_level(self, score, is_anomaly):
        """
        Определение уровня угрозы на основе оценки.
        
        Args:
            score (float): Оценка аномальности
            is_anomaly (bool): Является ли запрос аномальным
            
        Returns:
            str: Уровень угрозы
        """
        if not is_anomaly:
            return None
        
        if score >= 0.9:
            return 'Critical'
        elif score >= 0.75:
            return 'High'
        elif score >= 0.6:
            return 'Medium'
        else:
            return 'Low'

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
            
            # Специальная проверка для обнаружения разделенных фрагментов
            tokens = self._tokenize_sql_query(text)
            suspicious_patterns = self._analyze_tokens_for_suspicious_patterns(tokens)
            
            if suspicious_patterns:
                score = max(score, 0.8)
                matched_patterns.extend(suspicious_patterns)
        
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
        if 'or 1=1' in deobfuscated_lower or "or '1'='1'" in deobfuscated_lower or 'or true' in deobfuscated_lower:
            return 'boolean_based'
        elif 'union select' in deobfuscated_lower or 'union all select' in deobfuscated_lower:
            return 'union_based'
        elif 'waitfor delay' in deobfuscated_lower or 'sleep(' in deobfuscated_lower or 'benchmark(' in deobfuscated_lower or 'pg_sleep(' in deobfuscated_lower:
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

    def _generate_explanation(self, text: str, deobfuscated_text: str, threat_level: str, 
                             attack_type: str, detected_by: Optional[str]) -> str:
        """
        Генерация улучшенного объяснения для аномалии.
        
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
                "ml": "Обнаружено базовой моделью машинного обучения",
                "advanced_ml": "Обнаружено продвинутой моделью машинного обучения",
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
            
            # Добавляем информацию о различиях между исходным и декодированным запросом
            differences = self._highlight_differences(text, deobfuscated_text)
            if differences:
                explanation_parts.append("\nОсновные различия:")
                explanation_parts.append(differences)
        
        return "\n".join(explanation_parts)

    def _highlight_differences(self, original, deobfuscated):
        """
        Выделение основных различий между исходным и декодированным запросом.
        
        Args:
            original (str): Исходный запрос
            deobfuscated (str): Декодированный запрос
            
        Returns:
            str: Информация о различиях
        """
        differences = []
        
        # Проверка на URL-кодирование
        if '%' in original and original != self._decode_url(original):
            differences.append("- URL-кодирование: символы вида %XX заменены на соответствующие символы")
        
        # Проверка на шестнадцатеричное кодирование
        if re.search(r'0x[0-9a-fA-F]+', original):
            differences.append("- Шестнадцатеричное кодирование: 0x... заменено на соответствующие символы")
        
        # Проверка на комментарии
        if re.search(r'--.*?(\n|$)|#.*?(\n|$)|/\*.*?\*/', original, flags=re.DOTALL):
            differences.append("- SQL-комментарии: удалены комментарии вида --, # или /* */")
        
        # Проверка на конкатенацию строк
        if re.search(r"'\s*\+\s*'", original) or re.search(r"CONCAT\s*\(", original, re.IGNORECASE):
            differences.append("- Конкатенация строк: строки вида 'a'+'b' объединены в 'ab'")
        
        # Проверка на кодирование символов
        if re.search(r"CHAR\s*\(\s*\d+\s*\)", original, re.IGNORECASE):
            differences.append("- Кодирование символов: функции CHAR() заменены на соответствующие символы")
        
        # Проверка на регистр ключевых слов
        for keyword in ['SELECT', 'UNION', 'FROM', 'WHERE']:
            pattern = r'\b' + r''.join(f'[{c.lower()}{c.upper()}]' for c in keyword) + r'\b'
            if re.search(pattern, original) and not re.search(r'\b' + keyword + r'\b', original):
                differences.append(f"- Смешанный регистр: нормализованы ключевые слова (например, {keyword})")
                break
        
        if not differences:
            # Если не удалось определить конкретные различия, используем общее описание
            matcher = SequenceMatcher(None, original, deobfuscated)
            similarity = matcher.ratio()
            
            if similarity < 0.5:
                return "Существенные различия между оригинальным и декодированным запросом"
            elif similarity < 0.9:
                return "Умеренные различия, в основном связанные с декодированием обфускации"
            else:
                return "Незначительные различия в форматировании"
        
        return "\n".join(differences)

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
        
        # Дополнительные рекомендации на основе типа атаки
        if attack_type == 'boolean_based' or attack_type == 'union_based':
            recommendation_parts.append("• Используйте prepared statements или хранимые процедуры для всех SQL-запросов.")
        elif attack_type == 'time_based':
            recommendation_parts.append("• Установите тайм-ауты для всех SQL-запросов и мониторьте продолжительность выполнения.")
        elif attack_type == 'error_based':
            recommendation_parts.append("• Не отображайте детальные сообщения об ошибках SQL пользователям.")
        elif attack_type == 'destructive' or attack_type == 'stacked_queries':
            recommendation_parts.append("• Удостоверьтесь, что пользователь БД не имеет прав на изменение структуры базы данных.")
        
        return "\n".join(recommendation_parts)

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
        # Нормализация регистра ключевых слов
        for keyword in self.sql_keywords:
            # Регулярное выражение с учетом различных вариантов регистра
            pattern = r'\b' + r''.join(f'[{c.lower()}{c.upper()}]' for c in keyword) + r'\b'
            
            # Замена на ключевое слово в верхнем регистре
            text = re.sub(pattern, keyword, text)
        
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
                try:
                    decoded = technique(decoded)
                except Exception as e:
                    if self.verbose:
                        print(f"Ошибка при применении техники декодирования: {str(e)}")
            
            iteration += 1
        
        return decoded

    def _check_db_update(self) -> None:
        """Проверка и обновление базы данных уязвимостей."""
        current_time = time.time()
        
        # Проверяем, нужно ли обновлять базу данных
        if (current_time - self.last_db_update) >= self.db_update_interval:
            # Здесь может быть логика для загрузки обновлений из внешнего источника
            self.last_db_update = current_time

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
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'verbose': self.verbose,
            'use_ml': self.use_ml,
            'adaptive_threshold': self.adaptive_threshold,
            'context_aware': self.context_aware,
            'vectorizer': self.vectorizer,
            'ml_model': self.ml_model,
            'advanced_vectorizer': self.advanced_vectorizer,
            'advanced_ml_model': self.advanced_ml_model,
            'critical_patterns': self.critical_patterns,
            'secondary_patterns': self.secondary_patterns,
            'attack_categories': self.attack_categories,
            'security_recommendations': self.security_recommendations,
            'default_recommendation': self.default_recommendation,
            'sql_keywords': self.sql_keywords,
            'sql_operators': self.sql_operators,
            'sql_tokens_map': self.sql_tokens_map
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
        self.min_threshold = model_data.get('min_threshold', 0.4)  # Совместимость со старыми моделями
        self.max_threshold = model_data.get('max_threshold', 0.8)  # Совместимость со старыми моделями
        self.verbose = model_data['verbose']
        self.use_ml = model_data['use_ml']
        self.adaptive_threshold = model_data['adaptive_threshold']
        self.context_aware = model_data['context_aware']
        self.vectorizer = model_data['vectorizer']
        self.ml_model = model_data['ml_model']
        self.advanced_vectorizer = model_data.get('advanced_vectorizer')  # Совместимость со старыми моделями
        self.advanced_ml_model = model_data.get('advanced_ml_model')  # Совместимость со старыми моделями
        self.critical_patterns = model_data['critical_patterns']
        self.secondary_patterns = model_data['secondary_patterns']
        self.attack_categories = model_data['attack_categories']
        self.security_recommendations = model_data['security_recommendations']
        self.default_recommendation = model_data['default_recommendation']
        self.sql_keywords = model_data.get('sql_keywords', [])  # Совместимость со старыми моделями
        self.sql_operators = model_data.get('sql_operators', [])  # Совместимость со старыми моделями
        self.sql_tokens_map = model_data.get('sql_tokens_map', {})  # Совместимость со старыми моделями
        
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
