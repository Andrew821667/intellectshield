import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import logging
import math
from collections import Counter

# Импортируем базовый класс детектора
from detectors.base import BaseAnomalyDetector

class EnhancedSQLInjectionDetector(BaseAnomalyDetector):
    """
    Улучшенный детектор SQL-инъекций с сниженным количеством ложных срабатываний
    и расширенными возможностями обнаружения.
    
    Основные улучшения:
    1. Повышенный порог обнаружения (0.6 вместо 0.5)
    2. Белый список стандартных SQL-операций
    3. Улучшенный алгоритм контекстного анализа
    4. Оптимизированная обработка коротких запросов
    5. Механизм самообучения на безопасных запросах
    """
    
    def __init__(self, 
                 text_column: str = 'query', 
                 threshold: float = 0.6,  # Повышенный порог для снижения ложных срабатываний
                 context_window: int = 10,
                 min_query_length: int = 5,  # Увеличена минимальная длина запроса
                 use_entropy: bool = True,
                 use_regex: bool = True,
                 use_heuristics: bool = True,
                 use_context: bool = True,
                 weights: Dict[str, float] = None,
                 custom_patterns: List[str] = None,
                 whitelist_patterns: List[str] = None,  # Белый список шаблонов запросов
                 auto_learning: bool = True,  # Автоматическое обучение на безопасных запросах
                 verbose: bool = False):
        super().__init__()
        
        self.text_column = text_column
        self.threshold = threshold
        self.context_window = context_window
        self.min_query_length = min_query_length
        self.use_entropy = use_entropy
        self.use_regex = use_regex
        self.use_heuristics = use_heuristics
        self.use_context = use_context
        self.verbose = verbose
        self.auto_learning = auto_learning
        self.is_fitted = False
        
        # Скорректированные веса для компонентов
        self.weights = {
            'regex': 0.55,        # Увеличен вес регулярных выражений
            'entropy': 0.15,
            'heuristics': 0.25,
            'context': 0.05       # Уменьшен вес контекстного анализа
        }
        
        if weights is not None:
            self.weights.update(weights)
        
        # Нормализуем веса
        sum_weights = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= sum_weights

        # Расширенный набор паттернов SQL-инъекций
        self.sql_patterns = [
            # Базовые SQL-инъекции
            r"(?i)'\s*OR\s*'?1'?\s*=\s*'?1'?",
            r"(?i)'\s*OR\s*'?1'?\s*=\s*'?1'?\s*--",
            r"(?i)'\s*OR\s*'?1'?\s*=\s*'?1'?\s*#",
            r"(?i)'\s*OR\s*'?1'?\s*=\s*'?1'?\s*\/\*",
            
            # UNION-атаки
            r"(?i)('\s*|\s+)UNION(\s|\+)+(ALL|SELECT)",
            r"(?i)UNION\s+SELECT\s+",
            r"(?i)UNION\s+ALL\s+SELECT\s+",
            r"(?i)'\s*UNION\s*SELECT\s+",
            
            # Манипуляции с таблицами
            r"(?i)('\s*|\s+)DROP(\s|\+)+(TABLE|DATABASE)",
            r"(?i)('\s*|\s+)ALTER(\s|\+)+(TABLE|DATABASE)",
            r"(?i)('\s*|\s+)TRUNCATE(\s|\+)+TABLE",
            
            # Опасные команды
            r"(?i)('\s*|\s+)EXEC(UTE)?(\s|\+)+",
            r"(?i)'\s*;\s*xp_cmdshell",
            r"(?i)'\s*;\s*sp_execute",
            r"(?i)EXECUTE\s+IMMEDIATE",
            
            # Комментарии в SQL для обхода фильтров
            r"(?i)'\s*(\-\-|#|\/\*).+?(\*\/)?$",
            r"(?i)'\s*;.*--",  # Часто используется при многострочных инъекциях
            
            # Условные операторы, часто используемые в инъекциях
            r"(?i)'\s+OR\s+('|\")?\s*\d+\s*\=\s*\d+\s*('|\")?",
            r"(?i)'\s+AND\s+('|\")?\s*\d+\s*\=\s*\d+\s*('|\")?",
            r"(?i)'\s*OR\s+'?[0-9a-zA-Z]+'?\s*=\s*'?[0-9a-zA-Z]+'?",
            r"(?i)'\s+OR\s+'.*'\s*=\s*'.*'",
            r"(?i)'\s+OR\s+'\w+'\s*LIKE\s*'\w+'",
            
            # Попытки обойти экранирование
            r"(?i)'\s*OR\s*'?[0-9]+'?(\s|\+)*=(\s|\+)*'?[0-9]+'?",
            r"(?i)'\s*AND\s*'?[0-9]+'?(\s|\+)*=(\s|\+)*'?[0-9]+'?",
            r"(?i)'\s*;\s*",  # Точка с запятой в строке - часто признак инъекции
            
            # Техники слепой SQL-инъекции
            r"(?i)WAITFOR\s+DELAY\s+",
            r"(?i)BENCHMARK\(\d+,\s*\w+\)",
            r"(?i)SLEEP\(\d+\)",
            r"(?i)pg_sleep\(\d+\)",
            
            # Строковые манипуляции
            r"(?i)CONCAT\(.+?\)",
            r"(?i)GROUP_CONCAT\(.+?\)",
            r"(?i)CHAR\(.+?\)",
            r"(?i)CHR\(.+?\)",
            r"(?i)ASCII\(.+?\)",
            r"(?i)CAST\(.+?\)",
            r"(?i)CONVERT\(.+?\)",
            
            # Атаки на базы данных
            r"(?i)(INFORMATION_SCHEMA|sysobjects|syscolumns)",
            r"(?i)(pg_tables|pg_catalog|sqlite_master)",
            r"(?i)(user_tables|all_tables|dba_tables)",
            
            # Множественные кавычки, которые часто используются в SQL-инъекциях
            r"('|\"){2,}",
            r"(?i)'[\s\+]+\+[\s\+]+'"  # Конкатенация строк в SQL
        ]
        
        # Добавляем пользовательские паттерны
        if custom_patterns is not None:
            self.sql_patterns.extend(custom_patterns)
        
        # Компилируем регулярные выражения для паттернов инъекций
        self.compiled_patterns = [re.compile(pattern) for pattern in self.sql_patterns]
        
        # Белый список стандартных SQL-запросов (новая функциональность)
        self.whitelist_patterns = [
            # Стандартные SELECT запросы
            r"(?i)^SELECT\s+[\w\s,\*\(\)\.]+\s+FROM\s+[\w\.]+(\s+WHERE\s+[\w\s\=\<\>\-\+\.\,\'\"\(\)]+)?$",
            # Стандартные INSERT запросы
            r"(?i)^INSERT\s+INTO\s+[\w\.]+\s*\([\w\s,]+\)\s*VALUES\s*\([\w\s,\'\"\-\+\.]+\)$",
            # Стандартные UPDATE запросы
            r"(?i)^UPDATE\s+[\w\.]+\s+SET\s+[\w\s\=\,\'\"\-\+\.]+(\s+WHERE\s+[\w\s\=\<\>\-\+\.\,\'\"\(\)]+)?$",
            # Стандартные DELETE запросы
            r"(?i)^DELETE\s+FROM\s+[\w\.]+(\s+WHERE\s+[\w\s\=\<\>\-\+\.\,\'\"\(\)]+)?$",
            # Стандартные JOIN запросы
            r"(?i)^SELECT\s+[\w\s,\*\(\)\.]+\s+FROM\s+[\w\.]+\s+JOIN\s+[\w\.]+\s+ON\s+[\w\s\=\<\>\-\+\.\,\'\"\(\)]+(\s+WHERE\s+[\w\s\=\<\>\-\+\.\,\'\"\(\)]+)?$"
        ]
        
        # Добавляем пользовательские белые списки
        if whitelist_patterns is not None:
            self.whitelist_patterns.extend(whitelist_patterns)
        
        # Компилируем регулярные выражения для белого списка
        self.compiled_whitelist = [re.compile(pattern) for pattern in self.whitelist_patterns]
        
        # Контекстная информация
        self.context_history = []
        self.normal_queries_stats = {}
        
        # Логирование
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"EnhancedSQLInjectionDetector initialized with threshold={threshold}")

    def preprocess_data(self, data, train=False):
        """
        Реализация метода предобработки данных из базового класса BaseAnomalyDetector.
        """
        # Создаем копию данных для обработки
        preprocessed_data = data.copy()
        
        # Убеждаемся, что колонка с текстом присутствует
        if self.text_column not in preprocessed_data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
            
        return preprocessed_data

    def train(self, data: pd.DataFrame, **kwargs) -> 'EnhancedSQLInjectionDetector':
        """
        Обучение детектора на нормальных данных.
        Реализация метода train из базового класса BaseAnomalyDetector.
        """
        return self.fit(data, **kwargs)
        
    def fit(self, data: pd.DataFrame, **kwargs) -> 'EnhancedSQLInjectionDetector':
        """
        Обучение детектора на нормальных данных.
        """
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data, train=True)
        
        if self.text_column not in preprocessed_data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
        
        self.logger.info(f"Training on {len(preprocessed_data)} samples")
        
        # Фильтруем данные для обучения
        valid_queries = preprocessed_data[preprocessed_data[self.text_column].apply(
            lambda x: isinstance(x, str) and len(x) >= self.min_query_length
        )]
        
        if len(valid_queries) == 0:
            self.logger.warning("No valid queries found for training")
            return self
        
        # Собираем статистику
        self._collect_normal_stats(valid_queries[self.text_column])
        
        self.logger.info("Training completed")
        self.is_fitted = True
        
        return self
    
    def _collect_normal_stats(self, queries: pd.Series) -> None:
        """Сбор статистики по нормальным запросам."""
        # Статистика энтропии
        entropies = queries.apply(self._calculate_entropy)
        self.normal_queries_stats['mean_entropy'] = entropies.mean()
        self.normal_queries_stats['std_entropy'] = entropies.std()
        
        # Средняя длина запросов
        avg_length = queries.apply(len).mean()
        self.normal_queries_stats['avg_length'] = avg_length
        
        # Частота символов
        all_chars = ''.join(queries.tolist())
        char_counter = Counter(all_chars)
        total_chars = len(all_chars)
        
        self.normal_queries_stats['char_frequency'] = {
            char: count / total_chars for char, count in char_counter.items()
        }
        
        # Статистика для эвристического анализа
        self.normal_queries_stats['avg_uppercase_ratio'] = queries.apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        ).mean()
        
        self.normal_queries_stats['avg_special_char_ratio'] = queries.apply(
            lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace()) / len(x) if len(x) > 0 else 0
        ).mean()
        
        # Собираем статистику часто используемых ключевых слов в безопасных запросах
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN',
                       'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'OFFSET', 'DISTINCT',
                       'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'IN', 'BETWEEN', 'LIKE']
        
        keyword_usage = {}
        for keyword in sql_keywords:
            keyword_usage[keyword] = queries.apply(
                lambda x: 1 if keyword.lower() in x.lower() else 0
            ).mean()
        
        self.normal_queries_stats['keyword_usage'] = keyword_usage
    
    def _calculate_entropy(self, text: str) -> float:
        """Вычисление энтропии Шеннона для текста."""
        if not text or len(text) == 0:
            return 0.0
        
        counter = Counter(text)
        length = len(text)
        
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _is_in_whitelist(self, text: str) -> bool:
        """
        Проверка соответствия запроса паттернам из белого списка.
        Новая функциональность для снижения ложных срабатываний.
        """
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return False
            
        # Проверяем на соответствие белому списку
        for pattern in self.compiled_whitelist:
            if pattern.match(text.strip()):
                return True
                
        return False
    
    def _detect_regex_patterns(self, text: str) -> Tuple[bool, float, List[str]]:
        """Обнаружение паттернов SQL-инъекций с помощью регулярных выражений."""
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return False, 0.0, []
        
        # Сначала проверяем на белый список
        if self._is_in_whitelist(text):
            return False, 0.0, []
        
        detected_patterns = []
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                detected_patterns.append(self.sql_patterns[i])
        
        # Улучшенная оценка на основе количества найденных паттернов
        # Используем логистическую функцию для нормализации
        if detected_patterns:
            pattern_count = len(detected_patterns)
            score = 1 / (1 + math.exp(-pattern_count + 2))
        else:
            score = 0.0
        
        return len(detected_patterns) > 0, score, detected_patterns

    def _analyze_entropy(self, text: str) -> float:
        """Анализ энтропии текста для выявления аномалий."""
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return 0.0
        
        if not self.normal_queries_stats:
            return 0.0
        
        entropy = self._calculate_entropy(text)
        
        mean_entropy = self.normal_queries_stats.get('mean_entropy', 0)
        std_entropy = self.normal_queries_stats.get('std_entropy', 1)
        
        if std_entropy == 0:
            return 0.0
        
        # Улучшенный расчет Z-оценки с учетом длины запроса
        z_score = abs(entropy - mean_entropy) / std_entropy
        
        # Корректировка для коротких запросов
        query_length = len(text)
        avg_length = self.normal_queries_stats.get('avg_length', 20)
        
        # Если запрос короткий, снижаем его значимость
        if query_length < avg_length / 2:
            z_score *= 0.5
        
        score = 1 / (1 + math.exp(-z_score + 2))
        
        return min(score, 1.0)
    
    def _analyze_heuristics(self, text: str) -> float:
        """Расширенный эвристический анализ текста."""
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return 0.0
        
        heuristic_scores = []
        
        # Проверка спецсимволов
        special_chars_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_chars_ratio = special_chars_count / len(text) if len(text) > 0 else 0
        
        normal_ratio = self.normal_queries_stats.get('avg_special_char_ratio', 0.15)
        special_chars_score = min(special_chars_ratio / (normal_ratio * 2), 1.0) if normal_ratio > 0 else 0
        heuristic_scores.append(special_chars_score)
        
        # Проверка регистра
        uppercase_count = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_count / len(text) if len(text) > 0 else 0
        
        normal_uppercase = self.normal_queries_stats.get('avg_uppercase_ratio', 0.2)
        uppercase_score = min(uppercase_ratio / (normal_uppercase * 2), 1.0) if normal_uppercase > 0 else 0
        heuristic_scores.append(uppercase_score)
        
        # Расширенный список ключевых слов SQL
        sql_keywords = [
            'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN', 
            'UNION', 'DROP', 'TABLE', 'DATABASE', 'ALTER', 'CREATE', 'EXEC',
            'EXECUTE', 'TRUNCATE', 'INFORMATION_SCHEMA', 'WAITFOR', 'DELAY',
            'SLEEP', 'BENCHMARK', 'CAST', 'CONVERT', 'DECLARE', 'SHUTDOWN',
            'MASTER', 'ASCII', 'CONCAT', 'SYSDATE', 'VERSION'
        ]
        
        keyword_count = sum(1 for keyword in sql_keywords if keyword.lower() in text.lower())
        
        # Учитываем нормальное использование ключевых слов в безопасных запросах
        normal_keyword_usage = self.normal_queries_stats.get('keyword_usage', {})
        suspicious_keywords = 0
        
        for keyword in sql_keywords:
            expected_usage = normal_keyword_usage.get(keyword, 0.1)
            if keyword.lower() in text.lower() and expected_usage < 0.3:  # Редко используемое ключевое слово
                suspicious_keywords += 1
        
        # Улучшенная оценка для ключевых слов с учетом "подозрительности"
        keyword_score = 1 - (1 / (1 + 0.5 * suspicious_keywords))
        heuristic_scores.append(keyword_score)
        
        # Проверка повторяющихся символов
        repeated_chars = re.findall(r'(.)\1{3,}', text)
        repeated_score = min(len(repeated_chars) / 2, 1.0)
        heuristic_scores.append(repeated_score)
        
        # Проверка комментариев
        comment_pattern = re.compile(r'(--|#|\/\*)')
        has_comments = 1.0 if comment_pattern.search(text) else 0.0
        heuristic_scores.append(has_comments)
        
        # Проверка на точки с запятой (может указывать на многократные запросы)
        semicolons = text.count(';')
        semicolon_score = min(semicolons / 2, 1.0)
        heuristic_scores.append(semicolon_score)
        
        # Проверка на кавычки
        quote_count = text.count("'") + text.count('"')
        quote_score = min(quote_count / 4, 1.0)
        heuristic_scores.append(quote_score)
        
        # Проверка на смешанные операторы SQL
        operators = ['=', '<', '>', '<=', '>=', '!=', '<>', 'LIKE', 'IN', 'BETWEEN']
        operator_count = sum(1 for op in operators if op.lower() in text.lower())
        operator_score = min(operator_count / 3, 1.0)
        heuristic_scores.append(operator_score)
        
        # Вычисляем взвешенную оценку по всем эвристикам
        # Даем больший вес наиболее подозрительным характеристикам
        weighted_scores = [
            heuristic_scores[0] * 0.15,  # спецсимволы
            heuristic_scores[1] * 0.05,  # регистр
            heuristic_scores[2] * 0.30,  # ключевые слова SQL
            heuristic_scores[3] * 0.10,  # повторяющиеся символы
            heuristic_scores[4] * 0.20,  # комментарии
            heuristic_scores[5] * 0.10,  # точки с запятой
            heuristic_scores[6] * 0.05,  # кавычки
            heuristic_scores[7] * 0.05   # операторы SQL
        ]
        
        return sum(weighted_scores)

    def _analyze_context(self, text: str) -> float:
        """Улучшенный контекстный анализ текста."""
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return 0.0
        
        if not self.context_history:
            return 0.0
        
        similarities = []
        for prev_query in self.context_history[-self.context_window:]:
            similarity = self._calculate_similarity(text, prev_query)
            similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        # Улучшенный алгоритм оценки контекстных аномалий с корректировкой для коротких запросов
        context_score = 0.0
        
        if avg_similarity < 0.2:
            # Очень низкая схожесть - сильная аномалия
            context_score = 0.9
        elif avg_similarity < 0.4:
            # Низкая схожесть - умеренная аномалия
            context_score = 0.7
        elif avg_similarity < 0.6:
            # Средняя схожесть - небольшая аномалия
            context_score = 0.5
        else:
            # Высокая схожесть - не аномалия
            context_score = 0.1
        
        # Снижаем значимость контекста для коротких запросов
        query_length = len(text)
        avg_length = self.normal_queries_stats.get('avg_length', 20)
        
        if query_length < avg_length / 2:
            context_score *= 0.3
        
        return context_score
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Улучшенный расчет схожести между текстами."""
        # Используем n-граммы для более точной оценки
        def get_ngrams(text, n=3):
            return set([text[i:i+n] for i in range(len(text) - n + 1)])
        
        if len(text1) < 3 or len(text2) < 3:
            return 0.0
        
        # Используем смесь разных размеров n-грамм для лучших результатов
        ngrams1_2 = get_ngrams(text1.lower(), 2)
        ngrams2_2 = get_ngrams(text2.lower(), 2)
        
        ngrams1_3 = get_ngrams(text1.lower(), 3)
        ngrams2_3 = get_ngrams(text2.lower(), 3)
        
        # Рассчитываем коэффициент Жаккара для 2-грамм и 3-грамм
        if ngrams1_2 and ngrams2_2:
            intersection_2 = len(ngrams1_2.intersection(ngrams2_2))
            union_2 = len(ngrams1_2.union(ngrams2_2))
            jaccard_2 = intersection_2 / union_2 if union_2 > 0 else 0.0
        else:
            jaccard_2 = 0.0
            
        if ngrams1_3 and ngrams2_3:
            intersection_3 = len(ngrams1_3.intersection(ngrams2_3))
            union_3 = len(ngrams1_3.union(ngrams2_3))
            jaccard_3 = intersection_3 / union_3 if union_3 > 0 else 0.0
        else:
            jaccard_3 = 0.0
        
        # Возвращаем взвешенную комбинацию метрик
        return 0.4 * jaccard_2 + 0.6 * jaccard_3

    def _calculate_anomaly_score(self, text: str) -> Dict[str, Any]:
        """Расчет общей оценки аномальности с улучшенным алгоритмом."""
        result = {
            'anomaly_score': 0.0,
            'is_anomaly': False,
            'anomaly_type': None,
            'details': {}
        }
        
        if not isinstance(text, str) or len(text) < self.min_query_length:
            return result
        
        # Проверка на белый список
        if self._is_in_whitelist(text):
            result['details']['whitelist'] = True
            return result
        
        component_scores = {}
        
        # Регулярные выражения
        if self.use_regex:
            has_patterns, regex_score, detected_patterns = self._detect_regex_patterns(text)
            component_scores['regex'] = regex_score
            result['details']['regex'] = {
                'score': regex_score,
                'detected_patterns': detected_patterns
            }
        
        # Анализ энтропии
        if self.use_entropy:
            entropy_score = self._analyze_entropy(text)
            component_scores['entropy'] = entropy_score
            result['details']['entropy'] = {
                'score': entropy_score,
                'value': self._calculate_entropy(text)
            }
        
        # Эвристический анализ
        if self.use_heuristics:
            heuristic_score = self._analyze_heuristics(text)
            component_scores['heuristics'] = heuristic_score
            result['details']['heuristics'] = {
                'score': heuristic_score
            }
        
        # Контекстный анализ
        if self.use_context and self.context_history:
            context_score = self._analyze_context(text)
            component_scores['context'] = context_score
            result['details']['context'] = {
                'score': context_score
            }
        
        # Вычисляем средневзвешенную оценку
        weighted_score = 0.0
        for component, score in component_scores.items():
            if component in self.weights:
                weighted_score += score * self.weights[component]
        
        # Улучшенный алгоритм оценки аномалий
        # Даем больший вес наиболее значимым компонентам
        max_component_name, max_component_score = max(component_scores.items(), key=lambda x: x[1]) if component_scores else ('none', 0.0)
        
        # Если хотя бы один компонент имеет очень высокую оценку (>0.8), увеличиваем общую оценку
        if max_component_score > 0.8:
            weighted_score = max(weighted_score, 0.7)  # Гарантируем, что оценка будет достаточно высокой
            
        # Если регулярное выражение обнаружило явные паттерны SQL-инъекции, увеличиваем оценку
        if 'regex' in component_scores and component_scores['regex'] > 0.6:
            weighted_score = max(weighted_score, 0.65)
        
        # Корректировка для коротких запросов - снижаем вероятность ложных срабатываний
        query_length = len(text)
        avg_length = self.normal_queries_stats.get('avg_length', 20)
        
        if query_length < avg_length / 2:
            weighted_score *= 0.85  # Уменьшаем оценку для коротких запросов
        
        result['anomaly_score'] = min(weighted_score, 1.0)
        result['is_anomaly'] = result['anomaly_score'] >= self.threshold
        
        # Определяем тип аномалии
        if result['is_anomaly'] and component_scores:
            if max_component_name == 'regex' and max_component_score > 0:
                result['anomaly_type'] = 'SQL Injection'
            elif max_component_name == 'entropy' and max_component_score > 0:
                result['anomaly_type'] = 'Unusual Query Structure'
            elif max_component_name == 'heuristics' and max_component_score > 0:
                result['anomaly_type'] = 'Suspicious SQL Pattern'
            elif max_component_name == 'context' and max_component_score > 0:
                result['anomaly_type'] = 'Contextual Anomaly'
            else:
                result['anomaly_type'] = 'Unknown'
        
        return result

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обнаружение аномалий в данных.
        Реализация метода predict из базового класса BaseAnomalyDetector.
        """
        # Предобработка данных
        preprocessed_data = self.preprocess_data(data)
        
        if self.text_column not in preprocessed_data.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in data")
        
        self.logger.info(f"Predicting anomalies for {len(preprocessed_data)} samples")
        
        # Создаем DataFrame для результатов
        result_df = preprocessed_data.copy()
        
        # Добавляем столбцы для результатов
        result_df['anomaly_score'] = 0.0
        result_df['predicted_anomaly'] = 0
        result_df['anomaly_type'] = None
        result_df['analysis_details'] = None
        
        # Анализируем каждый запрос
        for idx, text in preprocessed_data[self.text_column].items():
            anomaly_result = self._calculate_anomaly_score(text)
            
            result_df.at[idx, 'anomaly_score'] = anomaly_result['anomaly_score']
            result_df.at[idx, 'predicted_anomaly'] = 1 if anomaly_result['is_anomaly'] else 0
            result_df.at[idx, 'anomaly_type'] = anomaly_result['anomaly_type']
            result_df.at[idx, 'analysis_details'] = anomaly_result['details']
            
            # Обновляем историю запросов для контекстного анализа
            if isinstance(text, str) and len(text) >= self.min_query_length:
                self.context_history.append(text)
                if len(self.context_history) > self.context_window * 2:
                    self.context_history = self.context_history[-self.context_window * 2:]
            
            # Автоматическое обучение на безопасных запросах (если включено)
            if self.auto_learning and anomaly_result['anomaly_score'] < 0.3:
                if isinstance(text, str) and len(text) >= self.min_query_length:
                    # Добавляем в статистику безопасных запросов
                    if 'safe_queries' not in self.normal_queries_stats:
                        self.normal_queries_stats['safe_queries'] = []
                    
                    self.normal_queries_stats['safe_queries'].append(text)
                    
                    # Периодически обновляем статистику
                    if len(self.normal_queries_stats['safe_queries']) % 10 == 0:
                        self._update_normal_stats()
        
        anomaly_count = result_df['predicted_anomaly'].sum()
        self.logger.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(preprocessed_data)*100:.2f}%)")
        
        return result_df
    
    def _update_normal_stats(self):
        """Обновляет статистику на основе сохраненных безопасных запросов."""
        if 'safe_queries' in self.normal_queries_stats and self.normal_queries_stats['safe_queries']:
            self._collect_normal_stats(pd.Series(self.normal_queries_stats['safe_queries']))
    
    def detect_and_explain(self, data: pd.DataFrame) -> pd.DataFrame:
        """Обнаружение и объяснение SQL-инъекций."""
        # Получаем базовые результаты
        result_df = self.predict(data)
        
        # Добавляем объяснения для аномалий
        explanations = []
        for idx, row in result_df.iterrows():
            if row['predicted_anomaly'] == 1:
                text = data.loc[idx, self.text_column]
                explanation = self._generate_explanation(text)
                explanations.append(explanation)
            else:
                explanations.append(None)
        
        result_df['explanation'] = explanations
        
        return result_df

    def _generate_explanation(self, text: str) -> str:
        """Генерация подробного объяснения для аномалии."""
        anomaly_result = self._calculate_anomaly_score(text)
        
        explanation_parts = []
        
        anomaly_type = anomaly_result.get('anomaly_type', 'Unknown')
        
        if anomaly_type == 'SQL Injection':
            explanation_parts.append("Обнаружены паттерны SQL-инъекции в запросе.")
            
            patterns = anomaly_result.get('details', {}).get('regex', {}).get('detected_patterns', [])
            if patterns:
                explanation_parts.append(f"Найденные паттерны ({len(patterns)}):")
                for i, pattern in enumerate(patterns[:3]):  # Показываем только первые 3 паттерна
                    pattern_desc = pattern.replace(r'(?i)', '').replace(r'\b', '').replace(r'\s+', ' ')
                    explanation_parts.append(f"  - Паттерн {i+1}: {pattern_desc}")
                
                if len(patterns) > 3:
                    explanation_parts.append(f"  - ... и еще {len(patterns) - 3} паттернов")
                    
                # Добавляем дополнительную информацию о типе SQL-инъекции
                if any("UNION" in p for p in patterns):
                    explanation_parts.append("Тип атаки: UNION-инъекция (попытка объединения запросов)")
                elif any("DROP" in p for p in patterns):
                    explanation_parts.append("Тип атаки: Деструктивная инъекция (попытка удаления данных)")
                elif any("OR" in p and "=" in p for p in patterns):
                    explanation_parts.append("Тип атаки: Boolean-based инъекция (манипуляция условиями)")
                elif any("EXEC" in p or "xp_" in p for p in patterns):
                    explanation_parts.append("Тип атаки: Выполнение команд (высокая опасность)")
                elif any("SLEEP" in p or "WAITFOR" in p or "BENCHMARK" in p for p in patterns):
                    explanation_parts.append("Тип атаки: Time-based инъекция (слепая)")
                elif any("--" in p or "#" in p or "/*" in p for p in patterns):
                    explanation_parts.append("Техника: Использование комментариев для обхода фильтрации")
        
        elif anomaly_type == 'Unusual Query Structure':
            explanation_parts.append("Запрос имеет необычную структуру с высокой энтропией.")
            
            entropy_value = anomaly_result.get('details', {}).get('entropy', {}).get('value', 0)
            normal_entropy = self.normal_queries_stats.get('mean_entropy', 0)
            
            explanation_parts.append(f"Энтропия запроса: {entropy_value:.2f}, " 
                                    f"средняя нормальная энтропия: {normal_entropy:.2f}")
            
            # Дополнительный анализ необычной структуры
            if "'" in text or '"' in text:
                quote_count = text.count("'") + text.count('"')
                explanation_parts.append(f"Запрос содержит необычное количество кавычек ({quote_count})")
            
            if ";" in text:
                explanation_parts.append("Запрос содержит точки с запятой, возможно множественные запросы")
            
            if len(text) > 2 * self.normal_queries_stats.get('avg_length', 0):
                explanation_parts.append("Запрос необычно длинный, что может указывать на инъекцию")
        
        elif anomaly_type == 'Suspicious SQL Pattern':
            explanation_parts.append("Запрос содержит подозрительные SQL-конструкции.")
            
            # Анализируем ключевые слова SQL
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'FROM', 'WHERE', 'JOIN', 
                           'UNION', 'DROP', 'TABLE', 'DATABASE', 'ALTER', 'CREATE', 'EXEC',
                           'EXECUTE', 'TRUNCATE', 'INFORMATION_SCHEMA', 'WAITFOR', 'DELAY']
            
            found_keywords = [keyword for keyword in sql_keywords if keyword.lower() in text.lower()]
            
            if found_keywords:
                explanation_parts.append(f"Найденные ключевые слова SQL: {', '.join(found_keywords)}")
            
            # Анализируем подозрительные конструкции
            if re.search(r'(--|#|\/\*)', text):
                explanation_parts.append("Запрос содержит комментарии, которые могут использоваться для обхода фильтров")
            
            if re.search(r'=\s*[\'\"]?\s*\w+\s*[\'\"]?\s*$', text):
                explanation_parts.append("Запрос заканчивается условием равенства, возможно часть инъекции")
            
            if re.search(r'(LIKE|IN|BETWEEN)', text, re.IGNORECASE):
                explanation_parts.append("Запрос содержит операторы сравнения, часто используемые в инъекциях")
                
            if re.search(r'(sysobjects|syscolumns|information_schema)', text, re.IGNORECASE):
                explanation_parts.append("Запрос обращается к системным таблицам, возможно разведка базы данных")
        
        elif anomaly_type == 'Contextual Anomaly':
            explanation_parts.append("Запрос значительно отличается от предыдущих запросов.")
            
            context_score = anomaly_result.get('details', {}).get('context', {}).get('score', 0)
            explanation_parts.append(f"Оценка контекстной аномальности: {context_score:.2f}")
            explanation_parts.append("Этот запрос сильно отличается от паттернов нормального поведения.")
            
            # Анализируем причины контекстной аномалии
            if len(text) > 2 * self.normal_queries_stats.get('avg_length', 0):
                explanation_parts.append("Запрос значительно длиннее обычных запросов")
            
            if text.count(" ") > 3 * self.normal_queries_stats.get('avg_length', 0) / 5:
                explanation_parts.append("Запрос содержит необычно большое количество пробелов")
                
            if re.search(r'[^\w\s]', text):
                special_chars = re.findall(r'[^\w\s]', text)
                unusual_chars = set(special_chars) - set(['.', ',', '(', ')', '='])
                if unusual_chars:
                    explanation_parts.append(f"Запрос содержит необычные символы: {''.join(unusual_chars)}")
        
        else:
            explanation_parts.append(f"Обнаружена аномалия неизвестного типа: {anomaly_type}")
        
        # Уровень угрозы
        score = anomaly_result.get('anomaly_score', 0)
        if score > 0.8:
            explanation_parts.append("УРОВЕНЬ УГРОЗЫ: ВЫСОКИЙ - необходимо немедленное внимание")
        elif score > 0.6:
            explanation_parts.append("УРОВЕНЬ УГРОЗЫ: СРЕДНИЙ - требуется дополнительный анализ")
        else:
            explanation_parts.append("УРОВЕНЬ УГРОЗЫ: НИЗКИЙ - возможно ложное срабатывание")
            
        explanation_parts.append(f"Общая оценка аномальности: {score:.2f}")
        
        # Рекомендации по смягчению
        explanation_parts.append("\nРекомендации:")
        if anomaly_type == 'SQL Injection':
            explanation_parts.append("- Используйте параметризованные запросы вместо конкатенации строк")
            explanation_parts.append("- Внедрите проверку входных данных и экранирование спецсимволов")
            explanation_parts.append("- Рассмотрите возможность применения ORM для доступа к базе данных")
        elif anomaly_type in ['Unusual Query Structure', 'Suspicious SQL Pattern']:
            explanation_parts.append("- Проверьте источник запроса и его легитимность")
            explanation_parts.append("- Внедрите систему белых списков для допустимых SQL-запросов")
        
        return "\n".join(explanation_parts)
        
    def add_to_whitelist(self, pattern: str) -> None:
        """
        Добавляет паттерн в белый список.
        
        Args:
            pattern: Регулярное выражение для безопасного запроса
        """
        if pattern not in self.whitelist_patterns:
            self.whitelist_patterns.append(pattern)
            self.compiled_whitelist.append(re.compile(pattern))
            self.logger.info(f"Pattern added to whitelist: {pattern}")
            
    def evaluate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Оценка производительности детектора.
        
        Args:
            data: DataFrame с колонкой is_anomaly для сравнения
            
        Returns:
            Dict: Словарь с метриками производительности
        """
        if 'is_anomaly' not in data.columns:
            raise ValueError("Для оценки необходимы данные с колонкой 'is_anomaly'")
        
        # Получаем предсказания
        predictions = self.predict(data)
        
        # Используем базовый метод evaluate из BaseAnomalyDetector
        return super().evaluate(predictions)
