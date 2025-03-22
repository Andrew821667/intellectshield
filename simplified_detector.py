import re
import pandas as pd

class AdvancedSQLDetector:
    """
    Улучшенный детектор SQL-инъекций V3 с оптимизированными функциями обнаружения.
    """
    
    def __init__(self, text_column='query', threshold=0.5, verbose=False):
        self.text_column = text_column
        self.threshold = threshold
        self.verbose = verbose
        
        # Критические паттерны SQL-инъекций
        self.critical_patterns = [
            r"OR 1=1",
            r"OR '1'='1'",
            r"--",
            r"#",
            r"UNION SELECT",
            r"; DROP",
            r"xp_cmdshell",
            r"WAITFOR DELAY"
        ]
        
        # Вторичные паттерны
        self.secondary_patterns = [
            r"OR 1>",
            r"LIKE '%",
            r"; ALTER",
            r"ORDER BY \d+",
            r"GROUP BY .* HAVING"
        ]
        
        # Категории атак
        self.attack_categories = {
            'boolean_based': "Boolean-based SQL Injection",
            'union_based': "UNION-based SQL Injection",
            'time_based': "Time-based Blind SQL Injection",
            'destructive': "Destructive SQL Command",
            'command_exec': "OS Command Execution"
        }
        
        # Рекомендации
        self.security_recommendations = {
            'boolean_based': "Используйте параметризованные запросы вместо прямой конкатенации строк.",
            'union_based': "Применяйте ORM-фреймворки и строгую валидацию входных данных.",
            'time_based': "Установите тайм-ауты для SQL-запросов.",
            'destructive': "Ограничьте права доступа пользователя БД.",
            'command_exec': "Никогда не используйте xp_cmdshell в продакшене."
        }
        
        self.default_recommendation = "Используйте параметризованные запросы и валидацию ввода."
    
    def fit(self, data):
        """Метод для совместимости с API."""
        if self.verbose:
            print(f"Детектор инициализирован с {len(self.critical_patterns)} критическими паттернами.")
        return self
    
    def predict(self, data):
        """Предсказание SQL-инъекций."""
        result = data.copy()
        result['anomaly_score'] = 0.0
        result['predicted_anomaly'] = 0
        result['threat_level'] = None
        result['attack_type'] = None
        
        for idx, row in data.iterrows():
            text = row[self.text_column]
            if not isinstance(text, str):
                continue
            
            # Приводим к нижнему регистру для регистронезависимого поиска
            text_lower = text.lower()
            
            # Проверка критических паттернов
            critical_match = False
            for pattern in self.critical_patterns:
                if re.search(pattern.lower(), text_lower, re.IGNORECASE):
                    critical_match = True
                    break
            
            if critical_match:
                # Определяем тип атаки
                attack_type = self._determine_attack_type(text_lower)
                
                result.loc[idx, 'anomaly_score'] = 1.0
                result.loc[idx, 'predicted_anomaly'] = 1
                result.loc[idx, 'threat_level'] = 'High'
                result.loc[idx, 'attack_type'] = attack_type
                continue
            
            # Проверка вторичных паттернов
            secondary_matches = []
            for pattern in self.secondary_patterns:
                if re.search(pattern.lower(), text_lower, re.IGNORECASE):
                    secondary_matches.append(pattern)
            
            if secondary_matches:
                score = min(1.0, len(secondary_matches) * 0.25)
                attack_type = self._determine_attack_type(text_lower)
                
                result.loc[idx, 'anomaly_score'] = score
                result.loc[idx, 'predicted_anomaly'] = 1 if score >= self.threshold else 0
                if score >= self.threshold:
                    result.loc[idx, 'threat_level'] = 'Medium' if score >= 0.75 else 'Low'
                    result.loc[idx, 'attack_type'] = attack_type
            
            # Эвристики для пропущенных случаев
            if result.loc[idx, 'predicted_anomaly'] == 0:
                # Проверка несбалансированных кавычек
                single_quotes = text.count("'")
                if single_quotes % 2 != 0:
                    result.loc[idx, 'anomaly_score'] = 0.6
                    result.loc[idx, 'predicted_anomaly'] = 1
                    result.loc[idx, 'threat_level'] = 'Medium'
                    result.loc[idx, 'attack_type'] = 'syntax_manipulation'
        
        return result
    
    def detect_and_explain(self, data):
        """Обнаружение и объяснение SQL-инъекций."""
        result = self.predict(data)
        
        # Добавляем объяснения и рекомендации
        explanations = []
        recommendations = []
        
        for idx, row in result.iterrows():
            if row['predicted_anomaly'] == 1:
                text = data.loc[idx, self.text_column]
                explanation = self._generate_explanation(text, row['threat_level'], row['attack_type'])
                recommendation = self._generate_recommendation(row['attack_type'])
                
                explanations.append(explanation)
                recommendations.append(recommendation)
            else:
                explanations.append(None)
                recommendations.append(None)
        
        result['explanation'] = explanations
        result['recommendation'] = recommendations
        
        return result
    
    def _determine_attack_type(self, text):
        """Определение типа атаки по тексту."""
        text_lower = text.lower()
        
        if 'or 1=1' in text_lower or "or '1'='1'" in text_lower:
            return 'boolean_based'
        elif 'union select' in text_lower:
            return 'union_based'
        elif 'waitfor' in text_lower or 'sleep' in text_lower or 'benchmark' in text_lower:
            return 'time_based'
        elif 'drop' in text_lower or 'delete' in text_lower or 'truncate' in text_lower:
            return 'destructive'
        elif 'xp_cmdshell' in text_lower or 'exec' in text_lower:
            return 'command_exec'
        else:
            return 'generic_injection'
    
    def _generate_explanation(self, text, threat_level, attack_type):
        """Генерация объяснения для аномалии."""
        explanation_parts = []
        
        # Добавляем информацию об уровне угрозы
        explanation_parts.append(f"[{threat_level} Risk] Обнаружены признаки SQL-инъекции в запросе.")
        
        # Добавляем категорию атаки
        if attack_type in self.attack_categories:
            explanation_parts.append(f"Тип атаки: {self.attack_categories[attack_type]}")
        else:
            explanation_parts.append("Тип атаки: Неклассифицированная SQL-инъекция")
        
        # Находим все совпадающие паттерны
        matching_patterns = []
        for pattern in self.critical_patterns + self.secondary_patterns:
            if re.search(pattern.lower(), text.lower(), re.IGNORECASE):
                matching_patterns.append(pattern)
        
        if matching_patterns:
            explanation_parts.append("Обнаруженные подозрительные паттерны:")
            for i, pattern in enumerate(matching_patterns[:3]):
                explanation_parts.append(f" - Паттерн {i+1}: {pattern}")
            
            if len(matching_patterns) > 3:
                explanation_parts.append(f" - ... и еще {len(matching_patterns) - 3} паттернов")
        
        return "\n".join(explanation_parts)
    
    def _generate_recommendation(self, attack_type):
        """Генерация рекомендаций по безопасности."""
        recommendation_parts = ["Рекомендации по защите:"]
        
        # Добавляем рекомендации по типу атаки
        if attack_type in self.security_recommendations:
            recommendation_parts.append(f"• {self.security_recommendations[attack_type]}")
        else:
            recommendation_parts.append(f"• {self.default_recommendation}")
        
        # Общие рекомендации
        recommendation_parts.append("• Используйте принцип наименьших привилегий для пользователей БД.")
        recommendation_parts.append("• Реализуйте строгую валидацию пользовательских входных данных.")
        recommendation_parts.append("• Внедрите мониторинг и логирование SQL-запросов для выявления аномалий.")
        
        return "\n".join(recommendation_parts)
