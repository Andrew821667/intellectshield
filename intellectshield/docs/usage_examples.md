# Примеры использования IntellectShield

В этом документе представлены примеры использования различных детекторов из библиотеки IntellectShield.

## Детектор SQL-инъекций

```python
import pandas as pd
from intellectshield.detectors.sql_injection.enhanced_v4 import EnhancedSQLInjectionDetectorV4

# Подготовка тестовых данных
test_queries = [
    "SELECT * FROM users",
    "SELECT * FROM users WHERE id = 5",
    "SELECT * FROM users WHERE id = '5' OR 1=1 --'",
    "SELECT * FROM users WHERE username = 'admin' UNION SELECT * FROM sensitive_data"
]

df = pd.DataFrame({
    'query': test_queries,
    'timestamp': pd.date_range(start='2023-01-01', periods=len(test_queries), freq='H')
})

# Использование детектора SQL-инъекций
sql_detector = EnhancedSQLInjectionDetectorV4()
results = sql_detector.predict(df)
print("Результаты детектора SQL-инъекций:")
print(results)
```

## Обучение детектора SQL-инъекций

```python
# Подготовка обучающих данных
training_queries = [
    "SELECT * FROM users WHERE username = ?",
    "INSERT INTO logs (timestamp, action) VALUES (?, ?)",
    "UPDATE settings SET value = ? WHERE key = ?",
    "SELECT * FROM users WHERE username = 'admin' OR 1=1",
    "SELECT * FROM users WHERE username = '' OR ''=''",
    "SELECT * FROM users WHERE username = 'admin'; DROP TABLE users; --'"
]

# Метки: 0 - легитимный запрос, 1 - SQL инъекция
labels = [0, 0, 0, 1, 1, 1]

# Создаем DataFrame с запросами и метками
training_df = pd.DataFrame({
    'query': training_queries,
    'is_attack': labels
})

# Обучение модели
sql_detector = EnhancedSQLInjectionDetectorV4()
sql_detector.fit(training_df)  # Передаем DataFrame с колонкой 'is_attack'

# Проверка работы обученной модели
predictions = sql_detector.predict(df)
```

## Ансамблевый детектор аномалий

```python
import numpy as np
from intellectshield.detectors.anomaly.isolation_forest import IsolationForestDetector
from intellectshield.detectors.adaptive.base import EnhancedAdaptiveDetector
from intellectshield.detectors.ensemble.ensemble import EnsembleAnomalyDetector

# Создаем объекты базовых детекторов
isolation_forest = IsolationForestDetector()
adaptive_detector = EnhancedAdaptiveDetector()

# Создаем ансамблевый детектор
ensemble = EnsembleAnomalyDetector()
ensemble.add_detector(isolation_forest)
ensemble.add_detector(adaptive_detector)

# Готовим данные для анализа
network_data = pd.DataFrame({
    'bytes_per_second': np.random.exponential(scale=1000, size=100),
    'packets_per_second': np.random.exponential(scale=50, size=100),
    'connection_count': np.random.poisson(lam=5, size=100),
    'timestamp': pd.date_range(start='2023-01-01', periods=100, freq='5min')
})

# Добавляем аномалии
network_data.loc[20:25, 'bytes_per_second'] *= 10
network_data.loc[50:52, 'packets_per_second'] *= 20
network_data.loc[80:85, 'connection_count'] *= 5

# Обучаем и выполняем предсказание
ensemble.train(network_data)
predictions = ensemble.predict(network_data)
```

## Детектор на основе изоляционного леса

```python
# Использование детектора на основе изоляционного леса
iso_detector = IsolationForestDetector()

# Обучаем модель
iso_detector.train(network_data)

# Определяем аномалии
anomaly_results = iso_detector.predict(network_data)
```

## Сохранение и загрузка моделей

```python
import os

# Создаем директорию для моделей
os.makedirs('saved_models', exist_ok=True)

# Сохраняем модель SQL-инъекций
model_path = 'saved_models/sql_injection_model.pkl'
sql_detector.save_model(model_path)

# Загружаем модель из файла
loaded_detector = EnhancedSQLInjectionDetectorV4()
loaded_detector.load_model(model_path)
```
