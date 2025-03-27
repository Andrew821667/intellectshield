# IntellectShield

IntellectShield - это интеллектуальная система обнаружения и предотвращения кибератак на основе машинного обучения.

## Возможности

- **Обнаружение SQL-инъекций**: Выявление атак с использованием SQL инъекций в пользовательских запросах
- **Детекторы аномалий**: Обнаружение статистических выбросов и нетипичного поведения
- **Адаптивные методы**: Алгоритмы, адаптирующиеся к изменениям в данных и обучающиеся на новых угрозах
- **Ансамблевые детекторы**: Комбинирование различных методов для повышения точности обнаружения

## Структура проекта

```
intellectshield/
├── detectors/           # Модули обнаружения атак
│   ├── adaptive/        # Адаптивные детекторы
│   ├── sql_injection/   # Детекторы SQL инъекций
│   ├── dos/             # Детекторы DoS атак
│   ├── ensemble/        # Ансамблевые детекторы
│   └── anomaly/         # Детекторы аномалий
├── utils/               # Вспомогательные утилиты
└── visualizers/         # Визуализаторы результатов
```

## Быстрый старт

### Установка

```bash
git clone https://github.com/Andrew821667/intellectshield.git
cd intellectshield
pip install -r requirements.txt
```

### Пример использования

Обнаружение SQL-инъекций:

```python
import pandas as pd
from intellectshield.detectors.sql_injection.enhanced_v4 import EnhancedSQLInjectionDetectorV4

# Подготовка данных
queries = [
    "SELECT * FROM users",
    "SELECT * FROM users WHERE id = 5",
    "SELECT * FROM users WHERE id = '5' OR 1=1 --'"
]

df = pd.DataFrame({'query': queries})

# Использование детектора
detector = EnhancedSQLInjectionDetectorV4()
results = detector.predict(df)
print(results)
```

## Документация

Подробная документация доступна в директории [docs](docs/):
- [Примеры использования](docs/usage_examples.md)
- [Структура проекта](docs/project_structure.md)

## Лицензия

MIT
