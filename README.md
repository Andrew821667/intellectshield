# IntellectShield

Библиотека для обнаружения аномалий и кибератак с использованием методов машинного обучения.

## Обзор

IntellectShield - это фреймворк для обнаружения аномалий и потенциальных кибератак в сетевом трафике и поведении систем. Проект использует различные алгоритмы машинного обучения для выявления подозрительной активности.

## Основные возможности

- Обнаружение аномалий с использованием различных алгоритмов (Isolation Forest, LOF, и др.)
- Обнаружение DoS-атак
- Анализ последовательностей событий для выявления подозрительной активности
- Ансамблевые методы для повышения точности обнаружения
- Адаптивные детекторы с автоматической настройкой параметров
- Визуализация результатов обнаружения

## Структура проекта

- `analyzers/` - Модули для анализа данных
- `api/` - API интерфейсы
- `data/` - Обработка и загрузка данных
- `detectors/` - Различные детекторы аномалий и атак
  - `base.py` - Базовый класс детектора
  - `dos.py` - Детектор DoS атак
  - `isolation_forest.py` - Детектор на основе Isolation Forest
  - `lof.py` - Детектор на основе Local Outlier Factor
  - `sequence.py` - Детектор для анализа последовательностей
  - `ensemble.py` - Ансамблевый детектор
  - `enhanced_adaptive_detector_*.py` - Улучшенный адаптивный детектор
- `utils/` - Вспомогательные утилиты
- `visualizers/` - Инструменты для визуализации результатов

## Установка

```bash
# Клонирование репозитория
git clone https://github.com/Andrew821667/intellectshield.git
cd intellectshield

# Установка зависимостей
pip install -r requirements.txt
```

## Использование

Пример использования детектора на основе Isolation Forest:

```python
from intellectshield.detectors.isolation_forest import IsolationForestDetector
from intellectshield.data.data_loader import DataLoader

# Загрузка данных
data_loader = DataLoader(file_path="your_data.csv")
X = data_loader.load()

# Создание и обучение детектора
detector = IsolationForestDetector()
detector.fit(X)

# Обнаружение аномалий
anomalies = detector.predict(X)
print(f"Обнаружено {sum(anomalies == -1)} аномалий из {len(X)} наблюдений")
```

Пример использования ансамблевого детектора:

```python
from intellectshield.detectors.ensemble import EnsembleAnomalyDetector
from intellectshield.detectors.isolation_forest import IsolationForestDetector
from intellectshield.detectors.lof import LOFDetector
from intellectshield.data.data_loader import DataLoader

# Загрузка данных
data_loader = DataLoader(file_path="your_data.csv")
X = data_loader.load()

# Создание базовых детекторов
detectors = [
    IsolationForestDetector(n_estimators=100),
    LOFDetector(n_neighbors=20)
]

# Создание ансамблевого детектора
ensemble = EnsembleAnomalyDetector(detectors=detectors, voting='majority')

# Обучение и предсказание
ensemble.fit(X)
anomalies = ensemble.predict(X)
```

## Лицензия

MIT
