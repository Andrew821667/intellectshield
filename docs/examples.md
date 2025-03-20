# Примеры использования IntellectShield

## Базовое использование

### Обнаружение аномалий с помощью Isolation Forest

```python
from intellectshield.detectors.isolation_forest import IsolationForestDetector
from intellectshield.data.data_loader import DataLoader

# Загрузка данных
loader = DataLoader(file_path="your_data.csv")
X = loader.load()

# Создание и обучение детектора
detector = IsolationForestDetector(n_estimators=100, contamination=0.1)
detector.fit(X)

# Обнаружение аномалий
anomalies = detector.predict(X)
print(f"Обнаружено {sum(anomalies == -1)} аномалий из {len(X)} наблюдений")
```

### Использование ансамблевого детектора

```python
from intellectshield.detectors.ensemble import EnsembleAnomalyDetector
from intellectshield.detectors.isolation_forest import IsolationForestDetector
from intellectshield.detectors.lof import LOFDetector

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
