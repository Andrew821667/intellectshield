# План рефакторинга для ./detectors/ensemble.py (сложность: 34)

1. Разделить класс EnsembleAnomalyDetector:
   - BaseEnsemble: Базовый функционал ансамбля
   - VotingEnsemble: Реализация голосования
   - StackingEnsemble: Реализация стекинга
   - WeightedEnsemble: Реализация взвешенного ансамбля

2. Использовать паттерн "Стратегия" для различных стратегий агрегации результатов

3. Упростить методы fit, predict и score
