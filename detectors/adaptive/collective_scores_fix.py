import numpy as np
import pandas as pd

def fix_collective_scores(data: pd.DataFrame) -> np.ndarray:
    """
    Вычисляет оценки коллективных аномалий.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Данные для анализа
        
    Returns:
    --------
    numpy.ndarray
        Оценки коллективных аномалий
    """
    # Инициализируем оценки аномальности нулями
    scores = np.zeros(len(data))
    
    # Проверяем наличие временной метки
    if 'timestamp' not in data.columns:
        return scores
    
    # Анализируем временные паттерны для ключевых признаков
    key_features = ['bytes_per_second', 'packets_per_second', 'connection_count']
    available_features = [f for f in key_features if f in data.columns]
    
    if not available_features:
        return scores
    
    try:
        # Сортируем данные по времени
        sorted_data = data.sort_values('timestamp').reset_index(drop=False)
        
        # Получаем маппинг между отсортированными и оригинальными индексами
        idx_mapping = dict(enumerate(sorted_data.index))
        
        # Размер окна для анализа последовательностей
        window_size = min(10, len(data) // 10)
        if window_size < 2:  # Предотвращаем слишком маленькие окна
            return scores
        
        # Анализируем каждый доступный признак
        for feature in available_features:
            # Создаем скользящее окно
            for i in range(len(sorted_data) - window_size + 1):
                window = sorted_data[feature].iloc[i:i+window_size].values
                
                # Анализ резких изменений в окне
                if len(window) > 1:
                    # Вычисляем разности между соседними значениями
                    diffs = np.abs(np.diff(window))
                    if len(diffs) == 0:
                        continue
                        
                    mean_diff = np.mean(diffs)
                    if mean_diff == 0:
                        continue
                    
                    # Если есть резкие изменения, увеличиваем оценку аномальности
                    for j in range(len(diffs)):
                        if diffs[j] > 3 * mean_diff:
                            orig_idx1 = idx_mapping[i + j]
                            orig_idx2 = idx_mapping[i + j + 1]
                            
                            # Используем позиции в оригинальном массиве scores
                            pos1 = data.index.get_loc(orig_idx1)
                            pos2 = data.index.get_loc(orig_idx2)
                            
                            anomaly_score = diffs[j] / (mean_diff + 1e-10) - 3
                            scores[pos1] += anomaly_score
                            scores[pos2] += anomaly_score
    except Exception as e:
        print(f"Ошибка при обнаружении коллективных аномалий: {e}")
        
    return scores
