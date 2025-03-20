"""Модуль для загрузки и предобработки данных.

Этот модуль предоставляет функциональность для загрузки данных из различных источников
и их подготовки для использования в детекторах аномалий.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import urllib.request
import gzip
import time
from sklearn.preprocessing import StandardScaler

def load_kdd_cup_data(sample_size=20000):
    """
    Загрузка и подготовка набора данных KDD Cup 1999 для обнаружения аномалий.
    
    Parameters:
    -----------
    sample_size : int
        Размер выборки из полного набора данных

    Returns:
    --------
    tuple
        (train_data, test_data) - наборы данных для обучения и тестирования
    """
    print("Загрузка и подготовка данных KDD Cup...")
    
    # Создаем директорию для данных
    data_dir = "datasets"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # URL для KDD Cup 1999 Dataset
    train_url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
    
    # Пути к локальным файлам
    train_gz_path = os.path.join(data_dir, "kddcup.data_10_percent.gz")
    train_path = os.path.join(data_dir, "kdd_train.csv")
    
    # Загрузка файлов, если они не существуют
    if not os.path.exists(train_path):
        print("Загрузка обучающего набора...")
        if not os.path.exists(train_gz_path):
            try:
                urllib.request.urlretrieve(train_url, train_gz_path)
            except Exception as e:
                print(f"Ошибка при загрузке данных: {e}")
                return generate_synthetic_data(sample_size=sample_size, anomaly_ratio=0.2)
        
        # Распаковка файла
        try:
            with gzip.open(train_gz_path, 'rb') as f_in:
                with open(train_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        except Exception as e:
            print(f"Ошибка при распаковке данных: {e}")
            return generate_synthetic_data(sample_size=sample_size, anomaly_ratio=0.2)
    
    # Имена столбцов для KDD Cup Dataset
    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label'
    ]
    
    # Чтение данных
    try:
        print("Чтение данных...")
        all_data = pd.read_csv(train_path, header=None, names=column_names, nrows=sample_size)
    except Exception as e:
        print(f"Ошибка при чтении данных: {e}")
        return generate_synthetic_data(sample_size=sample_size, anomaly_ratio=0.2)
    
    # Анализ типов атак
    attack_types = all_data['label'].value_counts()
    print("\nРаспределение типов атак в данных:")
    for attack, count in attack_types.items():
        print(f"{attack}: {count} ({count/len(all_data)*100:.2f}%)")
    
    # Улучшенная предобработка данных
    def improved_preprocess_kdd_data(data):
        """
        Улучшенная предобработка данных KDD Cup.
        """
        # Создаем копию данных для обработки
        df = data.copy()
        
        # 1. Логарифмическое преобразование числовых признаков с большим разбросом
        for col in ['duration', 'src_bytes', 'dst_bytes', 'count', 'srv_count']:
            # Добавляем 1, чтобы избежать log(0)
            df[f'{col}_log'] = np.log1p(df[col])
        
        # 2. Нормализация числовых признаков
        numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
        numeric_features = [col for col in numeric_features 
                            if col not in ['label', 'is_anomaly'] 
                            and not col.startswith('protocol_type_') 
                            and not col.startswith('service_') 
                            and not col.startswith('flag_')]
        
        scaler = StandardScaler()
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        # 3. Создание дополнительных признаков для выявления определенных типов атак
        
        # 3.1. Отношение входящего и исходящего трафика (для выявления утечек данных)
        df['bytes_ratio'] = df['src_bytes'] / (df['dst_bytes'] + 1)  # +1 чтобы избежать деления на 0
        
        # 3.2. Признак частоты ошибок (для выявления сканирования и brute force)
        if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
            df['error_rate_sum'] = df['serror_rate'] + df['rerror_rate']
        
        # 3.3. Активность на нескольких сервисах (для выявления сканирования)
        if 'diff_srv_rate' in df.columns and 'dst_host_diff_srv_rate' in df.columns:
            df['service_diversity'] = (df['diff_srv_rate'] + df['dst_host_diff_srv_rate']) / 2
        
        # 3.4. Признак для выявления DoS-атак
        if 'count' in df.columns and 'srv_count' in df.columns:
            df['connection_rate'] = df['count'] / (df['srv_count'] + 1)
        
        # 4. Добавляем признаки порядка для отслеживания временных паттернов
        df['order'] = range(len(df))
        df['order_normalized'] = df['order'] / len(df)
        
        # 5. Добавляем временную метку для визуализации
        start_time = pd.Timestamp('2023-01-01')
        time_deltas = pd.to_timedelta(df['order'] * 10, unit='seconds')
        df['timestamp'] = start_time + time_deltas
        
        return df
    
    # Преобразование меток (нормальные vs аномальные)
    all_data['is_anomaly'] = all_data['label'].apply(lambda x: 0 if x == 'normal.' else 1)
    
    # Преобразование категориальных признаков
    categorical_features = ['protocol_type', 'service', 'flag']
    all_data_encoded = pd.get_dummies(all_data, columns=categorical_features)
    
    # Улучшенная предобработка
    preprocessed_data = improved_preprocess_kdd_data(all_data_encoded)
    
    # Разделение на обучающую и тестовую выборки с учетом стратификации
    train_data, test_data = train_test_split(preprocessed_data, test_size=0.3, random_state=42,
                                            stratify=preprocessed_data['is_anomaly'])
    
    print(f"Размер обучающей выборки: {train_data.shape}")
    print(f"Размер тестовой выборки: {test_data.shape}")
    
    return train_data, test_data

def generate_synthetic_data(n_samples=10000, anomaly_ratio=0.05, n_features=10, noise_level=0.1):
    """
    Генерация синтетических данных для тестирования системы обнаружения аномалий.
    """
    print(f"Генерация синтетических данных (n_samples={n_samples}, anomaly_ratio={anomaly_ratio})...")

    # Определяем количество нормальных и аномальных образцов
    n_normal = int(n_samples * (1 - anomaly_ratio))
    n_anomalies = n_samples - n_normal

    # Генерируем нормальные образцы из нормального распределения
    normal_data = np.random.randn(n_normal, n_features)

    # Генерируем аномальные образцы
    # 1. Точечные выбросы (значения далеко от среднего)
    n_point_anomalies = n_anomalies // 3
    point_anomalies = np.random.randn(n_point_anomalies, n_features) * 5 + 5

    # 2. Контекстуальные аномалии (нормальные значения в неправильном контексте)
    n_contextual_anomalies = n_anomalies // 3
    contextual_anomalies = np.random.randn(n_contextual_anomalies, n_features)

    # Вносим контекстуальные аномалии в некоторые признаки
    for i in range(n_contextual_anomalies):
        # Выбираем случайные признаки для аномалий
        anomaly_features = np.random.choice(n_features, size=2, replace=False)

        # Создаем необычные комбинации значений
        for j in anomaly_features:
            contextual_anomalies[i, j] = contextual_anomalies[i, j] * -3

    # 3. Коллективные аномалии (последовательности аномальных значений)
    n_collective_anomalies = n_anomalies - n_point_anomalies - n_contextual_anomalies
    collective_anomalies = np.random.randn(n_collective_anomalies, n_features) * 0.5

    # Добавляем коррелированные значения для создания коллективных аномалий
    correlation_matrix = np.random.randn(n_features, n_features)
    correlation_matrix = np.dot(correlation_matrix.T, correlation_matrix)
    collective_anomalies = np.dot(np.random.randn(n_collective_anomalies, n_features), correlation_matrix)

    # Объединяем все аномалии
    anomalies = np.vstack([point_anomalies, contextual_anomalies, collective_anomalies])

    # Создаем метки (0 - нормальные, 1 - аномалии)
    normal_labels = np.zeros(n_normal)
    anomaly_labels = np.ones(n_anomalies)

    # Объединяем данные и метки
    X = np.vstack([normal_data, anomalies])
    y = np.hstack([normal_labels, anomaly_labels])

    # Добавляем шум
    X += np.random.randn(*X.shape) * noise_level

    # Перемешиваем данные
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Создаем дополнительные признаки для сетевого трафика
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='1min')
    protocol = np.random.choice([6, 17, 1], size=n_samples, p=[0.8, 0.15, 0.05])
    src_port = np.random.randint(1024, 65535, size=n_samples)
    
    # Определяем количество портов из common_ports
    common_ports = [80, 443, 22, 23, 25, 53, 110, 143, 3389]
    common_count = int(n_samples * 0.75)
    
    # Заполняем порты
    dst_port = np.zeros(n_samples, dtype=int)
    dst_port[:common_count] = np.random.choice(common_ports, size=common_count)
    dst_port[common_count:] = np.random.randint(1024, 10000, size=n_samples - common_count)
    np.random.shuffle(dst_port)
    
    duration = np.random.exponential(scale=10, size=n_samples)
    bytes_data = np.random.exponential(scale=1000, size=n_samples)
    packets = np.random.geometric(p=0.1, size=n_samples)

    # Создаем датафрейм
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['is_anomaly'] = y
    df['timestamp'] = timestamps[indices]
    df['protocol'] = protocol
    df['src_port'] = src_port
    df['dst_port'] = dst_port
    df['duration'] = duration
    df['bytes'] = bytes_data
    df['packets'] = packets

    # Модифицируем аномалии с учетом типичных сетевых атак
    # DoS атаки
    dos_indices = df[df['is_anomaly'] == 1].sample(frac=0.3).index
    df.loc[dos_indices, 'packets'] *= 10
    df.loc[dos_indices, 'bytes'] *= 5
    
    # Сканирование портов
    scan_indices = df[df['is_anomaly'] == 1].sample(frac=0.3).index
    df.loc[scan_indices, 'duration'] = np.random.uniform(0.001, 0.1, size=len(scan_indices))
    df.loc[scan_indices, 'packets'] = np.random.randint(1, 3, size=len(scan_indices))
    
    # Аномалии портов
    port_indices = df[df['is_anomaly'] == 1].sample(frac=0.3).index
    unusual_ports = [6667, 31337, 4444, 9001, 1337, 8080]
    df.loc[port_indices, 'dst_port'] = np.random.choice(unusual_ports, size=len(port_indices))

    # Создание признаков, эмулирующих KDD Cup
    # Создаем service и flag как категориальные признаки
    services = ['http', 'ftp', 'smtp', 'ssh', 'dns', 'pop3', 'imap', 'ldap', 'ntp']
    flags = ['SF', 'S0', 'REJ', 'RSTO', 'RSTR', 'SH', 'S1', 'S2', 'S3', 'OTH']
    
    df['service'] = np.random.choice(services, size=n_samples)
    df['flag'] = np.random.choice(flags, size=n_samples, p=[0.7, 0.1, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01])
    
    # Аномалии часто имеют специфические флаги (S0, REJ, RSTO)
    anomaly_flags = ['S0', 'REJ', 'RSTO']
    df.loc[df['is_anomaly'] == 1, 'flag'] = np.random.choice(anomaly_flags, size=len(df[df['is_anomaly'] == 1]))
    
    # One-hot encoding категориальных признаков
    df = pd.get_dummies(df, columns=['service', 'flag'])
    
    # Добавляем признаки для DoS-атак, как в KDD Cup
    df['count'] = np.random.poisson(3, size=n_samples)  # Количество соединений к тому же хосту за 2 секунды
    df.loc[dos_indices, 'count'] *= 5  # Увеличиваем для DoS-атак
    
    df['srv_count'] = np.random.poisson(2, size=n_samples)  # Количество соединений к тому же сервису за 2 секунды
    df.loc[dos_indices, 'srv_count'] *= 5  # Увеличиваем для DoS-атак
    
    # Добавляем признаки ошибок, характерные для сканирования
    df['serror_rate'] = np.random.beta(1, 20, size=n_samples)  # Распределение с малым количеством ошибок
    df['rerror_rate'] = np.random.beta(1, 30, size=n_samples)
    
    # Увеличиваем количество ошибок для сканирования
    df.loc[scan_indices, 'serror_rate'] = np.random.beta(10, 1, size=len(scan_indices))
    df.loc[scan_indices, 'rerror_rate'] = np.random.beta(5, 1, size=len(scan_indices))
    
    # Нормализуем данные
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col not in ['is_anomaly', 'protocol', 'src_port', 'dst_port']]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Разделение на обучающую и тестовую выборки
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42,
                                          stratify=df['is_anomaly'])

    print(f"Размер обучающей выборки: {train_data.shape}")
    print(f"Размер тестовой выборки: {test_data.shape}")

    return train_data, test_data
