# План устранения проблем безопасности

## Файлы с потенциальными проблемами:
- ./detectors/enhanced_adaptive_detector_profiles.py
- ./detectors/enhanced_adaptive_detector_detection.py

## Рекомендуемые изменения:

1. Удалить хардкод секретов:
   - Переместить пароли, ключи и другие чувствительные данные в переменные окружения
   - Использовать файл .env (добавить его в .gitignore)
   - Добавить поддержку конфигурационных файлов с возможностью переопределения через переменные окружения

2. Пример замены хардкода:
```python
# Было:
api_key = "1234567890abcdef"

# Должно стать:
import os
from dotenv import load_dotenv

load_dotenv()  # загружаем переменные из .env файла
api_key = os.getenv("API_KEY")  # получаем значение из переменной окружения
```

3. Добавить файл .env.example с примером конфигурации без реальных значений:
```
# API ключи
API_KEY=your_api_key_here

# Параметры подключения к базе данных
DB_HOST=localhost
DB_PORT=5432
DB_USER=username
DB_PASSWORD=password
DB_NAME=database_name
```

4. Добавить в README.md информацию о настройке переменных окружения
