# Detectors Module for IntellectShield

This module contains various anomaly detectors for cybersecurity applications.

## Base Detector

BaseAnomalyDetector is an abstract class that defines the common interface for all anomaly detectors.

## SQL Injection Detector

SQLInjectionDetector is specialized for detecting SQL injection attempts in database queries. It uses a combination of approaches:

1. Regular expressions to identify common SQL injection patterns
2. Entropy analysis to detect unusual query structures
3. Heuristic analysis to find suspicious SQL constructs
4. Contextual analysis to consider previous queries

### Key Features

- Detects various types of SQL injections (UNION-based, Boolean-based, destructive, etc.)
- Provides detailed explanations of detected anomalies
- Adjustable threshold and component weights
- Classification by threat level

### Usage Example

from detectors.sql_injection_detector import SQLInjectionDetector
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'query': [
        "SELECT * FROM users WHERE id = 123",
        "SELECT * FROM users WHERE username = '' OR 1=1 --"
    ]
})

# Initialize and train the detector
detector = SQLInjectionDetector(threshold=0.5, verbose=True)
detector.fit(df.iloc[:1])  # Train on normal queries

# Detect anomalies
results = detector.detect_and_explain(df)
print(results)

## License

This code is part of the IntellectShield project.
