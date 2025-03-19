from intellectshield.detectors.base import BaseAnomalyDetector

class SequenceAnomalyDetector(BaseAnomalyDetector):
    """Заглушка для SequenceAnomalyDetector"""
    def preprocess_data(self, data, train=False):
        return data
    
    def train(self, data, **kwargs):
        return self
    
    def predict(self, data):
        return data
