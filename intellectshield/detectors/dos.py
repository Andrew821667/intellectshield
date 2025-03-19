from intellectshield.detectors.base import BaseAnomalyDetector

class DoSDetector(BaseAnomalyDetector):
    """Заглушка для DoSDetector"""
    def preprocess_data(self, data, train=False):
        return data
    
    def train(self, data, **kwargs):
        return self
    
    def predict(self, data):
        return data
