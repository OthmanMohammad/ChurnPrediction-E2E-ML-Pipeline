import pickle
from sklearn.preprocessing import StandardScaler
from src.components.data_transformation import DataTransformation

class Preprocessor:
    def __init__(self):
        self.data_transformation = DataTransformation()
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data[self.data_transformation.continuous_columns])

    def transform(self, data):
        data[self.data_transformation.continuous_columns] = self.scaler.transform(data[self.data_transformation.continuous_columns])
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
