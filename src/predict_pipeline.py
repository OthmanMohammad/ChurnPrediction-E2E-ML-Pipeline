import os
import sys
import pickle
import pandas as pd
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.config import DataTransformationConfig

class PredictPipeline:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.load_model()
        self.preprocessor = self.load_preprocessor()

    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "..", "artifacts", "models", f"{self.model_name}.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        self.feature_names = DataTransformationConfig.feature_names
        return model

    def load_preprocessor(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        preprocessor_path = os.path.join(current_dir, "..", "artifacts", "models", "preprocessor.pkl")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor

    def preprocess_input(self, input_data):
        data_transformation = DataTransformation()

        # Check if required features are provided
        self.check_required_features(input_data)

        # Handle default values
        input_data = self.handle_default_values(input_data)

        # Remove 'Churn' column if it exists
        if 'Churn' in input_data.columns:
            input_data = input_data.drop('Churn', axis=1)

        transformed_data = data_transformation.transform(input_data)
        preprocessed_data = self.preprocessor.transform(transformed_data)  # Use the loaded preprocessor
        preprocessed_data = preprocessed_data.reindex(columns=self.feature_names, fill_value=0)
        return preprocessed_data


    
    def check_required_features(self, data):
        data_transformation_config = DataTransformationConfig()
        missing_required_features = [f for f in data_transformation_config.required_features if f not in data.columns]
        if missing_required_features:
            raise ValueError(f"The following required features are missing: {', '.join(missing_required_features)}")
        return data
    
    def handle_default_values(self, data):
        data_transformation_config = DataTransformationConfig()
        default_values = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Electronic check',
        }
        
        for column, default_value in default_values.items():
            if column not in data.columns:
                data[column] = default_value
            else:
                data[column].fillna(default_value, inplace=True)
                    
        return data


    def predict(self, input_data):
        preprocessed_data = self.preprocess_input(input_data)
        predictions = self.model.predict(preprocessed_data)
        return predictions
