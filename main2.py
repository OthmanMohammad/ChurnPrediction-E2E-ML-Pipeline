import sys
import os
import logging
import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.predict_pipeline import PredictPipeline

# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Test the predict pipeline
    test_predict_pipeline()

def test_predict_pipeline():
    # Create a sample input DataFrame
    sample_input = pd.DataFrame({
        "gender": ["Male"],
        "SeniorCitizen": [0],
        "Partner": ["No"],
        "Dependents": ["No"],
        "tenure": [2],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [63.58],
        "TotalCharges": ["148.15"]
    })

    # Create a PredictPipeline instance for the desired model, e.g., 'logistic_regression'
    predict_pipeline = PredictPipeline(model_name="xgboost")

    # Call the predict method with the sample input DataFrame
    predictions = predict_pipeline.predict(sample_input)

    # Print the predictions
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()

