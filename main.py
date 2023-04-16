import sys
import os
import logging
import pandas as pd
import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer

# Set logging level and format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Create DataIngestion object and initiate data ingestion
    data_ingestion_obj = DataIngestion()
    train_file_path, test_file_path = data_ingestion_obj.initiate_data_ingestion()

    # Create DataTransformation object and transform data
    data_transformation_obj = DataTransformation()
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    X_train = data_transformation_obj.transform(train_data)
    X_test = data_transformation_obj.transform(test_data)

    # Extract target variable from data
    y_train = X_train.pop('Churn').values
    y_test = X_test.pop('Churn').values

    # Create ModelTrainer object and initiate model training
    model_trainer_obj = ModelTrainer()
    model_results = model_trainer_obj.initiate_model_training(X_train, y_train, X_test, y_test)

    # Print model results
    for model_name, result in model_results.items():
        print(f"Results for {model_name}:")
        print(f"Best score: {result['best_score']}")
        print(f"Test accuracy: {result['test_accuracy']}")
        print(f"Best parameters: {result['best_model'].get_params()}")

if __name__ == "__main__":
    main()