import os
from dataclasses import dataclass

# Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_file_path: str = os.path.join("artifacts", "data", "split", "train.csv")
    test_file_path: str = os.path.join("artifacts", "data", "split", "test.csv")
    data_file_path: str = os.path.join("artifacts", "data", "data.csv")
    split_folder_path: str = os.path.join("artifacts", "data", "split")
    test_size: float = 0.2
    random_state: int = 42

# Data Transformation Configuration
class DataTransformationConfig:
    target_variable: str = "Churn"
    categorical_columns: list = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    continuous_columns: list = ["tenure", "MonthlyCharges", "TotalCharges"]
    feature_names: list = [
    'SeniorCitizen','tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

    required_features: list = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "InternetService",
        "OnlineSecurity",
        "TechSupport"
    ]

# Model Configuration
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

models = {
    'logistic_regression': LogisticRegression(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'xgboost': XGBClassifier(),
    'svm': SVC()
}

# Model Parameter Grids
param_grids = {
    'logistic_regression': {
        'penalty': ['l1', 'l2']
    },
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'random_forest': {
        'n_estimators': [100, 200, 500],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10]
    },
    'xgboost': {
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 10],
        'n_estimators': [100, 200, 500]
    },
    'svm': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10],
        'gamma': [0.1, 1, 10]
    }
}




# Model Trainer Configuration
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "models")

