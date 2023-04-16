import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import DataTransformationConfig

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.categorical_columns = self.data_transformation_config.categorical_columns
        self.continuous_columns = self.data_transformation_config.continuous_columns

    def transform(self, data):
        # Replace "Yes" and "No" with 1 and 0 in the target variable only if it exists
        if self.data_transformation_config.target_variable in data.columns:
            data[self.data_transformation_config.target_variable] = data[self.data_transformation_config.target_variable].replace({'Yes': 1, 'No': 0})

        # Convert TotalCharges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        # Drop 'customerID' as it's not a relevant feature for our analysis
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)

        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=self.categorical_columns, drop_first=True)

        # Scale continuous features
        scaler = StandardScaler()
        data[self.continuous_columns] = scaler.fit_transform(data[self.continuous_columns])

        # Drop rows with NaN values
        data.dropna(inplace=True)

        return data