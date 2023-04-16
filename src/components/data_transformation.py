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

         # Check if the expected columns are present in the data
        missing_columns = [col for col in self.categorical_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"The following expected columns are missing in the input data: {missing_columns}")

        # Convert TotalCharges to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

        # Drop 'customerID' as it's not a relevant feature for our analysis
        if 'customerID' in data.columns:
            data = data.drop('customerID', axis=1)
        
        # One-hot encode categorical features
        data = pd.get_dummies(data, columns=self.categorical_columns, drop_first=True)

        # Drop rows with NaN values
        data.dropna(inplace=True)

        return data