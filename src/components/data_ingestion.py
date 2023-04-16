import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DataIngestionConfig


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Reading input data")
            data = pd.read_csv(self.data_ingestion_config.data_file_path)

            # Create the split folder if it doesn't exist
            if not os.path.exists(self.data_ingestion_config.split_folder_path):
                os.makedirs(self.data_ingestion_config.split_folder_path)
                logging.info(f"Created split folder: {self.data_ingestion_config.split_folder_path}")

            logging.info("Splitting data into train and test sets")
            train_data, test_data = train_test_split(
                data,
                test_size=self.data_ingestion_config.test_size,
                random_state=self.data_ingestion_config.random_state,
                stratify=data["Churn"]
            )

            logging.info(f"Saving train data to {self.data_ingestion_config.train_file_path}")
            train_data.to_csv(self.data_ingestion_config.train_file_path, index=False)

            logging.info(f"Saving test data to {self.data_ingestion_config.test_file_path}")
            test_data.to_csv(self.data_ingestion_config.test_file_path, index=False)

            return self.data_ingestion_config.train_file_path, self.data_ingestion_config.test_file_path

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())
