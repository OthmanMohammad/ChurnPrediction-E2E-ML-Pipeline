import os
import sys
import pickle
import logging
from typing import Dict, Any
from src.exception import CustomException


def save_object(file_path: str, obj: Any):
    """
    Saves a Python object to a file using pickle serialization.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())


def load_object(file_path: str) -> Any:
    """
    Loads a Python object from a file using pickle deserialization.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(str(e), sys.exc_info())
