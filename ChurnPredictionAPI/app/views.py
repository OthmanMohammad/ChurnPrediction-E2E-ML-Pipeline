import os
import sys
import pickle
from flask import request, jsonify
from . import app
import pandas as pd

# Add the ChurnPrediction directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.predict_pipeline import PredictPipeline
from src.components.data_transformation import DataTransformationConfig

# Initialize the predict pipeline
predict_pipeline = PredictPipeline(model_name="logistic_regression")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data as a JSON object
    input_data = request.get_json()

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Call the predict method with the input DataFrame
    predictions = predict_pipeline.predict(input_df)

    # Convert the predictions to a JSON object
    output = {"prediction": int(predictions[0])}

    # Return the prediction as a JSON object
    return jsonify(output)
