import os
import sys
import pickle
from flask import request, jsonify
from app import app
import pandas as pd

# Add the ChurnPrediction directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.predict_pipeline import PredictPipeline
from src.components.data_transformation import DataTransformationConfig

# Initialize the predict pipeline
predict_pipeline = PredictPipeline(model_name="svm")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data as a JSON object
    input_data = request.get_json()

    # Convert the input data to a DataFrame
    input_df = pd.DataFrame(input_data, index=[0])

    # Preprocess the input DataFrame
    preprocessed_data = predict_pipeline.preprocess_input(input_df)
    print("Preprocessed data:\n", preprocessed_data)  # Print preprocessed data

    # If your model supports predict_proba(), you can print predicted probabilities:
    if hasattr(predict_pipeline.model, "predict_proba"):
        probabilities = predict_pipeline.model.predict_proba(preprocessed_data)
        print("Predicted probabilities:\n", probabilities)  # Print predicted probabilities

    # Call the predict method with the preprocessed DataFrame
    predictions = predict_pipeline.model.predict(preprocessed_data)

    # Convert the predictions to a JSON object
    output = {"prediction": int(predictions[0])}

    # Return the prediction as a JSON object
    return jsonify(output)
