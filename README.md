# Churn Prediction End-to-End Machine Learning Pipeline
## Introduction
This project aims to build an end-to-end machine learning pipeline for predicting customer churn using various technologies and services. Customer churn is a critical metric for businesses as it represents the rate at which customers stop doing business with a company. By predicting customer churn, companies can take proactive measures to retain customers and improve their services. This pipeline includes data preparation, model training, and evaluation, as well as API deployment. The project also demonstrates how to deploy the application using Docker and Amazon EC2, and how to set up continuous integration and deployment with GitHub Actions.


## Architecture Overview

The project's architecture, designed for modularity, scalability, and efficiency, employs an end-to-end machine learning pipeline consisting of several interconnected components. The pipeline seamlessly integrates [data ingestion](#data-preparation), [data transformation](#data-preparation), [model training](#model-training-and-evaluation), [model evaluation](#model-training-and-evaluation), and [deployment through a Flask API](#api-deployment).

[Data ingestion](#data-preparation) involves the acquisition and initial processing of raw data from multiple sources, ensuring compatibility with subsequent steps. The [data transformation module](#data-preparation) preprocesses and transforms the data, addressing issues like missing values, outliers, and feature scaling, making it suitable for model training. [Model training](#model-training-and-evaluation) explores various machine learning algorithms using the preprocessed data to optimize the solution. The [model evaluation component](#model-training-and-evaluation) selects the most appropriate model based on performance metrics such as accuracy, precision, recall, and F1 score. Lastly, the [Flask API](#api-deployment) deploys the trained model, providing users with an accessible interface for the model's predictive capabilities.

[Docker](#docker-deployment) is employed for containerization, facilitating portability and reproducibility throughout the project. Deployment on cloud platforms such as AWS is made seamless by leveraging [EC2 instances and configuring security groups](#aws-deployment-with-ec2-and-docker) to ensure proper access control. [Continuous integration and deployment](#continuous-integration-and-deployment-with-github-actions-ci/cd-pipeline) are achieved using GitHub Actions, contributing to the project's robustness and consistency across various environments.

The modular architecture allows for seamless additions or modifications to components without disrupting the entire pipeline's functionality. The incorporation of containerization, cloud deployment, and continuous integration and deployment streamlines the development and deployment process, ultimately enhancing the project's overall efficiency and reliability.


## Project Structure
The project structure is organized as follows:
```
Churn Prediction/
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── artifacts/
│   ├── data/
│   │   └── split/
│   │  
│   └── models/
│       ├── xgboost_params.pkl
│       ├── xgboost.pkl
│       └── preprocessor.pkl
├── churnpredictionapi/
│   ├── app/
│   │   ├── __init__.py
│   │   └── views.py
│   ├── config.py
│   ├── run.py 
│   └── wsgi.py
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── preprocessor.py
│   │   └── __init__.py
│   ├── config.py
│   ├── exception.py
│   ├── logger.py
│   ├── predict_pipeline.py
│   ├── utils.py
│   └── __init__.py
├── .gitignore
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

Data Preparation
-----------------
The data used in this project is a telecom dataset containing customer information and churn status. The dataset is first ingested using the `data_ingestion.py` module. The data is then preprocessed and transformed using the `preprocessor.py` and `data_transformation.py` modules. The preprocessing steps include handling missing values, encoding categorical variables, and scaling numerical features.

The `data_ingestion.py` module collects and processes raw data from various sources, while the `preprocessor.py` module handles missing values and encodes categorical variables using one-hot encoding. The `data_transformation.py` module scales numerical features using the `StandardScaler` from `scikit-learn`.

The preprocessing steps ensure that the data is suitable for training the machine learning models. They help to reduce bias and improve the model's accuracy and performance. By handling missing values and encoding categorical variables, we ensure that the model can learn from all available data. By scaling numerical features, we ensure that each feature has a similar influence on the model's predictions.


Model Training and Evaluation
-----------------------------

The model training and evaluation component of this project uses the `model_trainer.py` module to train multiple classifiers such as logistic regression, random forest, and XGBoost. Hyperparameter tuning is performed using a grid search and cross-validation, and the best model is selected based on evaluation metrics such as accuracy, precision, recall, and F1 score.

To train and evaluate the models, you can run the `main.py` script. This script loads the preprocessed data, creates a `ModelTrainer` instance, and calls the `train` and `evaluate` methods. The configuration for training and evaluation is defined in the `src/config.py` file, which contains parameters such as the models to train, hyperparameters, and evaluation metrics. If any parameter is not specified in the config file, default values are used.

To train and evaluate the models, simply run the `main.py` script using the following command:

```bash
python main.py
```

After training and evaluation, the trained models are saved in the `artifacts/models` directory. The best model for each classifier is saved along with its hyperparameters and evaluation metrics in the `artifacts/models` directory.


API Deployment
--------------
The API deployment component of this project is responsible for building and deploying the Flask API, which serves as the interface for making predictions. The API is built using the `ChurnPredictionAPI/app/views.py` module, which defines the API endpoints.

The API exposes a `/predict` endpoint that accepts input data as a JSON object and returns the predicted churn status. The input data is transformed using the same preprocessing steps used during training. The `Preprocessor` and `DataTransformationConfig` modules are used to ensure that the data is transformed consistently with the preprocessing steps used during training.

Once the input data is preprocessed and transformed, the trained model is loaded using the `load_model` function from `sklearn`. The model is then used to make predictions on the transformed input data, and the predicted churn status is returned as a JSON response.

To start the Flask API locally, run the `run.py` script located in the `ChurnPredictionAPI` directory. This will start the API on http://localhost:5000. You can then send a POST request to http://localhost:5000/predict with input data in JSON format to receive a prediction.

Here's an example of how to start the Flask API:

```bash
python ChurnPredictionAPI/run.py
```
Note that you need to be inside the `ChurnPredictionAPI` directory to run the `run.py` script.

Local Development and Testing
-----------------------------
To set up the project locally for development and testing purposes, follow these steps:

Clone the repository to your local machine:

```bash
git clone https://github.com/OthmanMohammad/ChurnPrediction-E2E-ML-Pipeline.git
```

Install the required dependencies:

```bash
cd ChurnPrediction-E2E-ML-Pipeline
pip install -r requirements.txt
```

Run the training script to train the machine learning models:

```bash
python main.py
```

Once the models are trained, start the Flask API to make predictions:

```bash
cd ChurnPredictionAPI
python run.py
```

With the API running, you can make predictions using tools like curl or Postman. Here's an example of how to make a prediction using curl:

```bash
curl -X POST \
  http://localhost:5000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "customerID": "1234-ABCD",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 50.5,
    "TotalCharges": 606.5
}'
```

Docker Deployment
-----------------
To deploy the application using Docker, follow these steps:

1. Ensure Docker is installed on your system. If not, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

2. Navigate to the project directory, where the provided `Dockerfile` is located. The `Dockerfile` has been carefully designed to create an optimized and lightweight environment. It leverages the `python:3.8-slim-buster` base image to minimize the size of the final image.

3. Build the Docker image by running the following command:

```bash
docker build -t churn_prediction:latest .
```

This command executes the instructions specified in the `Dockerfile`, which include:

- Setting up the working directory as `/app`.
- Copying the `requirements.txt` file into the container.
- Installing the required dependencies using `pip`, which are specified in the `requirements.txt` file.
- Copying the entire project into the container.
- Setting the container's entry point to start the Flask API using the `run.py` script.

4. Run the Docker container with the following command:

```bash
docker run -d -p 8000:5000 --name churn_prediction churn_prediction:latest
```

This command creates and starts a container named `churn_prediction` using the `churn_prediction:latest image`. The `-d` flag runs the container in detached mode, and the `-p` flag maps the host's port 5000 to the container's port 8000, where the Flask API is exposed.

Interact with the API using the exposed port 8000 on the host system. You can use tools like `curl` or Postman to send HTTP requests to the `/predict` endpoint and receive the churn prediction responses.

When finished, stop the container by running:

```bash
docker stop churn_prediction
```

And remove the container with:

```bash
docker rm churn_prediction
```

By following these advanced steps, you can effectively deploy the churn prediction application using Docker, ensuring a streamlined and efficient process that leverages best practices in containerization.


AWS Deployment with EC2 and Docker
----------------------------------
Deploying the churn prediction application on AWS requires a combination of EC2 instances, security group configurations, and Docker containerization. The following guide provides a thorough walkthrough of the process:

Access the AWS Management Console and sign in.

Proceed to the EC2 dashboard and initiate the creation of a new EC2 instance. Select an appropriate instance type, considering the balance between computational requirements and budget constraints.

During the instance configuration process, either create a new security group or modify an existing one to permit incoming traffic on port 8000. Add a custom TCP rule with the following specifications:

- **Type**: Custom TCP Rule
- **Protocol**: TCP
- **Port range**: 8000
- **Source**: Anywhere (0.0.0.0/0) or a specific IP address/range

Launch the EC2 instance and establish an SSH connection using your preferred SSH client and the provided key pair.

Refresh the package index and install the necessary packages to configure the Docker environment:

```zsh
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
```

Incorporate the Docker GPG key and repository:

```zsh
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
```

Proceed with the Docker CE installation:

```zsh
sudo apt-get update
sudo apt-get install docker-ce
```

Clone the churn prediction project repository to the EC2 instance:

```zsh
git clone https://github.com/OthmanMohammad/ChurnPrediction-E2E-ML-Pipeline.git
```

Change to the project directory and construct the Docker image:

```zsh
cd ChurnPrediction-E2E-ML-Pipeline
docker build -t churn_prediction:latest .
```

Execute the Docker container, making the API accessible on port 8000:

```zsh
docker run -d -p 8000:5000 --name churn_prediction churn_prediction:latest
```

Engage with the API using the EC2 instance's public IP address or domain name on port 8000. To send HTTP requests to the `/predict` endpoint and receive churn prediction responses, employ tools like curl or Postman.


Continuous Integration and Deployment with GitHub Actions (CI/CD pipeline)
--------------------------------------------------------------------------
In this churn prediction project, GitHub Actions is employed to establish a robust CI/CD pipeline, ensuring smooth integration and deployment processes. The pipeline configuration can be found in the `.github/workflows/ci-cd.yml` file, consisting of the following stages:

- **Checkout repository**: Utilizes the `actions/checkout@v2` action to access the project repository and retrieve its content.

- **Set up Python environment**: Leverages the `actions/setup-python@v2` action to configure the Python environment, specifying the desired Python version.

- **Install dependencies**: Executes the following commands to update the pip package manager and install the necessary dependencies from the `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

- **Run tests**: Executes test scripts to validate the application's functionality and ensure code reliability.

```
echo "Run your tests here"
```

- **Build and push Docker image**: Employs the `docker/setup-buildx-action@v1` action to set up Docker Buildx, followed by the `docker/login-action@v1` action to authenticate with Docker Hub. The `docker/build-push-action@v2` action is then used to build the Docker image using the project's `Dockerfile` and push it to Docker Hub, tagging it with the desired version.

