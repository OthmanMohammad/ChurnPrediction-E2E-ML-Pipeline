# Use the official Python image as the base image
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the entire project into the container
COPY . .

# Set the entry point for the container
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:5000", "ChurnPredictionAPI.app.views:app"]
