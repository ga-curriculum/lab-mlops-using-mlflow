# MLOps Lab: Experiment Tracking, Model Versioning, and Deployment with MLflow (on AWS Jupyter Instance)

**Duration:** 90 minutes

**Authors:** Claudio Canales

-----

# Table of Contents

1.  [Learning Objectives](#learning-objectives)
2.  [Prerequisites](#prerequisites)
3.  [Lab Environment](#lab-environment)
4.  [Scenario](#scenario)
5.  [Lab Outline](#lab-outline)
    *   [Part 1: Project Setup and Data Preparation (15 minutes)](#part-1-project-setup-and-data-preparation-15-minutes)
        *   [1. Introduction (5 minutes)](#1-introduction-5-minutes)
        *   [2. Project Setup on Jupyter Instance (5 minutes)](#2-project-setup-on-jupyter-instance-5-minutes)
        *   [3. Data Loading and Exploration (5 minutes)](#3-data-loading-and-exploration-5-minutes)
    *   [Part 2: Model Training and Experiment Tracking with MLflow (35 minutes)](#part-2-model-training-and-experiment-tracking-with-mlflow-35-minutes)
        *   [1. Introduction to MLflow Tracking (5 minutes)](#1-introduction-to-mlflow-tracking-5-minutes)
        *   [2. Model Training Script (20 minutes)](#2-model-training-script-20-minutes)
        *   [3. Running and Tracking Experiments (10 minutes)](#3-running-and-tracking-experiments-10-minutes)
    *   [Part 3: Model Packaging and Deployment with Flask (35 minutes)](#part-3-model-packaging-and-deployment-with-flask-35-minutes)
        *   [1. Introduction to MLflow Models (5 minutes)](#1-introduction-to-mlflow-models-5-minutes)
        *   [2. Creating a Model Serving API with Flask (20 minutes)](#2-creating-a-model-serving-api-with-flask-20-minutes)
        *   [3. Running the Flask Server in the Background (5 minutes)](#3-running-the-flask-server-in-the-background-5-minutes)
        *   [4. Testing the API (5 minutes)](#4-testing-the-api-5-minutes)
6.  [VI. Conclusion and Where to Go Next (5 minutes)](#vi-conclusion-and-where-to-go-next-5-minutes)
    *   [A. Recap of Key Takeaways](#a-recap-of-key-takeaways)
    *   [B. Where to Go Next](#b-where-to-go-next)

## Learning Objectives

By the end of this lab, you will be able to:

-   ✅ Set up a basic ML project environment on an AWS Jupyter instance.
-   ✅ Use MLflow to track and log parameters, metrics, and artifacts during model training.
-   ✅ Package a trained machine learning model using MLflow's standard model format.
-   ✅ Create a simple REST API using Flask to serve predictions from an MLflow-packaged model.
-   ✅ Understand the basic workflow of deploying a model in a simplified manner.
-   ✅ Run and test a model-serving API locally.
-   ✅ Gain practical experience with core MLOps concepts, including experiment tracking, model versioning, and simplified deployment.
-   ✅ Relate the hands-on lab activities to the theoretical concepts learned in the "AI Model Deployment" and "MLOps Fundamentals" lessons.


**Prerequisites:**

-   Basic Python programming knowledge.
-   Familiarity with machine learning concepts (e.g., training, testing, evaluation).
-   An AWS account with access to a Jupyter instance

**Lab Environment:**

-   AWS Jupyter instance with Python 3.11.9.
-   We will be using a public dataset.
-   **Important:** We will run the Flask server in the background within the Jupyter instance for simplicity. In a real-world scenario, you would deploy it separately.

**Scenario:** We will build a simplified MLOps pipeline for a customer churn prediction model for a telecommunications company, putting into practice the concepts learned in the previous lessons.

## Lab Outline:

**(Total Time: 90 minutes)**

**Part 1: Project Setup and Data Preparation (15 minutes)**

1.  **Introduction (5 minutes):**
    -   **Welcome and Overview:** Briefly review the lab scenario (customer churn prediction) and objectives. Emphasize that we'll be focusing on core MLOps principles using MLflow in a simplified, hands-on manner, reinforcing the concepts from the previous lessons.
    -   **Introduce the Dataset:** We'll use the Telco Customer Churn dataset from Kaggle. Explain that it's publicly available and contains information about customer demographics, services, and churn status.
        -   **Dataset Features:** Briefly describe the key features: `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`, `Churn` (target variable).
    -   **Discuss Data Versioning (Conceptually):** Explain that in a real-world scenario, you would use a data versioning tool like DVC. Here, we'll keep it simple and focus on tracking code and model versions.
    -   **Explain the Importance of Data Preprocessing:** Briefly mention that data preprocessing is crucial but will be kept basic to focus on MLOps.

2.  **Project Setup on Jupyter Instance (5 minutes):**
    -   **Open a New Jupyter Notebook:** Participants launch a new notebook.
    -   **Install Libraries:** Provide the `pip install` command:
        ```bash
        !pip install mlflow scikit-learn pandas flask boto3
        ```
    -   **Create Directories:** Use `!mkdir -p data model` to create directories for data and models.
    -   **Set up Git (Optional but Recommended):** If comfortable with Git, initialize a repository: `!git init` and add a `.gitignore`.

3.  **Data Loading and Exploration (5 minutes):**
    
    -   **Download the Dataset:** Provide the command to download the dataset directly from the Iguazio S3 bucket:

        ```bash
        !wget --no-verbose https://iguazio-sample-data.s3.amazonaws.com/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv -O data/WA_Fn-UseC_-Telco-Customer-Churn.csv
        ```

        -   **Explanation of the change:**
            -   `!wget`: Executes the `wget` command in the Jupyter notebook cell.
            -   `--no-verbose`: Makes the download less verbose in the output.
            -   `https://iguazio-sample-data.s3.amazonaws.com/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv`: This is the direct URL to the dataset in the S3 bucket.
            -   `-O data/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Specifies the output path and filename to save the downloaded dataset.
    -   **Load with Pandas:** Use `pd.read_csv()` to load the dataset.
    -   **Brief Exploration:** Encourage using `data.head()`, `data.info()`, `data.describe()`, and `data.isnull().sum()`.

**Complete Code Example (Part 1):**

```python
# Cell 1: Install Libraries
# !pip install mlflow scikit-learn pandas flask boto3 # already installed

# Cell 2: Create directories
# !mkdir -p data model # already executed

# Cell 3: Download the Data from Iguazio S3
!wget --no-verbose https://iguazio-sample-data.s3.amazonaws.com/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv -O data/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Cell 4: Load and Explore Data
import pandas as pd

# Load data
data = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Get basic info about the data
print(data.info())

# Describe the data
print(data.describe())
```

**Part 2: Model Training and Experiment Tracking with MLflow (35 minutes)**

1.  **Introduction to MLflow Tracking (5 minutes):**
    -   **Explain MLflow Components:** Briefly introduce Tracking, Projects, Models, and Registry. Highlight that we'll focus on Tracking and Models.
    -   **MLflow Tracking Concepts:**
        -   **Run:** A single execution of model training code.
        -   **Parameters:** Input parameters (e.g., hyperparameters).
        -   **Metrics:** Evaluation metrics (e.g., accuracy, precision).
        -   **Artifacts:** Output files (e.g., models, plots).
    -   **Demonstrate `mlflow.start_run()`:** Show how to start and end an MLflow run using a context manager.
    -   **Demonstrate Logging:**
        -   `mlflow.log_param()`, `mlflow.log_params()`: Log parameters.
        -   `mlflow.log_metric()`, `mlflow.log_metrics()`: Log metrics.
        -   `mlflow.log_artifact()`: Log files.
    -   **Explain the MLflow UI:** Mention that the UI visualizes and compares runs (we'll explore it later).

2.  **Model Training Script (20 minutes):**
    -   Participants create `train.py` or work in notebook cells.
    -   **Provide the full code for `train.py`:**

```python
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Load and preprocess data
def load_and_preprocess_data(data_path):
    data = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    data = data.drop(['customerID'], axis=1)
    
    # Convert TotalCharges to numeric, setting errors='coerce' to turn invalid parsing into NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    
    # Fill NaN values in TotalCharges with the mean (or another strategy)
    data['TotalCharges'].fillna(data['TotalCharges'].mean(), inplace=True)
    
    # One-hot encode categorical features
    categorical_cols = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    
    # Encode the target variable to numerical format
    label_encoder = LabelEncoder()
    data['Churn'] = label_encoder.fit_transform(data['Churn'])
    
    return data

# Split data into features (X) and target (y)
def split_data(data, target_column):
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

if __name__ == "__main__":
    data_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Check if the data file exists
    if not os.path.isfile(data_path):
        print(f"Error: Data file not found at {data_path}")
        exit()

    # Load and preprocess the data
    try:
        data = load_and_preprocess_data(data_path)
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        exit()

    # Split the data
    try:
        X, y = split_data(data, "Churn")
    except Exception as e:
        print(f"Error splitting data: {e}")
        exit()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Start an MLflow run
    with mlflow.start_run():
        # Define hyperparameters
        params = {
            "penalty": "l2",
            "C": 0.1,
            "solver": "liblinear",
            "random_state": 42
        }

        # Log parameters
        mlflow.log_params(params)

        # Train the model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # Print out the model metrics
        print(f"Model accuracy: {accuracy}")
        print(f"Model precision: {precision}")
        print(f"Model recall: {recall}")
        print(f"Model f1-score: {f1}")
```

-   **Code Explanation:**
    -   Loads and preprocesses data (basic cleaning and one-hot encoding).
    -   Splits data into training and testing sets.
    -   Trains a `LogisticRegression` model.
    -   **MLflow Integration:**
        -   `with mlflow.start_run():` starts a new MLflow run.
        -   `mlflow.log_params(params)`: Logs hyperparameters.
        -   `mlflow.log_metric()`: Logs evaluation metrics.
        -   `mlflow.sklearn.log_model(model, "model")`: Logs the trained model.

3.  **Running and Tracking Experiments (10 minutes):**
    -   **Execution:** Execute `train.py` from the notebook using `!python train.py` or in notebook cells.
    -   **MLflow UI:** Open the MLflow UI:
        -   In a new cell, run: `!mlflow ui`.
        -   The UI will be available at `http://<instance-public-ip>:5000`.
            -   **Important:** Configure the security group of your instance to allow inbound traffic on port 5000. Emphasize security implications.
    -   **Exploration:** Show how to:
        -   View the list of runs.
        -   Compare runs based on parameters and metrics.
        -   Examine details of a run (parameters, metrics, artifacts).
        -   Click on the "model" artifact to see the logged model's information.
    -   **Experimentation:** Encourage participants to:
        -   Modify hyperparameters in `train.py` (e.g., change `C`).
        -   Re-run the training script.
        -   Observe how new runs are tracked in the UI.
        -   Compare model performance with different hyperparameters.

**Part 3: Model Packaging and Deployment with Flask (35 minutes)**

1.  **Introduction to MLflow Models (5 minutes):**
    -   **MLflow Models Format:** Explain that MLflow Models provide a standard way to package models.
    -   **`model` Directory Structure:** Show the structure of the `model` directory (created by `mlflow.sklearn.log_model()`):
        -   `MLmodel` (YAML): Describes the model's format, signature, and environment.
        -   `model.pkl`: The serialized model file.
        -   `conda.yaml` or `requirements.txt`: Specifies Python dependencies.
    -   **Model Signature:** Briefly explain the concept of a model signature (input and output schema).

2.  **Creating a Model Serving API with Flask (20 minutes):**
    -   Create `serve.py` or work in notebook cells.
    -   **Provide the full code for `serve.py`:**

```python
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd
import os

app = Flask(__name__)

# Get the latest logged model from MLflow
def get_latest_model():

    # Get the MLflow tracking URI from the environment variable
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if mlflow_tracking_uri is None:
        # Handle the case where the environment variable is not set
        print("Error: MLFLOW_TRACKING_URI environment variable not set.")
        return None  # Or raise an exception

    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    runs = mlflow.search_runs(filter_string="metrics.accuracy > 0.5", order_by=["metrics.accuracy DESC"], max_results=1) # Set your threshold

    if runs.empty:
        return None

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/model"
    return mlflow.pyfunc.load_model(model_uri)

model = get_latest_model()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert the incoming JSON data to a Pandas DataFrame
        data_df = pd.DataFrame(data)

        # Make predictions using the loaded model
        predictions = model.predict(data_df)

        # Convert predictions to a list before returning as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
```

-   **Code Explanation:**
    -   **Imports:** `flask`, `mlflow.pyfunc`, `pandas`.
    -   **`get_latest_model()` function:**
        -   Retrieves the MLflow tracking URI from environment variable.
        -   Sets the MLflow tracking URI.
        -   Searches for the most recent run that meets the accuracy criteria.
        -   Constructs the model URI using the `run_id`.
        -   Loads the model using `mlflow.pyfunc.load_model()`.
    -   **Flask App:** Creates a Flask app.
    -   **`/predict` Endpoint:**
        -   Defines `/predict` route that accepts POST requests with JSON data.
        -   Converts JSON data to a Pandas DataFrame.
        -   Calls `model.predict()` to make predictions.
        -   Returns predictions as a JSON response.

3.  **Running the Flask Server in the Background (5 minutes):**
    -   Run Flask in the background within Jupyter for simplicity.
    -   **Important:** Remind participants this is not a production-ready strategy.
    -   Use this code in a cell:

```python
    import os

    # Set the MLFLOW_TRACKING_URI environment variable
    os.environ["MLFLOW_TRACKING_URI"] = "http://<your-instance-public-ip>:5000"  # Use public IP or DNS name

    # Run the serve.py script in the background using nohup
    !nohup python serve.py > server.log 2>&1 &
```

  -   **Explanation:**
      -   `os.environ["MLFLOW_TRACKING_URI"] = ...`: Sets the environment variable to point to your MLflow UI. **Use the correct IP and port.**
      -   `nohup`: Ensures the process continues running after cell execution.
      -   `python serve.py`: Starts the Flask app.
      -   `> server.log 2>&1`: Redirects output and error to `server.log`.
      -   `&`: Runs in the background.

4.  **Testing the API (5 minutes):**
    -   Provide `curl` commands or a Python script for testing.
-   **Example `curl` command:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '[{"SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "PaperlessBilling": "Yes", "MonthlyCharges": 29.85, "TotalCharges": 29.85, "InternetService_DSL": 1, "InternetService_Fiber optic": 0, "InternetService_No": 0, "Contract_Month-to-month": 1, "Contract_One year": 0, "Contract_Two year": 0, "PaymentMethod_Bank transfer (automatic)": 0, "PaymentMethod_Credit card (automatic)": 0, "PaymentMethod_Electronic check": 1, "PaymentMethod_Mailed check": 0}]' http://<your-instance-public-ip>:5000/predict
    ```
    **Important:**
    -   Replace `<your-instance-public-ip>` with your instance's IP.
    -   Ensure security group allows inbound traffic on port 5000.
    -   **Explain Input Data:** The `-d` part sends data as a JSON array. Feature names must match columns used during training.
-   **Python Script Example:**
    ```python
    import requests
    import json

    url = 'http://<your-instance-public-ip>:5000/predict'  # Replace with IP and port
    headers = {'Content-Type': 'application/json'}
    data = [{"SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "PaperlessBilling": "Yes", "MonthlyCharges": 29.85, "TotalCharges": 29.85, "InternetService_DSL": 1, "InternetService_Fiber optic": 0, "InternetService_No": 0, "Contract_Month-to-month": 1, "Contract_One year": 0, "Contract_Two year": 0, "PaymentMethod_Bank transfer (automatic)": 0, "PaymentMethod_Credit card (automatic)": 0, "PaymentMethod_Electronic check": 1, "PaymentMethod_Mailed check": 0}]

    response = requests.post(url, headers=headers, data=json.dumps(data))

    print(response.json())
    ```
-   **Guide participants on how to:**
    -   Send a request.
    -   Interpret the JSON response (predictions).

**Discussion Points:**

-   **How does this lab demonstrate core MLOps principles?** (Automation, version control, experiment tracking, model management, simplified deployment). Relate this back to the principles discussed in the "MLOps Fundamentals" lesson.
-   **What are the limitations of this deployment?** (Not scalable, no robust monitoring, basic error handling).
-   **How could this be extended to a production-ready pipeline?** (Workflow orchestrator, containerization, cloud deployment, monitoring tools, CI/CD). Connect this to the deployment architectures and strategies from the "AI Model Deployment" lesson.
-   **Trade-offs between simple local deployment (Flask) vs. complex cloud deployment?** (Scalability, cost, management, latency).

---

## VI. Conclusion and Where to Go Next (5 minutes)

### A. Recap of Key Takeaways

-   **MLflow is a valuable tool for experiment tracking, model management, and packaging.** It simplifies key aspects of the ML lifecycle.
-   **Experiment tracking helps understand and improve models.** Logging parameters, metrics, and artifacts with MLflow enables systematic experimentation and comparison of different model versions.
-   **Model packaging with MLflow standardizes model deployment.** The MLflow Model format ensures that models can be easily deployed across different platforms.
-   **Flask provides a simple way to create a REST API for model serving.** This demonstrates the basic principles of exposing a model for inference.
-   **This lab provides a foundation for building more complex MLOps pipelines.** It demonstrates core MLOps principles in a practical, hands-on manner.

### B. Where to Go Next

-   **Explore more advanced MLflow features:** Investigate MLflow Projects for packaging code and dependencies, and MLflow Registry for managing the model lifecycle.
-   **Implement a more robust deployment:** Learn about containerization with Docker and orchestration with Kubernetes for scalable and reliable deployments. Consider cloud-based deployment options like SageMaker Hosting or Azure ML managed endpoints.
-   **Incorporate CI/CD:** Integrate your MLflow workflow into a CI/CD pipeline for automated testing and deployment.
-   **Set up model monitoring:** Implement monitoring to track model performance and data drift in a production setting. Use tools like Prometheus and Grafana or cloud-based monitoring services.
-   **Experiment with different model serving frameworks:** Explore TensorFlow Serving, TorchServe, or other frameworks for serving models.
-   **Dive deeper into data versioning:** Learn about tools like DVC to manage and version your datasets.
-   **Address the limitations:** Consider how you would address the limitations of the simplified deployment approach used in this lab in a real-world scenario. Think about scalability, fault tolerance, security, and monitoring.
-   **Multi-Environment Deployments:** As a logical next challenge, explore how to manage and deploy models across different environments (Development, Staging, Production) using MLflow and potentially a CI/CD pipeline.
-   **Advanced MLflow Features:** Learn about more advanced MLflow features such as Projects for reproducible runs, and the Model Registry for managing the model lifecycle and collaborating on models.

By continuing to learn and experiment, you can build upon the foundation established in this lab and develop increasingly sophisticated MLOps pipelines, ultimately enabling you to deploy and manage AI models effectively and deliver real business value.
