# Employee Salary Classification API

This is a FastAPI application that provides an API for predicting employee salary classifications. It leverages a pre-trained machine learning model and encoders to make predictions on individual employee data or batch data provided via CSV.

---

## Features

- **Single Prediction:** Predict the salary classification for a single employee by providing their features.
- **Batch Prediction (JSON Response):** Upload a CSV file containing multiple employee records and receive a JSON response with predictions.
- **Batch Prediction (CSV Download):** Upload a CSV file and receive a CSV file back with the original data and appended predictions.
- **Health Check:** Endpoint to check the API status, model loading status, and versions of key libraries.
- **Model Information:** Retrieve details about the loaded machine learning model.
- **CORS Enabled:** Configured to allow requests from any origin (can be restricted in production).

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### 1\. Clone the repository

If this code is part of a larger repository, ensure you have cloned it. If it's a standalone project, create a directory and place the `main.py` (or similar, assuming the provided code is in `main.py`) file within it.

### 2\. Install Dependencies

Navigate to the project directory and install the required Python packages:

```bash
pip install "fastapi[all]" pandas scikit-learn joblib
```

### 3\. Place Model and Encoders

This application relies on a pre-trained model and pre-fitted encoders. Ensure you have the following files in your project directory:

- `best_model.pkl`: Your trained machine learning model (e.g., a scikit-learn model saved with `joblib.dump`).
- `encoders.pkl`: A Python pickle file containing a dictionary of fitted `LabelEncoder` objects for your categorical features.

If these files are not present, the application will issue warnings and attempt to create dummy encoders (not recommended for production use as predictions will be incorrect).

---

## Running the Application

To run the FastAPI application, use Uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- `main`: The name of your Python file (e.g., `main.py`).
- `app`: The FastAPI application instance within that file.
- `--host 0.0.0.0`: Makes the server accessible from external IPs.
- `--port 8000`: Specifies the port to run on.
- `--reload`: Enables auto-reloading on code changes (useful for development).

The API will be available at `http://0.0.0.0:8000`.

---

## API Endpoints

### 1\. Root Endpoint

- **URL:** `/`
- **Method:** `GET`
- **Description:** Basic endpoint to confirm the API is running.
- **Response:**
  ```json
  {
    "message": "Employee Salary Classification API",
    "status": "running"
  }
  ```

### 2\. Health Check

- **URL:** `/health`
- **Method:** `GET`
- **Description:** Provides information about the application's health, including model and encoder loading status.
- **Response Example:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_type": "LogisticRegression",
    "encoders_loaded": true,
    "available_encoders": [
      "workclass",
      "marital_status",
      "occupation",
      "relationship",
      "race",
      "gender",
      "native_country"
    ],
    "numpy_version": "1.23.5",
    "pandas_version": "1.5.3"
  }
  ```

### 3\. Single Prediction

- **URL:** `/predict`
- **Method:** `POST`
- **Description:** Predicts the salary classification for a single employee based on provided features.
- **Request Body (JSON):**
  ```json
  {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "educational_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "gender": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
  }
  ```
- **Response (JSON):**
  ```json
  {
    "prediction": "0", // or "1"
    "probability": [0.95, 0.05], // probabilities for each class
    "confidence": 0.95 // confidence of the predicted class
  }
  ```
  - **Note:** The `education` field is required in the input but will be dropped during preprocessing as per the model's training.

### 4\. Batch Prediction (JSON Response)

- **URL:** `/predict_batch`
- **Method:** `POST`
- **Description:** Upload a CSV file and get a JSON array of predictions. The CSV file is expected to have 15 columns in a specific order (matching `column_names_for_batch_input`).
- **Request Body:** `multipart/form-data` with a `File` field named `file`.
- **CSV File Format (example `input.csv`):**
  ```csv
  39,State-gov,77516,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K
  50,Self-emp-not-inc,83311,Bachelors,13,Married-civ-spouse,Exec-managerial,Husband,White,Male,0,0,13,United-States,<=50K
  38,Private,215646,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Male,0,0,40,United-States,<=50K
  ```
  (Note: The last column `income` is expected in the input CSV but will be ignored for prediction.)
- **Response (JSON):**
  ```json
  {
    "predictions": ["0", "0", "0"],
    "processed_count": 3
  }
  ```

### 5\. Batch Prediction (CSV Download)

- **URL:** `/predict_batch_download`
- **Method:** `POST`
- **Description:** Upload a CSV file and receive a CSV file back with the original data and an additional `predicted_salary` column.
- **Request Body:** `multipart/form-data` with a `File` field named `file`.
- **CSV File Format:** Same as for `/predict_batch`.
- **Response:** A downloadable CSV file (`salary_predictions.csv`) with the original data and an added `predicted_salary` column. Rows that were filtered out during preprocessing will have `predicted_salary` as `Filtered`.

### 6\. Model Information

- **URL:** `/model_info`
- **Method:** `GET`
- **Description:** Provides details about the loaded machine learning model, including its type, parameters, and expected features.
- **Response Example:**
  ```json
  {
    "model_type": "LogisticRegression",
    "model_params": "{'C': 1.0, 'class_weight': None, 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'l1_ratio': None, 'max_iter': 100, 'multi_class': 'auto', 'n_jobs': None, 'penalty': 'l2', 'random_state': None, 'solver': 'lbfgs', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}",
    "features_expected": [
      "workclass",
      "marital_status",
      "occupation",
      "relationship",
      "race",
      "gender",
      "native_country",
      "age",
      "fnlwgt",
      "educational_num",
      "capital_gain",
      "capital_loss",
      "hours_per_week"
    ],
    "numpy_version": "1.23.5",
    "pandas_version": "1.5.3",
    "scikit_learn_version": "1.2.2"
  }
  ```

---

## Preprocessing Details

The `preprocess_data` function applies several transformations to the input data before prediction:

- **Whitespace Stripping:** Removes leading/trailing whitespace from all string columns.
- **Missing Value Handling:** Replaces `?` with `Others` in categorical columns.
- **Data Filtering:** Filters out rows based on `workclass` (removes 'Without-pay', 'Never-worked') and numerical ranges for `age` (17-75) and `educational_num` (5-16).
- **Education Column Dropped:** The `education` column is dropped as it's typically redundant with `educational_num`.
- **Label Encoding:** Applies pre-fitted `LabelEncoder` to categorical features (`workclass`, `marital_status`, `occupation`, `relationship`, `race`, `gender`, `native_country`). Unseen categories are handled by mapping them to `Others` if available in the encoder's classes, or to the first class if not.

---

## Error Handling

- The API handles cases where the model or encoders are not found during startup, raising an `HTTPException` (or printing warnings/fallback).
- Invalid input data (e.g., non-CSV file for batch prediction) will result in `HTTP 400 Bad Request` errors.
- Internal server errors during prediction or preprocessing are caught and returned as `HTTP 400 Bad Request` or `HTTP 500 Internal Server Error` with detailed messages.
- A `503 Service Unavailable` error is returned if the model has not been loaded successfully.

---

## Development Notes

- **CORS:** For production deployments, it is highly recommended to replace `allow_origins=["*"]` with the specific URL(s) of your frontend application for security.
- **Model and Encoders:** Ensure `best_model.pkl` and `encoders.pkl` are trained on data that is representative of the data you expect to send for prediction.
- **Environment Variables:** For production, sensitive configurations or file paths should ideally be managed via environment variables.
