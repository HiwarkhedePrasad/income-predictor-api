from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import joblib
import numpy as np
from typing import List, Optional
import io
import os
from sklearn.preprocessing import LabelEncoder
import pickle

# Global variables for model and encoders
model = None
encoders = {}

async def load_model():
    global model, encoders
    try:
        # Load the trained model
        if os.path.exists("best_model.pkl"):
            model = joblib.load("best_model.pkl")
            print("Model loaded successfully!")
        else:
            # If model doesn't exist, we'll need to train it or provide a dummy model
            print("Warning: Model file not found. Please ensure best_model.pkl exists.")
            
        # Create label encoders for categorical features
        # Note: In production, you should save and load the actual encoders used during training
        categorical_features = [
            'workclass', 'education', 'marital_status', 'occupation', 
            'relationship', 'race', 'gender', 'native_country'
        ]
        
        for feature in categorical_features:
            encoders[feature] = LabelEncoder()
            
        print("Encoders initialized successfully!")
            
    except Exception as e:
        print(f"Error loading model: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await load_model()
    yield
    # Shutdown (if needed)
    pass

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Employee Salary Classification API",
    description="API for predicting employee salary classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class EmployeeFeatures(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    educational_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    gender: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

class PredictionResponse(BaseModel):
    prediction: str
    probability: Optional[List[float]] = None
    confidence: Optional[float] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[str]
    processed_count: int

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input data to match training data format"""
    try:
        # Handle missing values represented as '?'
        data = data.replace('?', 'Others')
        
        # Apply the same filtering as in training
        if 'workclass' in data.columns:
            data = data[~data['workclass'].isin(['Without-pay', 'Never-worked'])]
        
        # Apply age filtering (if age column exists)
        if 'age' in data.columns:
            data = data[(data['age'] <= 75) & (data['age'] >= 17)]
        
        # Apply educational-num filtering
        if 'educational_num' in data.columns:
            data = data[(data['educational_num'] <= 16) & (data['educational_num'] >= 5)]
        
        # Drop education column if it exists (as done in training)
        if 'education' in data.columns:
            data = data.drop(columns=['education'])
        
        # Apply label encoding to categorical features
        categorical_features = [
            'workclass', 'marital_status', 'occupation', 
            'relationship', 'race', 'gender', 'native_country'
        ]
        
        for feature in categorical_features:
            if feature in data.columns:
                # For new categories not seen during training, assign them to 'Others'
                unique_values = data[feature].unique()
                try:
                    data[feature] = encoders[feature].fit_transform(data[feature])
                except ValueError:
                    # Handle unseen categories
                    data[feature] = data[feature].apply(lambda x: 'Others' if x not in encoders[feature].classes_ else x)
                    data[feature] = encoders[feature].transform(data[feature])
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Employee Salary Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": len(encoders) > 0,
        "numpy_version": np.__version__
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(employee: EmployeeFeatures):
    """Predict salary classification for a single employee"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([employee.dict()])
        
        # Preprocess the data
        processed_data = preprocess_data(input_data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(processed_data)[0].tolist()
            confidence = max(probabilities)
        except:
            probabilities = None
            confidence = None
        
        return PredictionResponse(
            prediction=prediction,
            probability=probabilities,
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Predict salary classification for multiple employees from CSV file"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(processed_data).tolist()
        
        return BatchPredictionResponse(
            predictions=predictions,
            processed_count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict_batch_download")
async def predict_batch_with_download(file: UploadFile = File(...)):
    """Predict and return CSV file with predictions"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be CSV format")
    
    try:
        # Read the uploaded CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        original_df = df.copy()  # Keep original for output
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Make predictions
        predictions = model.predict(processed_data)
        
        # Add predictions to original dataframe
        original_df['predicted_salary'] = predictions
        
        # Save to temporary file
        output_file = "predictions_output.csv"
        original_df.to_csv(output_file, index=False)
        
        return FileResponse(
            output_file,
            media_type='text/csv',
            filename='salary_predictions.csv'
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"error": "Model not loaded"}
    
    try:
        model_info = {
            "model_type": type(model).__name__,
            "model_params": str(model.get_params()) if hasattr(model, 'get_params') else "N/A",
            "features_expected": list(encoders.keys()) + ['age', 'fnlwgt', 'educational_num', 'capital_gain', 'capital_loss', 'hours_per_week'],
            "numpy_version": np.__version__
        }
        return model_info
    except Exception as e:
        return {"error": f"Error getting model info: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)