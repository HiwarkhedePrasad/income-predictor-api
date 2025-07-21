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
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Global variables for model and encoders
model = None
encoders = {}

async def load_model():
    global model, encoders
    try:
        # Load the trained model
        if os.path.exists("best_model.pkl"):
            model = joblib.load("best_model.pkl")
            print(f"Model loaded successfully! Type: {type(model).__name__}")
        else:
            print("Warning: Model file not found. Please ensure best_model.pkl exists.")
            return
            
        # Load pre-fitted encoders
        if os.path.exists("encoders.pkl"):
            with open("encoders.pkl", 'rb') as f:
                encoders = pickle.load(f)
            print(f"Encoders loaded successfully! Available encoders: {list(encoders.keys())}")
        else:
            print("Warning: Encoders file not found. Please ensure encoders.pkl exists.")
            # Create dummy encoders as fallback
            categorical_features = [
                'workclass', 'marital_status', 'occupation', 
                'relationship', 'race', 'gender', 'native_country'
            ]
            for feature in categorical_features:
                encoders[feature] = LabelEncoder()
            print("Created fallback encoders (not recommended for production)")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up application...")
    await load_model()
    yield
    # Shutdown
    print("Shutting down application...")

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
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values represented as '?'
        df = df.replace('?', 'Others')
        
        # Apply the same filtering as in training
        if 'workclass' in df.columns:
            df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
        
        # Apply age filtering (if age column exists)
        if 'age' in df.columns:
            df = df[(df['age'] <= 75) & (df['age'] >= 17)]
        
        # Apply educational-num filtering
        if 'educational_num' in df.columns:
            df = df[(df['educational_num'] <= 16) & (df['educational_num'] >= 5)]
        
        # Drop education column if it exists (as done in training)
        if 'education' in df.columns:
            df = df.drop(columns=['education'])
        
        # Apply label encoding to categorical features
        categorical_features = [
            'workclass', 'marital_status', 'occupation', 
            'relationship', 'race', 'gender', 'native_country'
        ]
        
        for feature in categorical_features:
            if feature in df.columns and feature in encoders:
                # Handle unseen categories by mapping them to a default value
                def safe_transform(x):
                    try:
                        if x in encoders[feature].classes_:
                            return encoders[feature].transform([x])[0]
                        else:
                            # Map unseen categories to 'Others' if it exists, otherwise first class
                            default_class = 'Others' if 'Others' in encoders[feature].classes_ else encoders[feature].classes_[0]
                            return encoders[feature].transform([default_class])[0]
                    except Exception as e:
                        print(f"Error encoding {feature}: {x}, using default value")
                        return 0  # Return default integer value
                
                df[feature] = df[feature].apply(safe_transform)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"Warning: Column {col} is still object type, attempting conversion")
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        print(f"Preprocessed data shape: {df.shape}")
        print(f"Preprocessed data columns: {list(df.columns)}")
        print(f"Data types: {df.dtypes.to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error preprocessing data: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Employee Salary Classification API", "status": "running"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model is not None else None,
        "encoders_loaded": len(encoders) > 0,
        "available_encoders": list(encoders.keys()),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(employee: EmployeeFeatures):
    """Predict salary classification for a single employee"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        input_data = pd.DataFrame([employee.model_dump()])
        print(f"Input data: {input_data}")
        
        # Preprocess the data
        processed_data = preprocess_data(input_data)
        print(f"Processed data shape: {processed_data.shape}")
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        print(f"Raw prediction: {prediction}")
        
        # Convert numpy types to Python types for JSON serialization
        if isinstance(prediction, (np.int64, np.int32)):
            prediction = int(prediction)
        elif isinstance(prediction, (np.float64, np.float32)):
            prediction = float(prediction)
        else:
            prediction = str(prediction)
        
        # Get prediction probabilities if available
        probabilities = None
        confidence = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0]
                probabilities = [float(p) for p in proba]  # Convert to Python floats
                confidence = float(max(probabilities))
        except Exception as e:
            print(f"Could not get probabilities: {e}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probabilities,
            confidence=confidence
        )
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print(f"Loaded CSV with shape: {df.shape}")
        
        # Preprocess the data
        processed_data = preprocess_data(df)
        
        # Make predictions
        raw_predictions = model.predict(processed_data)
        
        # Convert numpy types to Python types
        predictions = []
        for pred in raw_predictions:
            if isinstance(pred, (np.int64, np.int32)):
                predictions.append(int(pred))
            elif isinstance(pred, (np.float64, np.float32)):
                predictions.append(float(pred))
            else:
                predictions.append(str(pred))
        
        return BatchPredictionResponse(
            predictions=predictions,
            processed_count=len(predictions)
        )
        
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
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
        raw_predictions = model.predict(processed_data)
        
        # Convert predictions to appropriate format
        predictions = []
        for pred in raw_predictions:
            if isinstance(pred, (np.int64, np.int32)):
                predictions.append(int(pred))
            elif isinstance(pred, (np.float64, np.float32)):
                predictions.append(float(pred))
            else:
                predictions.append(str(pred))
        
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
        print(f"Batch prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
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
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scikit_learn_version": getattr(__import__('sklearn'), '__version__', 'Unknown')
        }
        return model_info
    except Exception as e:
        return {"error": f"Error getting model info: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    # When deploying to Railway, they will set the PORT environment variable
    # Use 0.0.0.0 to bind to all available network interfaces
    port = int(os.environ.get("PORT", 8000)) 
    print(f"DEBUG: Uvicorn attempting to run on host=0.0.0.0, port={port}") # Keep this debug print for Railway logs
    uvicorn.run(app, host="0.0.0.0", port=port)