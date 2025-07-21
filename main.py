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
            # If model is critical, consider raising an exception here to fail startup
            # raise RuntimeError("best_model.pkl not found!")
            return
            
        # Load pre-fitted encoders
        if os.path.exists("encoders.pkl"):
            with open("encoders.pkl", 'rb') as f:
                encoders = pickle.load(f)
            print(f"Encoders loaded successfully! Available encoders: {list(encoders.keys())}")
        else:
            print("Warning: Encoders file not found. Please ensure encoders.pkl exists.")
            # Create dummy encoders as fallback (NOT recommended for production)
            categorical_features = [
                'workclass', 'marital_status', 'occupation', 
                'relationship', 'race', 'gender', 'native_country'
            ]
            for feature in categorical_features:
                encoders[feature] = LabelEncoder()
            print("Created fallback encoders (NOT recommended for production - predictions may be incorrect)")
            
    except Exception as e:
        print(f"Error loading model or encoders: {e}")
        import traceback
        traceback.print_exc()
        # Raise HTTP exception or a direct exception to stop app startup if critical
        raise HTTPException(status_code=500, detail=f"Failed to load model or encoders during startup: {e}")

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
    education: str # Will be dropped during preprocessing
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
        df = data.copy()
        
        # IMPORTANT: Strip whitespace from all string columns in incoming data
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()
        
        # Handle missing values represented as '?'
        df = df.replace('?', 'Others')
        
        # Apply the same filtering as in training
        if 'workclass' in df.columns:
            df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
        
        if 'age' in df.columns:
            df = df[(df['age'] <= 75) & (df['age'] >= 17)]
        
        if 'educational_num' in df.columns:
            df = df[(df['educational_num'] <= 16) & (df['educational_num'] >= 5)]
        
        if 'education' in df.columns:
            df = df.drop(columns=['education'])
            
        categorical_features = [
            'workclass', 'marital_status', 'occupation', 
            'relationship', 'race', 'gender', 'native_country'
        ]
        
        for feature in categorical_features:
            if feature in df.columns and feature in encoders:
                def safe_transform(x):
                    s_x = str(x) 
                    if s_x in encoders[feature].classes_:
                        return encoders[feature].transform([s_x])[0]
                    else:
                        if 'Others' in encoders[feature].classes_:
                            return encoders[feature].transform(['Others'])[0]
                        elif len(encoders[feature].classes_) > 0:
                            warnings.warn(f"Unseen value '{s_x}' for feature '{feature}'. 'Others' not in classes, using '{encoders[feature].classes_[0]}'.")
                            return encoders[feature].transform([encoders[feature].classes_[0]])[0]
                        else:
                            warnings.warn(f"Encoder for '{feature}' has no classes. Cannot transform '{s_x}'. Returning 0.")
                            return 0 
                
                df[feature] = df[feature].apply(safe_transform)
            elif feature in df.columns:
                print(f"Warning: Encoder for feature '{feature}' not found. Column might be untransformed or cause errors.")

        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"Warning: Column {col} is still object type after encoding, attempting final conversion.")
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
        input_data = pd.DataFrame([employee.model_dump()])
        print(f"Input data: {input_data}")
        
        processed_data = preprocess_data(input_data)
        
        if processed_data.empty:
            raise HTTPException(status_code=400, detail="Input data was entirely filtered out during preprocessing.")

        print(f"Processed data shape before prediction: {processed_data.shape}")
        
        prediction = model.predict(processed_data)[0]
        print(f"Raw prediction: {prediction}")
        
        if isinstance(prediction, (np.int64, np.int32)):
            prediction = int(prediction)
        elif isinstance(prediction, (np.float64, np.float32)):
            prediction = float(prediction)
        else:
            prediction = str(prediction)

        probabilities = None
        confidence = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(processed_data)[0]
                probabilities = [float(p) for p in proba]
                confidence = float(max(probabilities))
        except Exception as e:
            print(f"Could not get probabilities: {e}")
        
        return PredictionResponse(
            prediction=prediction,
            probability=probabilities,
            confidence=confidence
        )
        
    except HTTPException:
        raise
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
        contents = await file.read()
        column_names_for_batch_input = [
            'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None, names=column_names_for_batch_input)
        print(f"Loaded CSV with shape: {df.shape}")
        
        processed_data = preprocess_data(df)
        
        if processed_data.empty:
            return BatchPredictionResponse(predictions=[], processed_count=0)
            
        raw_predictions = model.predict(processed_data)
        
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
        
    except HTTPException:
        raise
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
        contents = await file.read()
        column_names_for_batch_input = [
            'age', 'workclass', 'fnlwgt', 'education', 'educational_num',
            'marital_status', 'occupation', 'relationship', 'race', 'gender',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')), header=None, names=column_names_for_batch_input)
        original_df = df.copy()
        
        processed_data = preprocess_data(df)
        
        if processed_data.empty:
            output_buffer = io.StringIO()
            pd.DataFrame(columns=list(original_df.columns) + ['predicted_salary']).to_csv(output_buffer, index=False)
            output_buffer.seek(0)
            return FileResponse(
                io.BytesIO(output_buffer.getvalue().encode('utf-8')),
                media_type='text/csv',
                filename='salary_predictions.csv'
            )

        raw_predictions = model.predict(processed_data)
        
        predictions = []
        for pred in raw_predictions:
            if isinstance(pred, (np.int64, np.int32)):
                predictions.append(int(pred))
            elif isinstance(pred, (np.float64, np.float32)):
                predictions.append(float(pred))
            else:
                predictions.append(str(pred))
        
        # Ensure predictions align with original rows after filtering.
        original_df['predicted_salary'] = pd.Series(predictions, index=processed_data.index)
        original_df.fillna({'predicted_salary': 'Filtered'}, inplace=True)
        
        output_buffer = io.StringIO()
        original_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        
        return FileResponse(
            io.BytesIO(output_buffer.getvalue().encode('utf-8')),
            media_type='text/csv',
            filename='salary_predictions.csv'
        )
        
    except HTTPException:
        raise
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