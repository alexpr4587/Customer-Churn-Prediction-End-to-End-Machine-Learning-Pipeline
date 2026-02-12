from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import uvicorn
import os

import churn_pipeline
import __main__
__main__.ColumnNameCleaner = churn_pipeline.ColumnNameCleaner
__main__.NumericConverter = churn_pipeline.NumericConverter
__main__.CategoricalCleaner = churn_pipeline.CategoricalCleaner
__main__.TotalServicesCreator = churn_pipeline.TotalServicesCreator
__main__.FeatureDropper = churn_pipeline.FeatureDropper
__main__.LowServicesFeature = churn_pipeline.LowServicesFeature
__main__.CustomEncoder = churn_pipeline.CustomEncoder
__main__.CustomOneHotEncoder = churn_pipeline.CustomOneHotEncoder

# Import the wrapper
from churn_pipeline import ChurnPredictor

app = FastAPI(
    title="Churn Prediction API",
    description="Production API for Customer Churn Prediction",
    version="1.0.0"
)

# Initialize the predictor globally
MODEL_PATH = os.getenv("MODEL_PATH", "churn_prediction_pipeline.pkl")
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        if os.path.exists(MODEL_PATH):
            predictor = ChurnPredictor(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Define the input data schema matching the training data columns
class CustomerData(BaseModel):
    customerID: str = Field(..., example="TEST-1234")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="DSL")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="Yes")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="Yes")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=29.85)
    TotalCharges: str = Field(..., example="29.85")

class PredictionResponse(BaseModel):
    customerID: str
    churn_prediction: str
    churn_probability: float
    risk_category: str

@app.get("/health")
def health_check():
    return {"status": "active", "model_loaded": predictor is not None}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_churn(customers: List[CustomerData]):
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert Pydantic models to DataFrame
        # dict(c) works, but I need to ensure keys match what the pipeline expects
        data = [c.dict() for c in customers]
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = predictor.predict(df)
        probabilities = predictor.predict_proba(df)
        
        results = []
        for i, (customer_id, pred, prob) in enumerate(zip(df['customerID'], predictions, probabilities)):
            # Probability of Churn (class 1)
            churn_prob = prob[1]
            results.append({
                "customerID": customer_id,
                "churn_prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(churn_prob), 4),
                "risk_category": "High Risk" if churn_prob > 0.5 else "Low Risk"
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=2000, reload=True)
