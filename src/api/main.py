"""FastAPI application for credit scoring"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict
import json
from pathlib import Path


app = FastAPI(
    title="Microfinance Credit Scoring API",
    description="API for predicting credit default risk",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = "models/credit_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "metrics/test_metrics.json"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load model or scaler: {e}")
    model = None
    scaler = None


class CreditApplication(BaseModel):
    amt_income_total: float = Field(..., description="Total income", gt=0)
    amt_credit: float = Field(..., description="Credit amount", gt=0)
    amt_annuity: float = Field(..., description="Loan annuity", gt=0)
    amt_goods_price: float = Field(..., description="Goods price", gt=0)
    age_years: float = Field(..., description="Age in years", ge=18, le=100)
    employment_years: float = Field(default=0, description="Employment years", ge=0)
    cnt_children: int = Field(default=0, description="Number of children", ge=0)
    cnt_fam_members: int = Field(default=1, description="Family size", ge=1)
    own_car: bool = Field(default=False, description="Owns car")
    own_realty: bool = Field(default=False, description="Owns realty")
    
    class Config:
        schema_extra = {
            "example": {
                "amt_income_total": 150000.0,
                "amt_credit": 500000.0,
                "amt_annuity": 25000.0,
                "amt_goods_price": 450000.0,
                "age_years": 35.0,
                "employment_years": 5.0,
                "cnt_children": 1,
                "cnt_fam_members": 3,
                "own_car": True,
                "own_realty": True
            }
        }


class PredictionResponse(BaseModel):
    default_probability: float
    risk_category: str
    recommendation: str
    features_used: Dict


@app.get("/")
def read_root():
    return {
        "message": "Microfinance Credit Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction",
            "/metrics": "Model metrics",
            "/docs": "API documentation"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if (model is not None and scaler is not None) else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.get("/metrics")
def get_metrics():
    try:
        with open(METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return {"metrics": metrics}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Metrics not found")


@app.post("/predict", response_model=PredictionResponse)
def predict_credit_risk(application: CreditApplication):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = {
            'AMT_INCOME_TOTAL': application.amt_income_total,
            'AMT_CREDIT': application.amt_credit,
            'AMT_ANNUITY': application.amt_annuity,
            'AMT_GOODS_PRICE': application.amt_goods_price,
            'AGE_YEARS': application.age_years,
            'EMPLOYMENT_YEARS': application.employment_years,
            'CNT_CHILDREN': application.cnt_children,
            'CNT_FAM_MEMBERS': application.cnt_fam_members,
            'FLAG_OWN_CAR': int(application.own_car),
            'FLAG_OWN_REALTY': int(application.own_realty),
            'CREDIT_INCOME_RATIO': application.amt_credit / (application.amt_income_total + 1),
            'ANNUITY_INCOME_RATIO': application.amt_annuity / (application.amt_income_total + 1),
            'GOODS_CREDIT_RATIO': application.amt_goods_price / (application.amt_credit + 1),
        }
        
        expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        
        if expected_features is not None:
            feature_df = pd.DataFrame([features])
            for feat in expected_features:
                if feat not in feature_df.columns:
                    feature_df[feat] = 0
            feature_df = feature_df[expected_features]
        else:
            feature_df = pd.DataFrame([features])
        
        features_scaled = scaler.transform(feature_df)
        probability = float(model.predict_proba(features_scaled)[0, 1])
        
        if probability < 0.3:
            risk_category = "Low"
            recommendation = "APPROVED - Low risk applicant"
        elif probability < 0.6:
            risk_category = "Medium"
            recommendation = "REVIEW REQUIRED - Medium risk applicant"
        else:
            risk_category = "High"
            recommendation = "REJECTED - High risk applicant"
        
        return PredictionResponse(
            default_probability=round(probability, 4),
            risk_category=risk_category,
            recommendation=recommendation,
            features_used=features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

