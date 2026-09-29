import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="MFS Fraud Detection API")

# Load artifacts
MODEL_PATH = "artifacts/model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"

model = None
preprocessor = None

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

class TransactionData(BaseModel):
    step: int
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

@app.post("/predict")
def predict(data: TransactionData):
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="Model or preprocessor not found in artifacts/ directory")

    try:
        # 1. Convert input to dataframe
        # model_dump is the standard for pydantic v2
        df = pd.DataFrame([data.model_dump()])
        
        # 2. Feature Engineering
        df["balance_change_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["balance_error_org"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
        df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["hour"] = df["step"] % 24
        
        # 3. Select columns
        columns_to_keep = [
            "step", "type", "amount", "balance_change_org", 
            "balance_error_org", "balance_change_dest", "hour"
        ]
        X = df[columns_to_keep]

        # 4. Data Transformation
        X_scaled = preprocessor.transform(X)

        # 5. Prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        result = "Fraud" if prediction == 1 else "Legitimate"

        return {
            "prediction": result,
            "fraud_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

import gradio as gr
from src.app.interface import demo

# Mount the Gradio UI directly onto the root path (/)
app = gr.mount_gradio_app(app, demo, path="/")
   