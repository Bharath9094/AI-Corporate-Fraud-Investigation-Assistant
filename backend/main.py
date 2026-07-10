import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd

import ml_module
import ai_forensics

app = FastAPI(title="AI Corporate Fraud Investigation Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development ease, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Schemas
class TransactionInput(BaseModel):
    department: str
    role: str
    category: str
    vendor: str
    amount: float
    payment_method: str
    location: str
    hour: int
    is_weekend: int
    is_night: int
    emp_hist_mean: float
    emp_hist_std: float

class PredictRequest(BaseModel):
    transaction: TransactionInput

class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = []

# API Endpoints

@app.get("/api/dashboard-summary")
def get_dashboard_summary():
    """Retrieves aggregated metrics and chart coordinates for the main UI dashboard."""
    try:
        if not os.path.exists(ml_module.CSV_PATH):
            df = ml_module.generate_synthetic_data()
        else:
            df = pd.read_csv(ml_module.CSV_PATH)
            
        # Get predictions
        prob, anomaly, scores = ml_module.run_predictions_on_df(df)
        df["fraud_prob"] = prob
        df["is_anomaly"] = anomaly
        df["anomaly_score"] = scores
        
        # Calculate summary numbers
        total_tx = len(df)
        flagged_tx = int(df["is_fraud"].sum())
        clean_tx = total_tx - flagged_tx
        avg_risk_score = float(df["fraud_prob"].mean())
        total_amount = float(df["amount"].sum())
        
        # 1. Department Breakdown (for Charts)
        dept_group = df.groupby("department").agg(
            total_spend=("amount", "sum"),
            fraud_spend=("amount", lambda x: x[df.loc[x.index, "is_fraud"] == 1].sum()),
            tx_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum")
        ).reset_index()
        
        department_metrics = []
        for _, row in dept_group.iterrows():
            department_metrics.append({
                "name": row["department"],
                "totalSpend": round(float(row["total_spend"]), 2),
                "fraudSpend": round(float(row["fraud_spend"]), 2),
                "txCount": int(row["tx_count"]),
                "fraudCount": int(row["fraud_count"])
            })
            
        # 2. Hourly distribution of flagged vs clean
        hour_group = df.groupby(["hour", "is_fraud"]).size().unstack(fill_value=0).reset_index()
        if 0 not in hour_group:
            hour_group[0] = 0
        if 1 not in hour_group:
            hour_group[1] = 0
            
        hourly_metrics = []
        for _, row in hour_group.iterrows():
            hourly_metrics.append({
                "hour": int(row["hour"]),
                "clean": int(row[0]),
                "flagged": int(row[1])
            })
            
        # 3. Category distribution (for Charts)
        cat_group = df.groupby("category")["amount"].sum().reset_index()
        category_metrics = [
            {"name": r["category"], "value": round(float(r["amount"]), 2)}
            for _, r in cat_group.iterrows()
        ]
        
        # 4. Top highest risk flagged transactions
        flagged_list = df[df["is_fraud"] == 1].sort_values(by="fraud_prob", ascending=False).head(5)
        top_flagged = []
        for _, r in flagged_list.iterrows():
            top_flagged.append({
                "transaction_id": r["transaction_id"],
                "employee_name": r["employee_name"],
                "department": r["department"],
                "vendor": r["vendor"],
                "amount": float(r["amount"]),
                "fraud_prob": float(r["fraud_prob"]),
                "fraud_type": r["fraud_type"]
            })

        return {
            "total_transactions": total_tx,
            "flagged_transactions": flagged_tx,
            "clean_transactions": clean_tx,
            "avg_risk_score": avg_risk_score,
            "total_spend": total_amount,
            "departments": department_metrics,
            "hourly_distribution": hourly_metrics,
            "category_distribution": category_metrics,
            "top_flagged": top_flagged
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard error: {str(e)}")

@app.get("/api/transactions")
def get_transactions(status: Optional[str] = "all", limit: int = 100):
    """Fetches list of transactions with risk evaluation details."""
    try:
        if not os.path.exists(ml_module.CSV_PATH):
            df = ml_module.generate_synthetic_data()
        else:
            df = pd.read_csv(ml_module.CSV_PATH)
            
        # Apply predictions
        prob, anomaly, scores = ml_module.run_predictions_on_df(df)
        df["fraud_prob"] = prob
        df["is_anomaly"] = anomaly
        df["anomaly_score"] = scores
        
        # Filter by status
        if status == "flagged":
            df_filtered = df[df["is_fraud"] == 1]
        elif status == "clean":
            df_filtered = df[df["is_fraud"] == 0]
        else:
            df_filtered = df
            
        # Sort by transaction_id descending or amount descending
        df_filtered = df_filtered.sort_values(by="transaction_id", ascending=False).head(limit)
        
        transactions_list = []
        for _, r in df_filtered.iterrows():
            transactions_list.append({
                "transaction_id": r["transaction_id"],
                "employee_id": r["employee_id"],
                "employee_name": r["employee_name"],
                "department": r["department"],
                "role": r["role"],
                "category": r["category"],
                "vendor": r["vendor"],
                "amount": float(r["amount"]),
                "payment_method": r["payment_method"],
                "location": r["location"],
                "hour": int(r["hour"]),
                "is_weekend": int(r["is_weekend"]),
                "is_night": int(r["is_night"]),
                "is_fraud": int(r["is_fraud"]),
                "fraud_type": r["fraud_type"],
                "emp_hist_mean": float(r["emp_hist_mean"]),
                "emp_hist_std": float(r["emp_hist_std"]),
                "amount_deviation_score": float(r["amount_deviation_score"]),
                "fraud_prob": float(r["fraud_prob"]),
                "is_anomaly": int(r["is_anomaly"]),
                "anomaly_score": float(r["anomaly_score"])
            })
            
        return transactions_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing transactions: {str(e)}")

@app.post("/api/predict")
def predict_fraud(payload: PredictRequest):
    """Scores a single transaction in real time using the Random Forest and Isolation Forest models."""
    try:
        tx_dict = payload.transaction.model_dump()
        
        # Compute deviation score manually
        diff = tx_dict["amount"] - tx_dict["emp_hist_mean"]
        tx_dict["amount_deviation_score"] = round(diff / tx_dict["emp_hist_std"], 2) if tx_dict["emp_hist_std"] > 0 else 0.0
        
        # Build mini DataFrame
        temp_df = pd.DataFrame([tx_dict])
        prob, anomaly, scores = ml_module.run_predictions_on_df(temp_df)
        
        # Classify as fraud if probability > 0.5
        is_fraud = 1 if prob[0] > 0.5 else 0
        
        # Add temporary fraud description label
        fraud_type = "None"
        if is_fraud:
            if tx_dict["amount_deviation_score"] > 5:
                fraud_type = "Anomalous Spend Limit Exceeded"
            elif tx_dict["is_night"] and tx_dict["amount"] > 1000:
                fraud_type = "Suspicious Off-hours Purchase"
            elif tx_dict["location"] == "International (High Risk)":
                fraud_type = "High-Risk International Transfer"
            else:
                fraud_type = "ML Model Flagged Scenario"
                
        tx_dict["is_fraud"] = is_fraud
        tx_dict["fraud_type"] = fraud_type
        
        # Generate local explainable AI metrics
        explanations = ml_module.explain_prediction(tx_dict)
        
        return {
            "fraud_probability": float(prob[0]),
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
            "is_anomaly": int(anomaly[0]),
            "anomaly_score": float(scores[0]),
            "explanations": explanations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/train")
def train_model():
    """Triggers the machine learning pipeline to retrain Random Forest / Isolation Forest models."""
    try:
        # Generate new synthetic dataset randomly to simulate "new incoming data"
        ml_module.generate_synthetic_data(num_samples=1600)
        metrics = ml_module.train_ml_model()
        return {
            "status": "success",
            "message": "Model retrained successfully on new corporate audit logs.",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {str(e)}")

@app.get("/api/model-metrics")
def get_model_metrics():
    """Fetches accuracy, precision, recall, feature importances, and historical parameters of the active ML model."""
    try:
        if not os.path.exists(ml_module.METRICS_PATH):
            metrics = ml_module.train_ml_model()
        else:
            with open(ml_module.METRICS_PATH, "r") as f:
                metrics = json.load(f)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

@app.post("/api/chat")
def chat_forensics(payload: ChatRequest):
    """Processes natural language inputs and provides forensic analyses or audit reports."""
    try:
        response = ai_forensics.generate_forensic_response(payload.prompt, payload.history)
        return {
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Initial setup check
    if not os.path.exists(ml_module.METRICS_PATH):
        print("No trained models found. Running initial ML training...")
        ml_module.generate_synthetic_data()
        ml_module.train_ml_model()
        
    print("Starting FastAPI Uvicorn Server on http://localhost:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
