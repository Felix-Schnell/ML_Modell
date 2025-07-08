from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime

# === Modelle & Hilfsdaten laden ===
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
feature_names = joblib.load("feature_names.pkl")
autoencoder = load_model("autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

# Encoder aus dem Autoencoder extrahieren
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=-2).output)

app = FastAPI()

API_VERSION = "1.2.3"
THRESHOLD = 0.4  # kann angepasst werden


# === Pydantic-Modelle ===
class TransactionHeader(BaseModel):
    store_id: uuid.UUID
    cash_desk: int
    transaction_start: datetime
    transaction_end: datetime
    total_amount: float
    payment_medium: str
    customer_feedback: Optional[int] = None

class TransactionLine(BaseModel):
    id: int
    product_id: uuid.UUID
    timestamp: datetime
    pieces_or_weight: float
    sales_price: float
    was_voided: bool
    camera_product_similar: bool
    camera_certainty: float

class TransactionRequest(BaseModel):
    transaction_header: TransactionHeader
    transaction_lines: List[TransactionLine]

class Explanation(BaseModel):
    human_readable_reason: str
    offending_products: List[str]

class PredictionResponse(BaseModel):
    version: str
    is_fraud: bool
    fraud_proba: Optional[float] = None
    estimated_damage: Optional[float] = None
    explanation: Optional[Explanation] = None


# === Feature Engineering ===
def extract_features(data: TransactionRequest) -> pd.DataFrame:
    h = data.transaction_header
    lines = data.transaction_lines

    duration_sec = (h.transaction_end - h.transaction_start).total_seconds()
    hour = h.transaction_start.hour
    weekday = h.transaction_start.weekday()
    prices = [l.sales_price for l in lines]
    certainties = [l.camera_certainty for l in lines]
    voided = [l.was_voided for l in lines]

    features = {
        "id": 0,
        "store_id": str(h.store_id),
        "cash_desk": h.cash_desk,
        "transaction_start": h.transaction_start.isoformat(),
        "transaction_end": h.transaction_end.isoformat(),
        "total_amount": h.total_amount,
        "n_lines": len(lines),
        "payment_medium": h.payment_medium,
        "sales_price_sum_x": sum(prices),
        "sales_price_mean_x": np.mean(prices),
        "sales_price_max_x": np.max(prices),
        "camera_certainty_mean_x": np.mean(certainties),
        "camera_certainty_min_x": np.min(certainties),
        "was_voided_sum_x": sum(voided),
        "category_nunique_x": 1,
        "sold_by_weight_sum_x": 0,
        "age_restricted_sum_x": 0,
        "opening_date": 0,
        "has_cash_payment": int(h.payment_medium == "CASH"),
        "average_price_per_article": h.total_amount / len(lines),
        "transaction_duration_seconds": duration_sec,
        "articles_per_minute": len(lines) / (duration_sec / 60) if duration_sec > 0 else 0,
        "voided_articles_ratio": sum(voided) / len(lines),
        "hour": hour,
        "weekday": weekday,
        "snack_count": 0,
        "snack_share": 0.0
    }

    return pd.DataFrame([features])


# === API Endpoints ===

@app.get("/")
def health_check():
    return {"status": "healthy", "version": API_VERSION}


@app.post("/fraud-prediction", response_model=PredictionResponse)
def predict_fraud(payload: TransactionRequest):
    try:
        input_df = extract_features(payload)

        base_features = [f for f in feature_names if not f.startswith("ae_feat_")]
        ae_features = [f for f in feature_names if f.startswith("ae_feat_")]

        # Base Features
        X_base = pd.DataFrame(0, index=[0], columns=base_features)
        for col in input_df.columns:
            if col in X_base.columns:
                X_base[col] = input_df[col]

        # Autoencoder-Features
        X_base_scaled = scaler.transform(X_base)
        ae_array = encoder.predict(X_base_scaled)

        X_input = pd.DataFrame(0, index=[0], columns=feature_names)
        X_input[base_features] = X_base
        for i, col in enumerate(ae_features):
            if col in X_input.columns:
                X_input[col] = ae_array[0][i]

        proba = float(xgb_model.predict_proba(X_input)[0][1])
        is_fraud = proba > THRESHOLD

        explanation = None
        if is_fraud:
            offending_products = ["PERSONAL CARE"]  # hier könntest du auch logik aus SHAP einbauen
            reason = []
            if np.mean([l.camera_certainty for l in payload.transaction_lines]) < 0.8:
                reason.append("Low camera certainty on products")
            if payload.transaction_header.total_amount > 25:
                reason.append("High transaction amount")
            explanation = Explanation(
                human_readable_reason="; ".join(reason),
                offending_products=offending_products
            )

        return PredictionResponse(
            version=API_VERSION,
            is_fraud=is_fraud,
            fraud_proba=round(proba, 6),
            estimated_damage=round(payload.transaction_header.total_amount * 0.2229, 2) if is_fraud else None,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage: {e}")
