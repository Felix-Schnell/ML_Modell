import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import os
import traceback
import shap

# === Konfiguration ===
app = FastAPI()
API_VERSION = "1.5.0_final"
THRESHOLD = 0.4

# === Globale Variablen für Modelle und Daten ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler = None
xgb_model = None
autoencoder = None
encoder = None
df_products = None
df_stores = None

# === Startup-Event zum Laden der Modelle ===
@app.on_event("startup")
def load_all_models():
    """Lädt alle Modelle und benötigten Daten in den Speicher."""
    global scaler, xgb_model, autoencoder, encoder, df_products, df_stores
    
    print(">>> Lade Modelle und Stammdaten...")
    try:
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        xgb_model = joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl"))
        autoencoder = load_model(os.path.join(BASE_DIR, "autoencoder_model.h5"), compile=False)
        encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=-2).output)
        df_products = pd.read_csv(os.path.join(BASE_DIR, "products.csv"))
        df_stores = pd.read_csv(os.path.join(BASE_DIR, "stores.csv"))
    except FileNotFoundError as e:
        print(f"FATALER FEHLER: Datei nicht gefunden: {e}")
        print("Stelle sicher, dass alle .pkl, .h5 und .csv Dateien im selben Ordner wie die API liegen.")
        # Beendet die Anwendung, wenn eine Datei fehlt
        os._exit(1)
    print(">>> Alle Modelle und Daten erfolgreich geladen. API ist bereit. <<<")

# === Pydantic-Modelle für die API-Schnittstelle ===
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

# === Feature Engineering (Pragmatische Version) ===
# === Ersetze die komplette extract_features-Funktion mit dieser Version ===

def extract_features(data: TransactionRequest) -> pd.DataFrame:
    h = data.transaction_header
    
    # 1. Transaktionszeilen in einen DataFrame umwandeln
    if not data.transaction_lines:
        return pd.DataFrame() 
    
    lines_data = [line.model_dump() for line in data.transaction_lines]
    df_lines = pd.DataFrame(lines_data)
    # Wichtig für den Merge mit den Stammdaten
    df_lines['product_id'] = df_lines['product_id'].astype(str)

    # 2. Produktinformationen zu den Zeilen hinzufügen (wie in der echten Pipeline)
    # df_products wird beim Start der API geladen
    df_lines_merged = pd.merge(df_lines, df_products, left_on="product_id", right_on="id", suffixes=('', '_product'))

    # 3. Features aus den Zeilen korrekt aggregieren (keine Platzhalter mehr)
    sales_price_sum = df_lines_merged['sales_price'].sum()
    sales_price_mean = df_lines_merged['sales_price'].mean() if not df_lines_merged.empty else 0
    sales_price_max = df_lines_merged['sales_price'].max() if not df_lines_merged.empty else 0
    camera_certainty_mean = df_lines_merged['camera_certainty'].mean() if not df_lines_merged.empty else 0
    camera_certainty_min = df_lines_merged['camera_certainty'].min() if not df_lines_merged.empty else 0
    was_voided_sum = df_lines_merged['was_voided'].sum()
    category_nunique = df_lines_merged['category'].nunique()
    sold_by_weight_sum = df_lines_merged['sold_by_weight'].sum()
    age_restricted_sum = df_lines_merged['age_restricted'].sum()

    # 4. Filialinformationen holen (falls vorhanden)
    store_info_row = df_stores[df_stores['id'] == str(h.store_id)]
    opening_date = store_info_row.iloc[0].get("opening_date") if not store_info_row.empty else 0 # Fallback auf 0

    # 5. Finale Features berechnen
    duration_sec = (h.transaction_end - h.transaction_start).total_seconds()
    n_lines = len(data.transaction_lines)
    epsilon = 1e-6

    # 6. Finales Dictionary mit allen Features erstellen
    features = {
        "cash_desk": h.cash_desk,
        "total_amount": h.total_amount,
        "n_lines": n_lines,
        "sales_price_sum_x": sales_price_sum, # Benennung an die Pipeline anpassen
        "sales_price_mean_x": sales_price_mean,
        "sales_price_max_x": sales_price_max,
        "camera_certainty_mean_x": camera_certainty_mean,
        "camera_certainty_min_x": camera_certainty_min,
        "was_voided_sum_x": was_voided_sum,
        "category_nunique_x": category_nunique,
        "sold_by_weight_sum_x": sold_by_weight_sum,
        "age_restricted_sum_x": age_restricted_sum,
        "has_cash_payment": int(h.payment_medium == "CASH"),
        "average_price_per_article": h.total_amount / (n_lines + epsilon),
        "transaction_duration_seconds": duration_sec,
        "articles_per_minute": n_lines / ((duration_sec / 60) + epsilon),
        "voided_articles_ratio": was_voided_sum / (n_lines + epsilon),
        "hour": h.transaction_start.hour,
        "weekday": h.transaction_start.weekday(),
        "snack_count": 0,
        "snack_share": 0.0,
        # Weitere Features, die deine Pipeline eventuell erwartet, hier ergänzen
        # Diese Namen müssen mit denen in `scaler.feature_names_in_` übereinstimmen
        "sales_price_sum_y": sales_price_sum,
        "sales_price_mean_y": sales_price_mean,
        "sales_price_max_y": sales_price_max,
        "camera_certainty_mean_y": camera_certainty_mean,
        "camera_certainty_min_y": camera_certainty_min,
        "was_voided_sum_y": was_voided_sum,
        "category_nunique_y": category_nunique,
        "sold_by_weight_sum_y": sold_by_weight_sum,
        "age_restricted_sum_y": age_restricted_sum,
        "opening_date": opening_date,
    }
    
    return pd.DataFrame([features])

# === API-Routen ===
@app.get("/")
def health_check():
    return {"status": "healthy", "version": API_VERSION}


# === Ersetze die komplette predict_fraud-Funktion mit dieser finalen Version ===

@app.post("/fraud-prediction", response_model=PredictionResponse)
def predict_fraud(payload: TransactionRequest):
    try:
        if not payload.transaction_lines:
            raise HTTPException(status_code=400, detail="Transaction has no lines.")
        
        # 1. Features extrahieren
        input_df = extract_features(payload)

        # 2. Daten für Scaler/Autoencoder vorbereiten
        base_features_needed = scaler.feature_names_in_.tolist()
        X_base = pd.DataFrame(columns=base_features_needed)
        X_base = pd.concat([X_base, input_df], ignore_index=True, sort=False).fillna(0)
        X_base = X_base[base_features_needed]

        # 3. Skalieren und Autoencoder anwenden
        X_base_scaled = scaler.transform(X_base)
        ae_features_predicted = encoder.predict(X_base_scaled)
        
        # 4. Finale Matrix für XGBoost erstellen
        ae_feature_names = [f'ae_feat_{i}' for i in range(ae_features_predicted.shape[1])]
        X_input = pd.concat([
            pd.DataFrame(X_base_scaled, columns=base_features_needed),
            pd.DataFrame(ae_features_predicted, columns=ae_feature_names)
        ], axis=1)

        # 5. Spalten in die vom Modell erwartete Reihenfolge bringen
        xgb_feature_names = xgb_model.get_booster().feature_names
        X_input = X_input[xgb_feature_names]

        # 6. Vorhersage treffen
        proba = float(xgb_model.predict_proba(X_input)[0][1])
        is_fraud = proba > THRESHOLD

        # 7. Erklärung mit SHAP generieren
        explanation = None
        if is_fraud:
            try:
                # Initialisiere den SHAP-Explainer mit dem trainierten Modell
                explainer = shap.Explainer(xgb_model)
                # Berechne die SHAP-Werte für die aktuelle Transaktion
                shap_values = explainer(X_input)

                # Finde das Feature mit dem größten Einfluss (höchster absoluter SHAP-Wert)
                abs_shap_values = np.abs(shap_values.values[0])
                top_feature_idx = abs_shap_values.argmax()
                
                top_feature_name = X_input.columns[top_feature_idx]
                top_feature_value = X_input.iloc[0, top_feature_idx]
                
                # Erstelle eine dynamische Begründung
                final_reason = (f"'{top_feature_name}' hatte mit einem Wert von {top_feature_value:.2f} "
                                f"den stärksten Einfluss auf die Betrugswahrscheinlichkeit.")

                explanation = Explanation(
                    human_readable_reason=final_reason,
                    offending_products=[] 
                )
            except Exception as shap_error:
                # Fallback, falls SHAP einen Fehler wirft
                explanation = Explanation(
                    human_readable_reason=f"Fraud indicators detected. (SHAP explanation failed: {shap_error})",
                    offending_products=[]
                )

        return PredictionResponse(
            version=API_VERSION,
            is_fraud=is_fraud,
            fraud_proba=round(proba, 6),
            estimated_damage=round(payload.transaction_header.total_amount * 0.2229, 2) if is_fraud else None,
            explanation=explanation
        )
    except Exception as e:
        print("--- FEHLER BEI DER VORHERSAGE ---")
        traceback.print_exc()
        print("---------------------------------")
        raise HTTPException(status_code=500, detail=f"Ein interner Fehler ist aufgetreten: {str(e)}")