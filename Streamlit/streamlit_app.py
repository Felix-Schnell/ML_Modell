import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.losses import MeanSquaredError  # type: ignore 

# Modelle laden
scaler = joblib.load("scaler.pkl")
model = joblib.load("xgboost_model.pkl")

autoencoder = load_model("autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())


import json
import io
import shap
import hashlib
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')



# === Seitentitel ===
st.set_page_config(page_title="Autoencoder Fraud Detection", layout="wide")
st.title("Fraud Detection mit Autoencoder")

# === Modell und Scaler laden ===
def load_autoencoder_model():
    model = load_model("autoencoder_model.h5", compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

def load_scaler():
    return joblib.load("scaler.pkl")

autoencoder = load_autoencoder_model()
scaler = load_scaler()

# === Dateiupload ===
st.header("1. Testdaten hochladen")
uploaded_file = st.file_uploader("Bitte lade eine vorbereitete Test-CSV hoch", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"{len(df)} Zeilen geladen")
    st.write(f"📁 Geladene Datei: {uploaded_file.name}, Erste ID: {df.iloc[0]['id'] if 'id' in df.columns else 'n/a'}")

    # Optional: Drop unwanted columns
    for col in ["label_fraud_bin", "damage"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Nur numerische Spalten
    X_raw = df.select_dtypes(include=['number']).copy()
    X_raw.fillna(X_raw.mean(), inplace=True)

    X_scaled = scaler.transform(X_raw)
    X_pred = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - X_pred) ** 2, axis=1)
    df["reconstruction_error"] = reconstruction_error

    reconstruction_diff = np.abs(X_scaled - X_pred)
    feature_names = X_raw.columns
    diff_df = pd.DataFrame(reconstruction_diff, columns=feature_names)
    df["most_suspicious_feature"] = diff_df.idxmax(axis=1)
    df["feature_error_value"] = diff_df.max(axis=1)

    second_largest_indices = np.argsort(-reconstruction_diff, axis=1)[:, 1]
    second_most_suspicious = [feature_names[i] for i in second_largest_indices]
    df["second_most_suspicious_feature"] = second_most_suspicious

    st.header("2. Anomalie-Erkennung")

    slider_max = min(1.0, float(reconstruction_error.max()))
    slider_default = min(float(np.percentile(reconstruction_error, 95)), slider_max)

    threshold = st.slider(
        "Fehlerschwellenwert",
        float(reconstruction_error.min()),
        slider_max,
        slider_default
    )

    df["is_fraud_pred"] = (reconstruction_error > threshold).astype(int)
    anomaly_count = (df["is_fraud_pred"] == 1).sum()

    st.subheader("Anomalien")
    st.write(f"🚨 Detektierte Anomalien: {anomaly_count} von {len(df)}")
    st.dataframe(df[df["is_fraud_pred"] == 1].head(10))


        # === Visualisierung: Top-2 auffällige Merkmale bei Fraud ===
    st.subheader("Top-2 auffällige Merkmale bei Fraud-Fällen")

    top1 = df[df["is_fraud_pred"] == 1]["most_suspicious_feature"]
    top2 = df[df["is_fraud_pred"] == 1]["second_most_suspicious_feature"]
    combined_features = pd.concat([top1, top2])
    combined_counts = combined_features.value_counts()

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(combined_counts.index, combined_counts.values)
    ax1.set_title("Häufigkeit der Top-2 auffälligen Merkmale bei Fraud")
    ax1.set_xlabel("Feature")
    ax1.set_ylabel("Anzahl")
    ax1.tick_params(axis='x', labelrotation=45)
    for label in ax1.get_xticklabels():
        label.set_horizontalalignment('right')

    plt.tight_layout()

    st.pyplot(fig1)



    # === Visualisierung & Tabelle: Fraud pro Monat auf Basis von 'transaction_start' ===
    if "transaction_start" in df.columns:
        try:
            df["transaction_start"] = pd.to_datetime(df["transaction_start"], format='mixed', errors='coerce')
            df = df.dropna(subset=["transaction_start"])  # Entferne fehlerhafte Einträge

            df["transaction_month"] = df["transaction_start"].dt.to_period("M").astype(str)
            frauds_per_month = df[df["is_fraud_pred"] == 1].groupby("transaction_month").size().reset_index(name="Fraud Count")

            st.subheader("📅 Anzahl Fraud-Fälle pro Monat")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(frauds_per_month["transaction_month"], frauds_per_month["Fraud Count"], marker='o')
            ax2.set_title("Monatliche Fraud-Fälle")
            ax2.set_xlabel("Monat")
            ax2.set_ylabel("Anzahl Fraud-Fälle")
            ax2.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig2)

            st.subheader("📊 Tabelle: Monatliche Fraud-Fälle")
            st.dataframe(frauds_per_month, use_container_width=True, height=300)

        except Exception as e:
            st.warning(f"Fehler beim Verarbeiten von 'transaction_start': {e}")

    # === Download-Button ===
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Ergebnisse als CSV herunterladen",
        data=csv,
        file_name="test_results_with_anomalies.csv",
        mime='text/csv'
    )

import json
import numpy as np
import pandas as pd
import shap
import joblib
from tensorflow.keras.models import Model
import streamlit as st
from tensorflow.keras.models import load_model
from datetime import datetime

st.header("2. Einzeltransaktion prüfen (JSON)")

example_json = '''
{
  "transaction_header": {
    "store_id": "123",
    "cash_desk": 2,
    "transaction_start": "2023-05-03T18:15:51",
    "transaction_end": "2023-05-03T18:18:39.342449",
    "total_amount": 20.00,
    "customer_feedback": 1,
    "payment_medium": "CREDIT_CARD"
  },
  "transaction_lines": [
    {
      "id": 1,
      "sales_price": 10,
      "was_voided": false,
      "camera_certainty": 0.9
    },
    {
      "id": 2,
      "sales_price": 10,
      "was_voided": false,
      "camera_certainty": 0.95
    }
  ]
}
'''

user_json = st.text_area("🔧 Transaktion im JSON-Format eingeben:", example_json, height=300)

if st.button("Einzeltransaktion analysieren"):
    try:
        data = json.loads(user_json)

        # === Modelle laden ===
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("xgboost_model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        autoencoder = load_model("autoencoder_model.h5", compile=False)

        # === Encoder extrahieren ===
        try:
            encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='dense_3').output)
        except:
            dense_layers = [l for l in autoencoder.layers if "dense" in l.name]
            encoder = Model(inputs=autoencoder.input, outputs=dense_layers[len(dense_layers)//2].output)

        # === Feature-Engineering ===
        def extract_features(data):
            header = data["transaction_header"]
            lines = data["transaction_lines"]

            # Zeitmerkmale
            start = datetime.fromisoformat(header["transaction_start"])
            end = datetime.fromisoformat(header["transaction_end"])
            duration_sec = (end - start).total_seconds()
            hour = start.hour
            weekday = start.weekday()

            sales_prices = [l["sales_price"] for l in lines]
            certainties = [l["camera_certainty"] for l in lines]
            voided = [l["was_voided"] for l in lines]

            # Dummy-Snacklogik (wenn keine Kategoriedaten vorliegen)
            snack_count = 0
            snack_share = 0.0

            # Optional kategoriebasierte Felder
            category_nunique_x = 1
            sold_by_weight_sum_x = 0
            age_restricted_sum_x = 0

            features = {
                "id": 0,  # Platzhalter
                "store_id": header.get("store_id", "unknown"),
                "cash_desk": header.get("cash_desk", 0),
                "transaction_start": header["transaction_start"],
                "transaction_end": header["transaction_end"],
                "total_amount": header.get("total_amount", 0),
                "n_lines": len(lines),
                "payment_medium": header.get("payment_medium", "OTHER"),
                "sales_price_sum_x": sum(sales_prices),
                "sales_price_mean_x": np.mean(sales_prices),
                "sales_price_max_x": np.max(sales_prices),
                "camera_certainty_mean_x": np.mean(certainties),
                "camera_certainty_min_x": np.min(certainties),
                "was_voided_sum_x": sum(voided),
                "category_nunique_x": category_nunique_x,
                "sold_by_weight_sum_x": sold_by_weight_sum_x,
                "age_restricted_sum_x": age_restricted_sum_x,
                "opening_date": 0,  # Dummy: Jahre seit Store-Eröffnung
                "has_cash_payment": int(header.get("payment_medium", "OTHER") == "CASH"),
                "average_price_per_article": header.get("total_amount", 0) / len(lines),
                "transaction_duration_seconds": duration_sec,
                "articles_per_minute": len(lines) / (duration_sec / 60) if duration_sec > 0 else 0,
                "voided_articles_ratio": sum(voided) / len(lines) if len(lines) > 0 else 0,
                "hour": hour,
                "weekday": weekday,
                "snack_count": snack_count,
                "snack_share": snack_share
            }

            return pd.DataFrame([features])


        input_df = extract_features(data)

        # === Featureliste trennen
        model_features = list(feature_names)
        base_features = [f for f in model_features if not f.startswith("ae_feat_")]
        ae_features = [f for f in model_features if f.startswith("ae_feat_")]

        # === Base-Features korrekt setzen
        X_base = pd.DataFrame(0, index=[0], columns=base_features)
        for col in input_df.columns:
            if col in X_base.columns:
                X_base[col] = input_df[col]

        # === AE-Features berechnen ===
        X_base_scaled = scaler.transform(X_base)
        ae_array = encoder.predict(X_base_scaled)

        # === Modellinput initialisieren ===
        X_input = pd.DataFrame(0, index=[0], columns=model_features)

        # === Base-Features einfügen ===
        for col in base_features:
            X_input.at[0, col] = X_base.at[0, col]

        # === AE-Features einfügen ===
        for i in range(len(ae_features)):
            col_name = f"ae_feat_{i}"
            if col_name in model_features:
                X_input.at[0, col_name] = ae_array[0][i]

        # === Vorhersage ===
        proba = model.predict_proba(X_input)[0][1]
        is_fraud = proba > 0.01

        result = {
            "is_fraud": bool(is_fraud),
            "fraud_proba": round(float(proba), 6),
            "threshold_used": 0.01,
            "estimated_damage": round(data["transaction_header"]["total_amount"] * 0.2229, 2) if is_fraud else None
        }

        if is_fraud:
            try:
                explainer = shap.Explainer(model, X_input)
                shap_values = explainer(X_input)
                top_idx = np.argmax(np.abs(shap_values.values[0]))
                result["explanation"] = {
                    "most_influential_feature": X_input.columns[top_idx],
                    "feature_value": float(X_input.iloc[0][top_idx]),
                    "shap_contribution": float(shap_values.values[0][top_idx])
                }
            except Exception as e:
                result["explanation"] = {"error": f"SHAP-Fehler: {e}"}

        st.subheader("📊 Ergebnis")
        st.json(result)

    except Exception as e:
        st.error(f"❌ Fehler beim Verarbeiten des JSON: {e}")


