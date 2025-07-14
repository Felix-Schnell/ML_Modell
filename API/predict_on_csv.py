import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model, Model
import os

# === Konfiguration ===
INPUT_CSV_PATH = "df_model_ready_test.csv"
OUTPUT_CSV_PATH = "df_test_with_predictions.csv"
MODEL_DIR = "." # Aktueller Ordner
THRESHOLD = 0.02 # Derselbe Schwellenwert wie in der API

print(">>> Starte Batch-Vorhersage-Skript <<<")

# --- 1. Modelle laden ---
print("Lade Modelle...")
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    autoencoder = load_model(os.path.join(MODEL_DIR, "autoencoder_model.h5"), compile=False)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=-2).output)
    print("Modelle erfolgreich geladen.")
except FileNotFoundError as e:
    print(f"FATALER FEHLER: Modelldatei nicht gefunden: {e}")
    print("Stelle sicher, dass alle .pkl und .h5 Dateien im selben Ordner wie das Skript liegen.")
    exit()

# --- 2. Testdaten laden ---
print(f"Lese Testdaten aus '{INPUT_CSV_PATH}'...")
try:
    df_test = pd.read_csv(INPUT_CSV_PATH)
except FileNotFoundError:
    print(f"FATALER FEHLER: CSV-Datei nicht gefunden: '{INPUT_CSV_PATH}'")
    exit()

# --- 3. Daten für das Modell vorbereiten ---
# (Dieser Teil spiegelt exakt die Logik aus der API wider)

base_features_needed = scaler.feature_names_in_.tolist()

# Sicherstellen, dass alle benötigten Spalten vorhanden sind und mit 0 füllen, falls nicht
# Dies ist wichtig, falls die Test-CSV aus irgendeinem Grund Spalten nicht enthält
X_base = df_test.copy()
for col in base_features_needed:
    if col not in X_base.columns:
        X_base[col] = 0
# --- 3. Daten für das Modell vorbereiten ---
# ...
# Fehlende Werte mit den Mittelwerten aus dem Training auffüllen, die im Scaler gespeichert sind.
# Dies stellt sicher, dass die Testdaten exakt wie die Trainingsdaten behandelt werden.
imputation_values = pd.Series(scaler.mean_, index=scaler.feature_names_in_)
X_base[base_features_needed] = X_base[base_features_needed].fillna(imputation_values)

# Als letzte Sicherheitsmaßnahme: Falls eine Spalte im Training nie NaNs hatte und hier doch, fülle mit 0
X_base[base_features_needed] = X_base[base_features_needed].fillna(0) 

X_base = X_base[base_features_needed] # Reihenfolge sicherstellen

# --- 4. Vorhersage-Pipeline ausführen ---
print("Führe Modell-Pipeline aus...")

# a) Features skalieren
X_base_scaled = scaler.transform(X_base)

# b) Autoencoder-Features generieren
ae_features_predicted = encoder.predict(X_base_scaled)

# c) Basis-Features und Autoencoder-Features kombinieren
ae_feature_names = [f'ae_feat_{i}' for i in range(ae_features_predicted.shape[1])]
X_input = pd.concat([
    pd.DataFrame(X_base, columns=base_features_needed, index=X_base.index), # <--- KORRIGIERT
    pd.DataFrame(ae_features_predicted, columns=ae_feature_names, index=X_base.index)
], axis=1)

# d) Spalten an die exakte Reihenfolge des XGBoost-Modells anpassen
xgb_feature_names = xgb_model.get_booster().feature_names
for col in xgb_feature_names:
    if col not in X_input.columns:
        X_input[col] = 0 # Fügt Spalten hinzu, falls sie fehlen (z.B. durch get_dummies)
X_input = X_input[xgb_feature_names]

# --- 5. Finale Vorhersage treffen ---
print("Treffe finale Vorhersagen...")

# a) Wahrscheinlichkeiten vorhersagen
probabilities = xgb_model.predict_proba(X_input)[:, 1] # Nur die Wahrscheinlichkeit für "Fraud" (Klasse 1)

# b) Binäre Vorhersage (0 oder 1) basierend auf dem Schwellenwert
predictions = (probabilities > THRESHOLD).astype(int)

# --- 6. Ergebnis speichern ---
# Die neue Spalte zum ursprünglichen DataFrame hinzufügen
df_test['fraud_prediction'] = predictions

# Den DataFrame mit der neuen Spalte in eine neue CSV-Datei speichern
df_test.to_csv(OUTPUT_CSV_PATH, index=False)

print("-" * 30)
print(f"✅ Fertig! Ergebnisse wurden in '{OUTPUT_CSV_PATH}' gespeichert.")
print(f"Insgesamt wurden {len(df_test)} Zeilen verarbeitet.")
fraud_count = df_test['fraud_prediction'].sum()
print(f"Davon als Betrug (1) erkannt: {fraud_count}")
print(f"Davon als kein Betrug (0) erkannt: {len(df_test) - fraud_count}")
print(">>> Skript erfolgreich beendet. <<<")