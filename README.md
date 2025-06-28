# 🧪 Dataanalyse – Vorbereitung für Feature Engineering

Diese Datei dient der **explorativen Datenanalyse (EDA)** der Trainingsdaten und hilft dabei, **relevante Features** für das spätere Feature Engineering zu identifizieren.

---

## 📌 Zweck der Datei

- Untersuchung der **Trainingsdaten**
- Analyse der **Verteilung, Korrelation und Qualität** der Merkmale
- Verknüpfung und Auswertung der unterstützenden Tabellen:
  - `products.csv`
  - `stores.csv`
- Vorbereitung von Ideen für Feature Engineering (z. B. neue abgeleitete Variablen, Encoding-Möglichkeiten, Umgang mit fehlenden Werten)

---

## 📂 Eingabedateien

- `train.csv` – Hauptdatensatz mit Transaktionen/Fraud-Labels
- `products.csv` – Zusatzinformationen zu Produkten
- `stores.csv` – Zusatzinformationen zu Filialen oder Verkaufsorten

---

## 🔍 Inhalte der Analyse

- Überblick über Datenqualität und Nullwerte
- Verteilung von Zielvariablen (z. B. Fraud vs. Nicht-Fraud)
- Feature-Typen (kategorisch, numerisch)
- Statistische Kenngrößen und Visualisierungen
- Erste Hypothesen zur Bedeutung von Features

---

## 🛠️ Ausgabe/Ziel

Die Erkenntnisse aus dieser Datei dienen als **Grundlage für das Feature Engineering** in der Pipeline-Datei. Es erfolgt **keine Modellierung oder Preprocessing** – nur Analyse.

---

## 📝 Hinweise

- Läuft lokal mit Standard-Python-Installation (z. B. Jupyter)
- Verwendete Bibliotheken: `pandas`, `matplotlib`, `seaborn`, `numpy`
