import requests
import json
import uuid
from datetime import datetime

# ==================== WICHTIG: DATEN ANPASSEN ====================
# Die IDs hier müssen ECHTE IDs aus deinen CSV-Dateien sein,
# sonst bekommst du wieder den "out-of-bounds"-Fehler von früher.
#
# 1. Öffne deine "stores.csv" und kopiere eine ID hierher.
# 2. Öffne deine "products.csv" und kopiere zwei IDs hierher.
# =================================================================

test_payload = {
    "transaction_header": {
        "store_id": "d3892a9e-8cbc-4237-be9c-211435dc92c0",  # <-- ANPASSEN
        "cash_desk": 1,
        "transaction_start": "2025-07-10T00:00:00",
        "transaction_end": "2025-07-10T00:01:15",
        "total_amount": 42.50,
        "payment_medium": "CARD",
        "customer_feedback": 5
    },
    "transaction_lines": [
        {
            "id": 101,
            "product_id": "9e7c9dba-db45-449c-a057-e80b3ef77426",  # <-- ANPASSEN
            "timestamp": "2025-07-10T00:00:10",
            "pieces_or_weight": 1.0,
            "sales_price": 25.00,
            "was_voided": False,
            "camera_product_similar": True,
            "camera_certainty": 0.98
        },
        {
            "id": 102,
            "product_id": "d542da73-8f5b-4ac7-840a-c67fa3eb2885",  # <-- ANPASSEN
            "timestamp": "2025-07-10T00:00:45",
            "pieces_or_weight": 1.0,
            "sales_price": 17.50,
            "was_voided": False,
            "camera_product_similar": True,
            "camera_certainty": 0.95
        }
    ]
}

# --- Ab hier nichts mehr ändern ---
api_url = "http://127.0.0.1:8000/fraud-prediction"
headers = {"Content-Type": "application/json"}

print("Sende Test-Anfrage an die API...")
try:
    response = requests.post(api_url, data=json.dumps(test_payload, default=str), headers=headers)
    print("\nAntwort vom Server (Status Code):", response.status_code)
    print("Antwort-Inhalt:")
    print(response.json())

except requests.exceptions.RequestException as e:
    print(f"\nFehler bei der API-Anfrage: {e}")