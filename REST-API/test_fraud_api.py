from fastapi.testclient import TestClient
from fraud_api import app  # Importiere deine FastAPI-App

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data and data["status"] == "healthy"
    assert "version" in data

def test_fraud_prediction():
    payload = {
        "transaction_header": {
            "store_id": "123e4567-e89b-12d3-a456-426614174000",
            "cash_desk": 1,
            "transaction_start": "2025-07-08T10:00:00",
            "transaction_end": "2025-07-08T10:05:00",
            "total_amount": 100.0,
            "payment_medium": "CASH"
        },
        "transaction_lines": [
            {
                "id": 1,
                "product_id": "123e4567-e89b-12d3-a456-426614174001",
                "timestamp": "2025-07-08T10:01:00",
                "pieces_or_weight": 1.0,
                "sales_price": 50.0,
                "was_voided": False,
                "camera_product_similar": True,
                "camera_certainty": 0.9
            }
        ]
    }
    response = client.post("/fraud-prediction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert isinstance(data["is_fraud"], bool)
    assert "fraud_proba" in data
    # Optional weitere Assertions, z.B. Typen oder Wertebereiche prüfen
