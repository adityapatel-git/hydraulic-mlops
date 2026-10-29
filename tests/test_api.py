from fastapi.testclient import TestClient

from api import main


class MockModel:
    def predict(self, data):
        return [100]


main.model = MockModel()

client = TestClient(main.app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "hydraulic-valve-classifier"
    assert data["alias"] == "champion"
    assert data["target"] == "valve_condition"
    assert data["features"] == 102


def test_predict_valid_input():
    features = {
        f"feature_{i}": float(i)
        for i in range(102)
    }

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["valve_condition"] == 100
    assert data["condition_description"] == "Optimal switching behavior"


def test_predict_invalid_feature_count():
    features = {
        f"feature_{i}": float(i)
        for i in range(101)
    }

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 400
    assert "Expected 102 features" in response.json()["detail"]