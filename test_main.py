
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Student AI API is running!"
    )


def test_predict():
    response = client.post(
        "/predict",
        json={
            "study_hours": 5,
            "attendance": 75,
            "previous_mark": 65,
            "assignment_score": 80,
        },
    )

    assert response.status_code == 200
    assert isinstance(
        response.json()["predicted_mark"], (int, float)
    )


def test_invalid_input():
    response = client.post(
        "/predict",
        json={
            "study_hours": 5,
            "attendance": 150,
            "previous_mark": 65,
            "assignment_score": 80,
        },
    )

    assert response.status_code == 422
