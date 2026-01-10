"""Tests for the FastAPI hate-speech detection API."""


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy"]
    assert "message" in data


def test_model_info(client):
    """Test the model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_info" in data
    assert "config" in data
    assert "sklearn_loaded" in data["model_info"]
    assert "openai_available" in data["model_info"]


def test_predict_sklearn(client, sample_tweet):
    """Test sklearn prediction endpoint."""
    response = client.post("/predict", json=sample_tweet)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "text" in data
    assert data["text"] == sample_tweet["text"]


def test_predict_sklearn_batch(client, sample_batch_tweets):
    """Test sklearn batch prediction endpoint."""
    response = client.post("/predict/batch", json=sample_batch_tweets)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == len(sample_batch_tweets)
    for pred in data["predictions"]:
        assert "prediction" in pred
        assert "confidence" in pred


def test_chat_status(client):
    """Test the chat status endpoint."""
    response = client.get("/chat/status")
    assert response.status_code == 200
    data = response.json()
    assert "openai_available" in data
    assert "status" in data


def test_predict_empty_text(client):
    """Test prediction with empty text."""
    response = client.post("/predict", json={"text": ""})
    # Should either handle gracefully or return an error
    assert response.status_code in [200, 400, 422]


def test_predict_missing_text(client):
    """Test prediction with missing text field."""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error


def test_categories_in_prediction(client, sample_tweet):
    """Test that predictions return valid categories."""
    response = client.post("/predict", json=sample_tweet)
    assert response.status_code == 200
    data = response.json()

    valid_categories = [
        "Hate Speech",
        "Complaint",
        "Neutral",
        "Negative (not Hate Speech)",
        "Unknown"
    ]

    assert data["prediction"] in valid_categories
