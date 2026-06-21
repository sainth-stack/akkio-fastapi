import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code in (200, 503)
    data = response.json()
    assert "api" in data
    assert "timestamp" in data
