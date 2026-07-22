"""Tests for Yotuubef Studio FastAPI backend server."""

import pytest
from fastapi.testclient import TestClient
from src.api.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "app" in data


def test_list_projects():
    response = client.get("/api/projects")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_get_configuration():
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "yaml_content" in data
    assert "env_vars" in data


def test_preflight_check():
    response = client.get("/api/config/preflight")
    assert response.status_code == 200
    data = response.json()
    assert "all_passed" in data
    assert "checks" in data


def test_list_background_videos():
    response = client.get("/api/videos/backgrounds")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
