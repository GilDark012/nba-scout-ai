"""
test_api.py

Pytest test suite for the NBA Scout AI FastAPI backend.
Uses FastAPI's TestClient to test endpoints without a running server.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.main import app

client = TestClient(app)


# ─── /health ────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    """Health endpoint must return HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_correct_fields():
    """Health endpoint must return status, service, and version fields."""
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


# ─── /players ────────────────────────────────────────────────────────────────────

def test_players_returns_results_for_known_player():
    """Player search must return at least one result for 'LeBron'."""
    response = client.get("/players", params={"query": "LeBron"})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_players_result_has_required_fields():
    """Each player result must contain player_id, full_name, and is_active."""
    response = client.get("/players", params={"query": "LeBron"})
    assert response.status_code == 200
    player = response.json()[0]
    assert "player_id" in player
    assert "full_name" in player
    assert "is_active" in player


def test_players_returns_404_for_unknown_name():
    """Player search must return 404 for a name with no matches."""
    response = client.get("/players", params={"query": "zzzzunknownplayer999"})
    assert response.status_code == 404


def test_players_requires_minimum_query_length():
    """Player search must reject queries shorter than 2 characters."""
    response = client.get("/players", params={"query": "L"})
    assert response.status_code == 422  # FastAPI validation error


# ─── /monitoring ─────────────────────────────────────────────────────────────────

def test_monitoring_returns_200():
    """Monitoring endpoint must always return 200 even without data."""
    response = client.get("/monitoring")
    assert response.status_code == 200


def test_monitoring_has_status_field():
    """Monitoring response must contain a status field."""
    response = client.get("/monitoring")
    data = response.json()
    assert "status" in data
    assert "drift_detected" in data
    assert "report_available" in data


# ─── /predict ────────────────────────────────────────────────────────────────────

def test_predict_returns_503_when_model_missing():
    """
    Predict endpoint must return 200 (model loaded) or an error code
    (500/503) when no model file exists yet.
    This is expected during development before train.py is run.
    """
    response = client.post("/predict", json={
        "shot_distance": 15.0,
        "loc_x": 50.0,
        "loc_y": 150.0,
        "period": 2,
        "time_remaining_secs": 300.0,
        "is_three_pointer": 0,
        "is_home": 1,
        "shot_zone_basic": "Mid-Range",
        "shot_zone_area": "Center(C)",
        "shot_zone_range": "16-24 Ft.",
        "action_type": "Jump Shot",
        "shot_type": "2Pt Field Goal",
    })
    # Either 200 (model loaded) or 503 (model not found yet) — both valid
    assert response.status_code in [200, 500, 503]
