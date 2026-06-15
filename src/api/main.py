"""
main.py

FastAPI backend for NBA Scout AI.
Provides endpoints for player search, shot prediction,
scouting reports, and monitoring summaries.
"""

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from src.api.schemas import (
    PredictRequest, PredictResponse,
    PlayerSearchResult, PlayerReportResponse, MonitoringResponse
)
from src.api.services import search_players, predict_shot, build_player_report

load_dotenv()

app = FastAPI(
    title="NBA Scout AI",
    description="Production ML API for NBA shot efficiency prediction and player scouting.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """
    Health check endpoint for liveness probes and uptime monitoring.

    Returns:
        Status message.
    """
    return {"status": "healthy", "service": "NBA Scout AI", "version": "1.0.0"}


@app.get("/players", response_model=list[PlayerSearchResult], tags=["Players"])
def get_players(query: str = Query(..., min_length=2, description="Player name search query")):
    """
    Search for NBA players by partial name match.

    Args:
        query: Partial player name (min 2 characters).

    Returns:
        List of up to 20 matching players with IDs.
    """
    results = search_players(query)
    if not results:
        raise HTTPException(status_code=404, detail=f"No players found matching '{query}'")
    return results


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Predict whether a specific shot will be made.
    Returns probability, outcome label, confidence, and interpretation.

    Args:
        request: Shot features as PredictRequest.

    Returns:
        PredictResponse with probability and interpretation.
    """
    try:
        result = predict_shot(request.model_dump())
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/player-report", response_model=PlayerReportResponse, tags=["Scouting"])
def player_report(
    player_name: str = Query(..., description="Full NBA player name"),
    season: str = Query("2023-24", description="Season (e.g. 2023-24)"),
    last_n_games: int = Query(10, ge=3, le=50, description="Number of recent games"),
    opponent: str = Query(None, description="Opponent team abbreviation (optional)"),
):
    """
    Generate a full scouting report for a player.
    Includes zone stats, recent form, hot/cold zones, and a projection note.

    Returns:
        PlayerReportResponse with complete scouting data.
    """
    try:
        return build_player_report(player_name, season, last_n_games, opponent)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report error: {str(e)}")


@app.get("/monitoring", response_model=MonitoringResponse, tags=["Monitoring"])
def monitoring_summary():
    """
    Return a summary of model monitoring status.
    Reads the latest Evidently summary if available.

    Returns:
        MonitoringResponse with drift and performance status.
    """
    report_path = Path("reports/monitoring_report.html")
    summary_path = Path("reports/monitoring_summary.json")

    if summary_path.exists():
        import json
        data = json.loads(summary_path.read_text())
        return MonitoringResponse(**data)

    return MonitoringResponse(
        status="no_data",
        drift_detected=False,
        total_predictions=0,
        avg_confidence=0.0,
        report_available=report_path.exists(),
        summary="No monitoring data yet. Run monitor.py after collecting predictions.",
    )


@app.get("/monitoring/report", response_class=HTMLResponse, tags=["Monitoring"])
def monitoring_report_html():
    """
    Serve the full Evidently HTML monitoring report inline.

    Returns:
        HTML monitoring report.
    """
    report_path = Path("reports/monitoring_report.html")
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="No monitoring report generated yet. Run monitor.py first.")
    return HTMLResponse(content=report_path.read_text())
