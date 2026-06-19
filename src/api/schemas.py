"""
schemas.py

Pydantic models for request and response validation in the FastAPI backend.
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class PredictRequest(BaseModel):
    """Input schema for shot prediction endpoint."""
    shot_distance: float = Field(..., ge=0, le=94, description="Shot distance in feet")
    loc_x: float = Field(..., description="Shot X location on court")
    loc_y: float = Field(..., description="Shot Y location on court")
    period: int = Field(..., ge=1, le=7, description="Game period (1-4, OT=5+)")
    time_remaining_secs: float = Field(..., ge=0, description="Seconds remaining in period")
    is_three_pointer: int = Field(..., ge=0, le=1, description="1 if 3-pointer attempt")
    is_home: int = Field(..., ge=0, le=1, description="1 if home game")
    shot_zone_basic: str = Field(..., description="Shot zone category")
    shot_zone_area: str = Field(..., description="Shot zone area")
    shot_zone_range: str = Field(..., description="Shot zone range")
    action_type: str = Field(..., description="Shot action type (e.g. Jump Shot)")
    shot_type: str = Field(..., description="Shot type (2PT or 3PT)")
    shot_clock: float = 12.0        # avg shot clock value
    dribbles: float = 2.0           # avg dribbles before shot
    touch_time: float = 2.5         # avg touch time in seconds
    close_def_dist: float = 4.0     # avg closest defender distance (feet)
    rolling_fg_pct: Optional[float] = Field(0.45, description="Recent rolling FG%")
    rolling_attempts: Optional[float] = Field(10.0, description="Recent rolling attempts")
    zone_fg_pct: Optional[float] = Field(0.45, description="Historical zone FG%")


class PredictResponse(BaseModel):
    """Response schema for shot prediction."""
    shot_made_probability: float
    shot_missed_probability: float
    predicted_outcome: str
    confidence: str
    interpretation: str


class PlayerSearchResult(BaseModel):
    """Single player search result."""
    player_id: int
    full_name: str
    is_active: bool


class ZoneStats(BaseModel):
    """Zone-level shooting statistics."""
    zone: str
    attempts: int
    made: int
    fg_pct: float
    label: str  # e.g. "🔥 Hot Zone" or "❄️ Cold Zone"


class RecentForm(BaseModel):
    """Recent form summary across last N games."""
    game_date: str
    fg_pct: float
    attempts: int


class PlayerReportResponse(BaseModel):
    """Full scouting report response for a player."""
    player_name: str
    season: str
    total_attempts: int
    overall_fg_pct: float
    three_pt_pct: float
    two_pt_pct: float
    zone_stats: List[ZoneStats]
    recent_form: List[RecentForm]
    hot_zones: List[str]
    cold_zones: List[str]
    form_trend: str  # "improving", "declining", "stable"
    projection_note: str


class MonitoringResponse(BaseModel):
    """Monitoring summary response."""
    status: str
    drift_detected: bool
    total_predictions: int
    avg_confidence: float
    report_available: bool
    summary: str
