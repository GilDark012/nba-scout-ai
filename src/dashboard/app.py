"""
app.py

Main Streamlit dashboard for NBA Scout AI.
A non-technical scouting product for coaches, fans, and journalists.
Connects to the FastAPI backend for live data and model predictions.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import os
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from src.dashboard.components import shot_chart, zone_heatmap, recent_form_chart, stat_card
from src.dashboard.scouting_report import generate_report, generate_bullet_insights, generate_defensive_focus

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NBA Scout AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #ffffff; }
    .stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00d4aa; border-bottom: 3px solid #00d4aa; }
    div[data-testid="metric-container"] { background: #1a1a2e; border-radius: 10px; padding: 12px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2570/2570537.png", width=60)
    st.title("🏀 NBA Scout AI")
    st.caption("AI-powered scouting for everyone")
    st.divider()

    player_query = st.text_input("Search Player", placeholder="e.g. LeBron James", value="LeBron James")
    season = st.selectbox("Season", ["2023-24", "2022-23", "2021-22", "2020-21"], index=0)
    last_n_games = st.slider("Last N Games (form analysis)", 5, 30, 10)
    opponent = st.text_input("Opponent Team (optional)", placeholder="e.g. BOS")

    search_btn = st.button("🔍 Load Scouting Report", use_container_width=True, type="primary")
    st.divider()
    st.caption("Data: NBA Stats API · Model: XGBoost · Monitoring: Evidently AI")


# ─── State ──────────────────────────────────────────────────────────────────────
if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "shot_df" not in st.session_state:
    st.session_state.shot_df = None


# ─── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_report(player_name: str, season: str, last_n: int, opp: str) -> dict | None:
    """Fetch scouting report from FastAPI backend with caching."""
    try:
        params = {"player_name": player_name, "season": season,
                  "last_n_games": last_n, "opponent": opp or ""}
        r = requests.get(f"{BACKEND_URL}/player-report", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ Could not load report: {e}")
        return None


@st.cache_data(ttl=300, show_spinner=False)
def load_monitoring() -> dict | None:
    """Fetch monitoring summary from FastAPI backend."""
    try:
        r = requests.get(f"{BACKEND_URL}/monitoring", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ─── Trigger ────────────────────────────────────────────────────────────────────
if search_btn and player_query:
    with st.spinner(f"Loading scouting data for **{player_query}**..."):
        data = load_report(player_query, season, last_n_games, opponent)
        if data:
            st.session_state.report_data = data

# ─── Hero ───────────────────────────────────────────────────────────────────────
if not st.session_state.report_data:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px'>
        <h1 style='font-size:48px'>🏀 NBA Scout AI</h1>
        <p style='font-size:20px;color:#888'>
            AI-powered shot analysis for coaches, scouts, journalists, and fans.<br>
            Search a player on the left to generate a full scouting report.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Main Content ────────────────────────────────────────────────────────────────
data = st.session_state.report_data

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview", "🗺️ Shot Map", "📈 Projection", "📝 Scouting Report", "🔬 Monitoring"
])


# ─── TAB 1: Overview ────────────────────────────────────────────────────────────
with tab1:
    st.header(f"📋 {data['player_name']} — {data['season']}")
    st.caption(f"{data['total_attempts']} total shot attempts analyzed")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        stat_card("Overall FG%", f"{data['overall_fg_pct']:.0%}",
                  delta=f"Form: {data['form_trend'].capitalize()}")
    with col2:
        stat_card("2PT FG%", f"{data['two_pt_pct']:.0%}", color="#f0c040")
    with col3:
        stat_card("3PT FG%", f"{data['three_pt_pct']:.0%}", color="#7c83fd")
    with col4:
        trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}[data["form_trend"]]
        stat_card("Form Trend", f"{trend_icon} {data['form_trend'].capitalize()}", color="#ff4b4b")

    st.markdown("---")
    st.subheader("Recent Game Form")
    if data.get("recent_form"):
        fig_form = recent_form_chart(data["recent_form"])
        st.plotly_chart(fig_form, use_container_width=True)
    else:
        st.info("No recent form data available.")

    col_hot, col_cold = st.columns(2)
    with col_hot:
        st.subheader("🔥 Hot Zones")
        if data["hot_zones"]:
            for z in data["hot_zones"]:
                st.success(z)
        else:
            st.info("No hot zones identified.")
    with col_cold:
        st.subheader("❄️ Cold Zones")
        if data["cold_zones"]:
            for z in data["cold_zones"]:
                st.error(z)
        else:
            st.info("No cold zones identified.")


# ─── TAB 2: Shot Map ────────────────────────────────────────────────────────────
with tab2:
    st.header("🗺️ Shot Chart & Zone Breakdown")
    st.caption("Green circles = made shots · Red X = missed shots")

    # Build DataFrame from zone stats for fallback
    if data.get("zone_stats"):
        col_map, col_zones = st.columns([1.4, 1])
        with col_map:
            st.info("💡 To display the live shot scatter chart, data is fetched via the /player-report endpoint. Zone breakdown is shown right.")
            fig_zones = zone_heatmap(data["zone_stats"])
            st.plotly_chart(fig_zones, use_container_width=True)
        with col_zones:
            st.subheader("Zone Stats Table")
            zone_df = pd.DataFrame([
                {"Zone": z["zone"], "FG%": f"{z['fg_pct']:.0%}",
                 "Attempts": z["attempts"], "Made": z["made"], "Label": z["label"]}
                for z in data["zone_stats"]
            ])
            st.dataframe(zone_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No zone data available.")


# ─── TAB 3: Projection ──────────────────────────────────────────────────────────
with tab3:
    st.header("📈 Next-Game Projection")
    st.caption("Scenario-based estimate from rolling form and zone efficiency — not a guaranteed forecast.")

    st.markdown(f"""
    <div style='background:#1a1a2e;border-radius:12px;padding:24px;border-left:4px solid #00d4aa'>
        <h3 style='color:#00d4aa;margin-top:0'>Projected FG% Range</h3>
        <p style='font-size:36px;font-weight:800;color:white'>{data['overall_fg_pct']:.0%} ± 5%</p>
        <p style='color:#aaa'>{data['projection_note']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if opponent:
        st.info(f"🎯 Opponent filter applied: **{opponent.upper()}** — data may reflect reduced sample size.")
    else:
        st.caption("Tip: Select an opponent team in the sidebar for a matchup-specific projection.")

    form_trend = data["form_trend"]
    if form_trend == "improving":
        st.success("📈 Player is in form — expect efficiency at or above their average.")
    elif form_trend == "declining":
        st.warning("📉 Player is in a cold stretch — efficiency may be below historical average.")
    else:
        st.info("➡️ Player is shooting consistently — projection closely tracks historical average.")

    st.markdown("---")
    st.caption("⚠️ **Uncertainty Note:** Basketball shot prediction is inherently noisy. All projections are scenario-based estimates derived from historical patterns. A sample size below 50 shots is considered low confidence.")


# ─── TAB 4: Scouting Report ─────────────────────────────────────────────────────
with tab4:
    st.header("📝 Auto-Generated Scouting Report")
    st.caption("Plain-English scouting analysis · Powered by zone efficiency + form data")

    report_text = generate_report(data)
    st.markdown(report_text)

    st.markdown("---")
    col_ins, col_def = st.columns(2)
    with col_ins:
        st.subheader("💡 Key Insights")
        bullets = generate_bullet_insights(data)
        for b in bullets:
            st.markdown(f"- {b}")
    with col_def:
        st.subheader("🛡️ Defensive Recommendation")
        defense = generate_defensive_focus(data)
        st.markdown(f"""
        <div style='background:#1a1a2e;border-radius:8px;padding:16px;border-left:3px solid #ff4b4b'>
            <p style='color:#ddd;margin:0'>{defense}</p>
        </div>
        """, unsafe_allow_html=True)


# ─── TAB 5: Monitoring ─────────────────────────────────────────────────────────
with tab5:
    st.header("🔬 Model Health & Monitoring")
    st.caption("Powered by Evidently AI · Checks for data drift and performance shifts")

    mon = load_monitoring()
    if mon:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            status_color = "#ff4b4b" if mon["drift_detected"] else "#00d4aa"
            stat_card("Model Status", "⚠️ Drift" if mon["drift_detected"] else "✅ Healthy",
                      color=status_color)
        with col_m2:
            stat_card("Total Predictions", str(mon["total_predictions"]), color="#7c83fd")
        with col_m3:
            stat_card("Report", "Available ✅" if mon["report_available"] else "Not yet generated",
                      color="#f0c040")

        st.markdown("---")
        st.info(mon["summary"])

        if mon.get("report_available"):
            report_url = f"{BACKEND_URL}/monitoring/report"
            st.markdown(f"📄 [**View Full Evidently Report →**]({report_url})", unsafe_allow_html=False)
    else:
        st.warning("⚠️ Could not connect to monitoring backend. Make sure the API is running.")
        st.caption("Run `python src/monitoring/monitor.py` to generate the first report.")
