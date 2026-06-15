"""
components.py

Reusable Streamlit + Plotly UI components for the NBA Scout AI dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st


def draw_court(fig: go.Figure) -> go.Figure:
    """
    Draw a simplified NBA half-court on a Plotly figure.

    Args:
        fig: Existing Plotly figure to add court lines to.

    Returns:
        Figure with court lines added.
    """
    court_color = "#1a1a2e"
    line_color = "#e0e0e0"

    shapes = [
        # Outer boundary
        dict(type="rect", x0=-250, y0=-47.5, x1=250, y1=422.5,
             line=dict(color=line_color, width=2), fillcolor=court_color),
        # Paint (key)
        dict(type="rect", x0=-80, y0=-47.5, x1=80, y1=142.5,
             line=dict(color=line_color, width=2), fillcolor="rgba(30,60,114,0.3)"),
        # Free throw line
        dict(type="line", x0=-80, y0=142.5, x1=80, y1=142.5,
             line=dict(color=line_color, width=2)),
        # Backboard
        dict(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5,
             line=dict(color=line_color, width=3)),
    ]
    for shape in shapes:
        fig.add_shape(**shape)

    # Three-point arc
    theta = list(range(-65, 246))
    arc_x = [237.5 * __import__("math").cos(__import__("math").radians(t)) for t in theta]
    arc_y = [237.5 * __import__("math").sin(__import__("math").radians(t)) for t in theta]
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, mode="lines",
        line=dict(color=line_color, width=2),
        showlegend=False, hoverinfo="skip"
    ))
    return fig


def shot_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive shot chart scatter plot on a court background.

    Args:
        df: Shot DataFrame with LOC_X, LOC_Y, SHOT_MADE_FLAG.

    Returns:
        Plotly figure of the shot chart.
    """
    fig = go.Figure()
    fig = draw_court(fig)

    colors = df["SHOT_MADE_FLAG"].map({1: "#00d4aa", 0: "#ff4b4b"})
    symbols = df["SHOT_MADE_FLAG"].map({1: "circle", 0: "x"})

    for made, label, color, symbol in [(1, "Made", "#00d4aa", "circle"), (0, "Missed", "#ff4b4b", "x")]:
        mask = df["SHOT_MADE_FLAG"] == made
        sub = df[mask]
        hover_text = [
            f"{row.get('ACTION_TYPE', 'Shot')}<br>{row.get('SHOT_ZONE_BASIC', '')}<br>{row.get('SHOT_DISTANCE', 0):.0f} ft"
            for _, row in sub.iterrows()
        ] if len(sub) <= 2000 else [f"{label}"] * len(sub)

        fig.add_trace(go.Scatter(
            x=sub["LOC_X"], y=sub["LOC_Y"],
            mode="markers",
            marker=dict(color=color, symbol=symbol, size=5, opacity=0.65,
                        line=dict(width=0.5, color="white")),
            name=label,
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title="Shot Chart",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-60, 440], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(bgcolor="#1a1a2e", font=dict(color="white")),
        height=520,
    )
    return fig


def zone_heatmap(zone_stats: list) -> go.Figure:
    """
    Create a bar chart of shooting percentage by zone with color coding.

    Args:
        zone_stats: List of ZoneStats dicts (zone, fg_pct, attempts, label).

    Returns:
        Plotly bar chart.
    """
    df = pd.DataFrame([z if isinstance(z, dict) else z.dict() for z in zone_stats])
    df["color"] = df["label"].apply(
        lambda l: "#00d4aa" if "Hot" in l else ("#ff4b4b" if "Cold" in l else "#f0c040")
    )
    df = df.sort_values("fg_pct", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["fg_pct"] * 100,
        y=df["zone"],
        orientation="h",
        marker_color=df["color"],
        text=[f"{p:.1f}% ({a} att)" for p, a in zip(df["fg_pct"] * 100, df["attempts"])],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>FG%: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Zone Shooting Efficiency",
        xaxis_title="FG%",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        xaxis=dict(range=[0, 80], gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        height=420,
    )
    return fig


def recent_form_chart(recent_form: list) -> go.Figure:
    """
    Create a line chart showing recent game-by-game FG% trend.

    Args:
        recent_form: List of RecentForm dicts (game_date, fg_pct, attempts).

    Returns:
        Plotly line chart.
    """
    df = pd.DataFrame([f if isinstance(f, dict) else f.dict() for f in recent_form])
    avg = df["fg_pct"].mean()

    fig = go.Figure()
    fig.add_hline(y=avg, line_dash="dash", line_color="#888",
                  annotation_text=f"Avg: {avg:.0%}", annotation_font_color="white")
    fig.add_trace(go.Scatter(
        x=df["game_date"], y=df["fg_pct"] * 100,
        mode="lines+markers",
        line=dict(color="#00d4aa", width=2),
        marker=dict(size=8, color=df["fg_pct"].apply(
            lambda p: "#00d4aa" if p >= avg else "#ff4b4b"
        )),
        text=[f"{a} attempts" for a in df["attempts"]],
        hovertemplate="<b>%{x}</b><br>FG%: %{y:.1f}%<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Recent Form — FG% per Game",
        xaxis_title="Game Date",
        yaxis_title="FG%",
        yaxis=dict(range=[0, 80], gridcolor="#333"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=340,
    )
    return fig


def stat_card(label: str, value: str, delta: str = None, color: str = "#00d4aa"):
    """
    Render a simple styled stat card in Streamlit using HTML.

    Args:
        label: Card title label.
        value: Main value to display.
        delta: Optional trend indicator string.
        color: Accent color for the value.
    """
    delta_html = f"<p style='margin:0;font-size:12px;color:#888'>{delta}</p>" if delta else ""
    st.markdown(f"""
    <div style='background:#1a1a2e;border-radius:10px;padding:16px 20px;border-left:4px solid {color}'>
        <p style='margin:0;font-size:13px;color:#888;text-transform:uppercase;letter-spacing:1px'>{label}</p>
        <p style='margin:4px 0;font-size:26px;font-weight:700;color:{color}'>{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)
