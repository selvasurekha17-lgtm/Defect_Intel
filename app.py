"""
DefectIntel — AI-Powered Manufacturing Defect Intelligence & Early Warning System
Premium Streamlit SaaS Dashboard (app.py)

Pages:
    1. Overview            — KPI cards, group chart, batch chart, recent alerts
    2. AI Investigator     — Semantic search + cluster suggestion + confirm & add
    3. Batch Intelligence  — Two-batch comparison charts and delta table
    4. Alert Center        — 4-state alert workflow with filters and investigation
    5. Defect Explorer     — Searchable, filterable table with column explanations
    6. Data Management     — Presets, CSV upload, threshold tuning
    7. Model Evaluation    — ARI, NMI, Silhouette, cluster shape stats
"""

from io import BytesIO
from pathlib import Path
import textwrap
import uuid

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import (
    DATA_PATH,
    PRESET_DATASETS,
    ALERT_STATUSES,
    acknowledge_alert,
    unacknowledge_alert,
    reset_acknowledgements,
    analyze_dataset,
    batch_analytics,
    clustering_evaluation,
    compute_full_evaluation,
    compute_silhouette,
    assign_to_cluster,
    get_acknowledged_alerts,
    get_alert_statuses,
    set_alert_status,
    reset_alert_statuses,
    group_profiles,
    search_defect,
    validate_dataset,
    generate_investigation_recommendation,
    load_model,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DefectIntel | Manufacturing Quality Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design System CSS — Light Industrial SaaS Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Base ── */
:root {
  --navy: #17324D; --navy2: #2A4866; --navy3: #3D5A73;
  --teal: #14B8A6; --teal-d: #0F766E; --teal-l: #CCFBF1; --teal-ll: #F0FDFA;
  --sky: #38BDF8; --sky-l: #E0F2FE;
  --bg: #F7FAFC; --surface: #EDF6FC; --card: rgba(255,255,255,0.97);
  --border: #DDE9F2; --border2: #C8DCE9;
  --red: #EF4444; --red-l: #FEE2E2;
  --amber: #F59E0B; --amber-l: #FEF3C7;
  --green: #10B981; --green-l: #D1FAE5;
  --blue: #3B82F6; --blue-l: #DBEAFE;
  --text: #17324D; --text2: #3D5A73; --text3: #64748B;
}
html, body, [class*="css"] {
  font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
  color: var(--text);
}
.stApp { background-color: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #EAF3FB 0%, #E2EEF8 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--navy) !important; }

/* ── Metric Cards — light bg, dark text, no dark-on-dark ── */
[data-testid="stMetric"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  padding: 18px 22px !important;
  box-shadow: 0 4px 16px rgba(23,50,77,.05) !important;
  position: relative;
  overflow: hidden;
  transition: transform .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(23,50,77,.09) !important;
}
[data-testid="stMetric"]::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--teal), var(--sky));
}
[data-testid="stMetricLabel"] { color: var(--text3) !important; font-size: .82rem !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] {
  color: var(--navy) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 1.9rem !important;
  font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { color: var(--teal) !important; }

/* ── Cards (border containers) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 18px !important;
  box-shadow: 0 4px 16px rgba(23,50,77,.04) !important;
  transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
  margin-bottom: .8rem;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--border2) !important;
  box-shadow: 0 8px 24px rgba(23,50,77,.08) !important;
}

/* ── Buttons ── */
.stButton>button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  transition: all .18s ease !important;
  border: 1px solid var(--border) !important;
  color: var(--navy) !important;
}
.stButton>button[kind="primary"] {
  background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
  color: #fff !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(20,184,166,.28) !important;
}
.stButton>button[kind="primary"]:hover {
  box-shadow: 0 6px 18px rgba(20,184,166,.42) !important;
  transform: translateY(-1px) !important;
}

/* ── Text Inputs ── */
.stTextInput>div>div>input, .stTextArea>div>div>textarea {
  background: #fff !important;
  border: 1px solid #C8DCE9 !important;
  border-radius: 12px !important;
  color: var(--navy) !important;
}
.stTextInput>div>div>input:focus, .stTextArea>div>div>textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px rgba(20,184,166,.15) !important;
}

/* ── Selectbox / Multiselect ── */
[data-baseweb="select"] { border-radius: 12px !important; }

/* ── Expanders ── */
details[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  background: var(--card) !important;
}
details[data-testid="stExpander"] summary {
  font-weight: 600 !important;
  color: var(--navy) !important;
  padding: 12px 16px !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; }

/* ── Badge helpers ── */
.badge {
  display: inline-block;
  font-size: .76rem; font-weight: 700;
  padding: 3px 10px; border-radius: 999px;
  text-transform: uppercase; letter-spacing: .04em;
}
.badge-high { background: var(--red-l); color: #B91C1C; border: 1px solid rgba(239,68,68,.25); }
.badge-medium { background: var(--amber-l); color: #B45309; border: 1px solid rgba(245,158,11,.25); }
.badge-low { background: var(--blue-l); color: #1D4ED8; border: 1px solid rgba(59,130,246,.25); }
.badge-new { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }
.badge-ack { background: var(--sky-l); color: #0369A1; border: 1px solid rgba(56,189,248,.3); }
.badge-invest { background: var(--amber-l); color: #92400E; border: 1px solid rgba(245,158,11,.3); }
.badge-resolved { background: var(--green-l); color: #047857; border: 1px solid rgba(16,185,129,.3); }
.badge-teal { background: var(--teal-l); color: var(--teal-d); border: 1px solid rgba(20,184,166,.3); }
.badge-confirm { background: #EDE9FE; color: #5B21B6; border: 1px solid rgba(139,92,246,.3); }

/* ══════════════════════════════════════════════
   HERO — Premium 3D Light Theme
══════════════════════════════════════════════ */
.hero-wrap {
  position: relative; overflow: hidden;
  min-height: 340px;
  padding: 52px 60px 52px 60px;
  margin: 6px 0 32px;
  border-radius: 28px;
  border: 1.5px solid rgba(20,184,166,.22);
  background:
    radial-gradient(ellipse at 92% 20%, rgba(204,251,241,.85) 0%,  transparent 52%),
    radial-gradient(ellipse at 10% 90%, rgba(224,242,254,.65) 0%,  transparent 48%),
    radial-gradient(ellipse at 55% 50%, rgba(240,253,250,.50) 0%,  transparent 70%),
    linear-gradient(135deg, #FFFFFF 0%, #F5FBFF 42%, #EAF7F4 100%);
  box-shadow:
    0 0 0 1px rgba(255,255,255,.85) inset,
    0 24px 60px rgba(20,184,166,.09),
    0 6px 20px rgba(23,50,77,.05);
  display: flex; align-items: center; justify-content: space-between; gap: 28px;
}

/* Animated grid overlay */
.hero-wrap::before {
  content: "";
  position: absolute; inset: 0; border-radius: 28px;
  background-image:
    linear-gradient(rgba(20,184,166,.07) 1px, transparent 1px),
    linear-gradient(90deg, rgba(20,184,166,.07) 1px, transparent 1px);
  background-size: 38px 38px;
  animation: grid-drift 22s linear infinite;
  pointer-events: none;
}
/* Soft aurora sweep */
.hero-wrap::after {
  content: "";
  position: absolute; top: -60%; left: -20%;
  width: 90%; height: 200%;
  background: conic-gradient(
    from 0deg at 50% 50%,
    transparent 0deg,
    rgba(56,189,248,.06) 60deg,
    rgba(20,184,166,.10) 130deg,
    rgba(99,102,241,.05) 200deg,
    transparent 280deg
  );
  animation: aurora 16s ease-in-out infinite alternate;
  pointer-events: none; border-radius: 50%;
}

/* ── Hero text ── */
.hero-text { position: relative; z-index: 3; max-width: 640px; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 9px;
  font-size: .79rem; font-weight: 800; text-transform: uppercase; letter-spacing: .10em;
  color: #0F766E;
  background: linear-gradient(135deg, rgba(20,184,166,.12), rgba(56,189,248,.10));
  padding: 6px 16px; border-radius: 999px;
  border: 1px solid rgba(20,184,166,.35);
  margin-bottom: 18px;
  backdrop-filter: blur(4px);
  box-shadow: 0 2px 12px rgba(20,184,166,.15);
}
.pulse-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: radial-gradient(circle, #14B8A6 30%, #38BDF8 100%);
  box-shadow: 0 0 6px #14B8A6, 0 0 14px rgba(20,184,166,.6);
  animation: pulse-dot 2s ease-in-out infinite;
}
.hero-text h1 {
  font-size: clamp(2.1rem, 4vw, 3.2rem);
  font-weight: 800; color: #17324D;
  line-height: 1.08; letter-spacing: -.035em;
  margin: 0 0 16px;
  background: linear-gradient(135deg, #17324D 0%, #0F766E 55%, #0369A1 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub { font-size: 1.05rem; color: #4A6379; line-height: 1.65; margin: 0 0 22px; }
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.hero-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .78rem; font-weight: 600;
  color: #0F766E; background: rgba(20,184,166,.08);
  border: 1px solid rgba(20,184,166,.2);
  padding: 4px 12px; border-radius: 999px;
  animation: pill-pop .6s cubic-bezier(.34,1.56,.64,1) both;
}
.hero-pill:nth-child(2){animation-delay:.1s}
.hero-pill:nth-child(3){animation-delay:.2s}
.hero-pill:nth-child(4){animation-delay:.3s}

/* ══ 3-D Neural Orb Scene ══ */
.ai-scene {
  position: relative; z-index: 3;
  width: 300px; height: 300px;
  flex-shrink: 0;
  perspective: 900px;
  display: flex; align-items: center; justify-content: center;
}

/* Outer glow platform */
.ai-scene::before {
  content: "";
  position: absolute; bottom: 8px; left: 50%;
  transform: translateX(-50%) rotateX(75deg);
  width: 200px; height: 60px; border-radius: 50%;
  background: radial-gradient(ellipse, rgba(20,184,166,.22) 0%, transparent 72%);
  filter: blur(8px);
  animation: platform-pulse 3s ease-in-out infinite alternate;
}

.orb-root {
  width: 210px; height: 210px;
  position: relative; transform-style: preserve-3d;
  animation: scene-spin 22s linear infinite;
}

/* Core nucleus */
.nucleus {
  position: absolute;
  top: 50%; left: 50%;
  width: 88px; height: 88px;
  margin: -44px 0 0 -44px;
  border-radius: 50%;
  background:
    radial-gradient(circle at 30% 25%, rgba(255,255,255,.95) 0%,  #A5F3FC 25%,  #38BDF8 55%, #14B8A6 80%);
  box-shadow:
    0 0 0 10px rgba(20,184,166,.08),
    0 0 30px rgba(20,184,166,.4),
    0 0 60px rgba(56,189,248,.25),
    inset 0 2px 12px rgba(255,255,255,.9);
  animation: core-breathe 3.8s ease-in-out infinite alternate;
  z-index: 10;
}
/* Inner shimmer on nucleus */
.nucleus::after {
  content: "";
  position: absolute; top: 12%; left: 14%;
  width: 32%; height: 32%; border-radius: 50%;
  background: rgba(255,255,255,.75);
  filter: blur(4px);
  animation: shine-move 4s ease-in-out infinite alternate;
}

/* Orbital rings */
.orb-ring {
  position: absolute; inset: 0;
  border-radius: 50%; transform-style: preserve-3d;
}
.or1 {
  border: 2px solid rgba(20,184,166,.5);
  box-shadow: 0 0 8px rgba(20,184,166,.2);
  animation: ring-a 8s linear infinite;
}
.or2 {
  border: 1.5px solid rgba(56,189,248,.45);
  animation: ring-b 11s linear infinite reverse;
  inset: 14px;
}
.or3 {
  border: 1px dashed rgba(99,102,241,.3);
  animation: ring-c 16s linear infinite;
  inset: -10px;
}
.or4 {
  border: 1px solid rgba(20,184,166,.18);
  animation: ring-d 20s linear infinite reverse;
  inset: -28px;
}

/* Orbital nodes */
.onode {
  position: absolute; border-radius: 50%;
  background: radial-gradient(circle, #fff 0%, #38BDF8 50%, #14B8A6 100%);
  border: 1.5px solid rgba(255,255,255,.8);
  box-shadow: 0 0 8px #14B8A6, 0 0 16px rgba(56,189,248,.5);
  animation: node-glow 2.4s ease-in-out infinite alternate;
}
.on1 { width: 12px; height: 12px; top:-6px; left:50%; margin-left:-6px; }
.on2 { width: 10px; height: 10px; bottom:-5px; left:50%; margin-left:-5px; background:radial-gradient(circle,#fff 0%,#38BDF8 60%); }
.on3 { width: 11px; height: 11px; top:50%; left:-5px; margin-top:-5px; background:radial-gradient(circle,#fff 0%,#14B8A6 60%); animation-delay:.8s; }
.on4 { width: 9px; height: 9px; top:50%; right:-4px; margin-top:-4px; animation-delay:1.6s; }
.on5 { width: 8px; height: 8px; top:14px; right:14px; background:radial-gradient(circle,#fff 0%,#818CF8 60%); box-shadow:0 0 8px #818CF8; animation-delay:.5s; }
.on6 { width: 8px; height: 8px; bottom:14px; left:14px; background:radial-gradient(circle,#fff 0%,#34D399 60%); box-shadow:0 0 8px #34D399; animation-delay:1.2s; }

/* Floating particles */
.particle {
  position: absolute; border-radius: 50%;
  pointer-events: none; z-index: 2;
  animation: float-particle linear infinite;
}
.p1{width:5px;height:5px;background:rgba(20,184,166,.6);top:18%;left:8%;animation-duration:7s;animation-delay:0s;}
.p2{width:4px;height:4px;background:rgba(56,189,248,.55);top:65%;left:12%;animation-duration:9s;animation-delay:1.4s;}
.p3{width:6px;height:6px;background:rgba(20,184,166,.4);top:30%;right:6%;animation-duration:8s;animation-delay:2.2s;}
.p4{width:3px;height:3px;background:rgba(129,140,248,.6);top:78%;right:14%;animation-duration:6s;animation-delay:0.8s;}
.p5{width:5px;height:5px;background:rgba(52,211,153,.5);top:48%;left:4%;animation-duration:11s;animation-delay:3s;}
.p6{width:4px;height:4px;background:rgba(56,189,248,.45);top:12%;right:22%;animation-duration:10s;animation-delay:1.8s;}
.p7{width:3px;height:3px;background:rgba(20,184,166,.5);bottom:12%;right:8%;animation-duration:7.5s;animation-delay:4s;}
.p8{width:6px;height:6px;background:rgba(99,102,241,.35);top:52%;right:2%;animation-duration:12s;animation-delay:2.5s;}

/* Data-connection lines (SVG in HTML) */
.conn-lines {
  position: absolute; inset: 0;
  pointer-events: none; z-index: 1;
  opacity: .45;
}

/* ── KEYFRAMES ── */
@keyframes grid-drift {
  from { background-position: 0 0; }
  to   { background-position: 38px 38px; }
}
@keyframes aurora {
  from { transform: rotate(0deg) scale(1);   opacity: .5; }
  to   { transform: rotate(20deg) scale(1.1); opacity: .8; }
}
@keyframes scene-spin {
  from { transform: rotateY(0deg)   rotateX(12deg); }
  to   { transform: rotateY(360deg) rotateX(12deg); }
}
@keyframes ring-a {
  to { transform: rotateX(68deg)  rotateZ(360deg); }
}
@keyframes ring-b {
  to { transform: rotateX(-52deg) rotateZ(360deg); }
}
@keyframes ring-c {
  to { transform: rotateX(28deg)  rotateZ(360deg); }
}
@keyframes ring-d {
  to { transform: rotateX(-14deg) rotateZ(360deg); }
}
@keyframes core-breathe {
  from { transform: scale(.92); box-shadow: 0 0 0 10px rgba(20,184,166,.06), 0 0 24px rgba(20,184,166,.35), 0 0 50px rgba(56,189,248,.18); }
  to   { transform: scale(1.07); box-shadow: 0 0 0 16px rgba(20,184,166,.10), 0 0 42px rgba(20,184,166,.55), 0 0 80px rgba(56,189,248,.32); }
}
@keyframes shine-move {
  from { transform: translate(0,0);    opacity: .9; }
  to   { transform: translate(14%,8%); opacity: .4; }
}
@keyframes node-glow {
  from { box-shadow: 0 0 6px #14B8A6, 0 0 12px rgba(56,189,248,.4); transform: scale(.9); }
  to   { box-shadow: 0 0 12px #14B8A6, 0 0 22px rgba(56,189,248,.7); transform: scale(1.15); }
}
@keyframes float-particle {
  0%,100% { transform: translateY(0px)  translateX(0px)  scale(1);   opacity: .7; }
  25%      { transform: translateY(-14px) translateX(6px)  scale(1.2); opacity: 1; }
  50%      { transform: translateY(-8px)  translateX(-8px) scale(.9);  opacity: .5; }
  75%      { transform: translateY(-20px) translateX(4px)  scale(1.1); opacity: .9; }
}
@keyframes platform-pulse {
  from { opacity: .5; transform: translateX(-50%) rotateX(75deg) scaleX(.85); }
  to   { opacity: 1;  transform: translateX(-50%) rotateX(75deg) scaleX(1.05); }
}
@keyframes pulse-dot {
  0%,100% { transform: scale(1);    opacity: 1; box-shadow: 0 0 6px #14B8A6; }
  50%     { transform: scale(1.45); opacity: .5; box-shadow: 0 0 18px #14B8A6; }
}
@keyframes pill-pop {
  from { transform: scale(0); opacity: 0; }
  to   { transform: scale(1); opacity: 1; }
}

/* ── Similarity bar ── */
.sim-bar-track { background: #E2EDF4; border-radius: 999px; height: 7px; width: 100%; overflow: hidden; }
.sim-bar-fill { height: 7px; border-radius: 999px; }

/* ── Column legend ── */
.col-legend { background: var(--teal-ll); border: 1px solid rgba(20,184,166,.2); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }
.col-legend p { margin: 4px 0; font-size: .88rem; color: var(--text); }
.col-legend strong { color: var(--teal-d); }

@media (prefers-reduced-motion: reduce) {
  *, ::before, ::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
}
@media (max-width:800px) {
  .hero-wrap { padding: 24px; flex-direction: column; }
  .ai-3d-wrap { width: 150px; height: 150px; }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def analyze_csv(csv_bytes: bytes, distance_threshold: float):
    source = pd.read_csv(BytesIO(csv_bytes))
    return analyze_dataset(source, distance_threshold=distance_threshold)


def short_label(text: str, width: int = 46) -> str:
    return textwrap.shorten(str(text), width=width, placeholder="…")


def build_labels(profiles):
    return {p["cluster"]: short_label(p["label"]) for p in profiles}


def themed_bar(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=32, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#17324D", size=12),
        xaxis=dict(gridcolor="#E8EFF6", zerolinecolor="#C8DCE9",
                   tickfont=dict(color="#4A6379")),
        yaxis=dict(gridcolor="#E8EFF6", zerolinecolor="#C8DCE9",
                   tickfont=dict(color="#4A6379")),
        legend=dict(bgcolor="rgba(255,255,255,.75)", bordercolor="#E2EDF4",
                    borderwidth=1, font=dict(size=11)),
    )
    return fig


def severity_badge(sev: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(sev, "badge-new")
    return f'<span class="badge {cls}">{sev}</span>'


def status_badge(status: str) -> str:
    labels = {
        "new": ("NEW", "badge-new"),
        "acknowledged": ("ACKNOWLEDGED", "badge-ack"),
        "under_investigation": ("INVESTIGATING", "badge-invest"),
        "resolved": ("RESOLVED", "badge-resolved"),
    }
    label, cls = labels.get(status, ("UNKNOWN", "badge-new"))
    return f'<span class="badge {cls}">{label}</span>'


def sim_gauge(score: float) -> str:
    pct = int(score * 100)
    color = "#14B8A6" if score >= .70 else "#38BDF8" if score >= .50 else "#94A3B8"
    return (
        f'<div class="sim-bar-track">'
        f'<div class="sim-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session State initialisation
# ─────────────────────────────────────────────────────────────────────────────

if "active_csv_bytes" not in st.session_state:
    st.session_state["active_csv_bytes"] = DATA_PATH.read_bytes()
    st.session_state["active_data_name"] = "Mobile Electronics (100 reports)"
if "distance_threshold" not in st.session_state:
    st.session_state["distance_threshold"] = 0.50
if "rule_min_count" not in st.session_state:
    st.session_state["rule_min_count"] = 5
if "rule_increase_thresh" not in st.session_state:
    st.session_state["rule_increase_thresh"] = 50
if "rule_high_thresh" not in st.session_state:
    st.session_state["rule_high_thresh"] = 100
if "pending_reports" not in st.session_state:
    st.session_state["pending_reports"] = []
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = None   # used to override navigation from KPI click


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
  <span style="font-size:1.55rem;color:#14B8A6;">◈</span>
  <span style="font-size:1.3rem;font-weight:800;color:#17324D;letter-spacing:-.02em;">DefectIntel</span>
</div>
<div style="font-size:.72rem;font-weight:700;color:#0F766E;letter-spacing:.1em;margin-bottom:16px;">
  MANUFACTURING QUALITY AI
</div>
""", unsafe_allow_html=True)

_page_opts = [
    "🏠 Overview",
    "🔍 AI Investigator",
    "📊 Batch Intelligence",
    "🚨 Alert Center",
    "📋 Defect Explorer",
    "📁 Data Management",
    "🧪 Model Evaluation",
    "🔬 Visual Inspection (Beta)",
]

# Allow KPI-click navigation
_default_idx = 0
if st.session_state["nav_page"] in _page_opts:
    _default_idx = _page_opts.index(st.session_state["nav_page"])
    st.session_state["nav_page"] = None

page = st.sidebar.radio("WORKSPACE", _page_opts, index=_default_idx)

st.sidebar.markdown("---")
st.sidebar.markdown("**Comparative Batch Selection**")
batch_options_sidebar = []   # filled after df is loaded

# ─────────────────────────────────────────────────────────────────────────────
# Load & Analyse Active Dataset
# ─────────────────────────────────────────────────────────────────────────────

active_bytes = st.session_state["active_csv_bytes"]
active_threshold = float(st.session_state["distance_threshold"])

try:
    with st.spinner("Running semantic AI analysis (all-MiniLM-L6-v2)…"):
        df, embeddings = analyze_csv(active_bytes, active_threshold)
except Exception as _err:
    st.error(f"Could not analyse the active dataset: {_err}")
    st.info("Go to Data Management to upload a compatible CSV or restore the sample dataset.")
    st.stop()

profiles = group_profiles(df, embeddings)
labels = build_labels(profiles)
batch_options = sorted(df["batch_id"].astype(str).unique().tolist())

# ── Sidebar batch selectors ──
previous_batch = current_batch = None
if len(batch_options) >= 2:
    previous_batch = st.sidebar.selectbox("Baseline Lot", batch_options, index=0, key="sb_prev")
    cmp_opts = [b for b in batch_options if b != previous_batch]
    current_batch = st.sidebar.selectbox("Comparison Lot", cmp_opts, index=0, key="sb_curr")
    st.sidebar.caption("Batch IDs are production lot labels. Alphabetical order does not imply chronological order.")

# ── Compute alerts ──
try:
    summary, alerts = (
        batch_analytics(
            df,
            previous_batch=previous_batch,
            current_batch=current_batch,
            min_current_count=st.session_state["rule_min_count"],
            increase_threshold=st.session_state["rule_increase_thresh"],
            high_increase_threshold=st.session_state["rule_high_thresh"],
        )
        if previous_batch and current_batch
        else batch_analytics(df)
    )
except ValueError as _ve:
    st.sidebar.warning(str(_ve))
    summary, alerts = pd.DataFrame(), []

alert_statuses = get_alert_statuses()
active_alerts = [a for a in alerts if alert_statuses.get(a["alert_id"], {}).get("status", "new") != "resolved"]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** `{st.session_state['active_data_name']}`")
st.sidebar.markdown(f"**Reports:** `{len(df):,}` validated")
st.sidebar.markdown(f"**Groups:** `{df['ai_cluster'].nunique():,}` AI clusters")
if active_alerts:
    st.sidebar.markdown(
        f"**Alerts:** <span class='badge badge-high'>{len(active_alerts)} ACTIVE</span>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Overview":
    # Hero
    st.markdown("""
    <section class="hero-wrap">
      <div class="hero-text">
        <div class="hero-badge"><span class="pulse-dot"></span>DefectIntel · AI Quality Intelligence</div>
        <h1>Understand Defects<br>Before They Escalate.</h1>
        <p>AI-powered manufacturing quality intelligence across industries.<br>
           Semantic clustering · Similarity search · Evidence-based early warnings.</p>
      </div>
      <div class="ai-3d-wrap" aria-hidden="true">
        <div class="orb-cluster">
          <div class="ring r1"><span class="snode sn1"></span><span class="snode sn2"></span></div>
          <div class="ring r2"><span class="snode sn3"></span><span class="snode sn4"></span></div>
          <div class="ring r3"></div>
          <div class="nucleus"></div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    st.caption(
        "Decision-support tool for quality engineers. Similarity scores are ranking signals, not "
        "calibrated probabilities. Report-count alerts are not defect rates."
    )

    # KPI cards — clicking Active Alerts navigates to Alert Center
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reports Analyzed", f"{len(df):,}")
    c2.metric("AI Defect Groups", f"{df['ai_cluster'].nunique():,}")
    c3.metric("Production Batches", f"{df['batch_id'].nunique():,}")
    with c4:
        st.metric("Active Alerts", len(active_alerts))
        if st.button("→ Open Alert Center", key="kpi_alert_nav"):
            st.session_state["nav_page"] = "🚨 Alert Center"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Most Frequent Groups chart + Group Representatives table
    lc, rc = st.columns([1.38, 1])
    with lc:
        with st.container(border=True):
            st.markdown("### Most Frequent Defect Groups")
            st.caption("Titles are AI-selected representative descriptions, not verified diagnoses.")
            top_n = df["ai_cluster"].value_counts().sort_values(ascending=False).head(8)
            gdf = pd.DataFrame({
                "Group": [labels.get(int(c), f"Group {c}") for c in top_n.index],
                "Reports": top_n.values,
            })
            fig_grp = px.bar(
                gdf, x="Reports", y="Group", orientation="h",
                color="Reports", color_continuous_scale=[[0, "#38BDF8"], [1, "#14B8A6"]],
                hover_data={"Reports": True, "Group": True},
            )
            fig_grp.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(themed_bar(fig_grp, 330), use_container_width=True)

            # Drill-down
            sel_grp = st.selectbox(
                "Drill-down: inspect a group",
                options=[p["cluster"] for p in sorted(profiles, key=lambda x: -x["report_count"])],
                format_func=lambda c: f"Group {c} — {labels.get(c, '?')} ({next((p['report_count'] for p in profiles if p['cluster']==c), 0)} reports)",
                key="ov_drilldown",
            )
            group_rows = df[df["ai_cluster"] == sel_grp]
            cols_show = [c for c in ["report_id", "batch_id", "defect_description", "category", "severity"] if c in group_rows.columns]
            st.dataframe(group_rows[cols_show].head(12), use_container_width=True, hide_index=True, height=220)

    with rc:
        with st.container(border=True):
            st.markdown("### Group Representatives")
            st.caption("Centroid-nearest report selected per group. Report counts shown.")
            pdf = pd.DataFrame(profiles).sort_values("report_count", ascending=False)[
                ["cluster", "label", "report_count"]
            ].rename(columns={"cluster": "Group ID", "label": "Representative Issue", "report_count": "Reports"})
            st.dataframe(pdf.head(10), use_container_width=True, hide_index=True, height=330)

    # Reports per Batch chart
    with st.container(border=True):
        st.markdown("### Reports per Production Lot")
        st.caption("Report counts only — production volumes are not supplied, so defect rates cannot be computed.")
        lot_counts = df["batch_id"].value_counts().sort_index().reset_index()
        lot_counts.columns = ["Lot", "Reports"]
        fig_lots = px.bar(
            lot_counts, x="Lot", y="Reports",
            color="Reports", color_continuous_scale=[[0, "#EDF6FC"], [1, "#0284C7"]],
            hover_data={"Lot": True, "Reports": True},
        )
        fig_lots.update_layout(coloraxis_showscale=False)
        st.plotly_chart(themed_bar(fig_lots, 280), use_container_width=True)

    # ARI / NMI snippet
    eval_res = clustering_evaluation(df)
    if eval_res:
        with st.expander("Clustering Concordance vs Reference Labels (Optional Evaluation)"):
            st.caption(
                f"Comparison against `{eval_res['reference_column']}` labels for "
                f"{eval_res['reports_evaluated']} reports. ARI and NMI measure "
                f"clustering agreement — they are NOT classification accuracy."
            )
            e1, e2 = st.columns(2)
            e1.metric("Adjusted Rand Index (ARI)", f"{eval_res['adjusted_rand_index']:.3f}",
                      help="Agreement adjusted for chance. Range: -1 to +1. Higher is better.")
            e2.metric("Normalized Mutual Info (NMI)", f"{eval_res['normalized_mutual_info']:.3f}",
                      help="Information-theoretic agreement. Range: 0 to 1. Higher is better.")

    # Recent unresolved alerts
    if active_alerts:
        st.markdown("### Unresolved Batch Warnings")
        for al in active_alerts[:3]:
            title = labels.get(al["cluster"], f"Group {al['cluster']}")
            pct = "No baseline data" if al["increase_percent"] is None else f"+{al['increase_percent']:.1f}%"
            s = alert_statuses.get(al["alert_id"], {}).get("status", "new")
            with st.container(border=True):
                h1, h2 = st.columns([3, 1])
                h1.markdown(
                    f"{severity_badge(al['severity'])} &nbsp; {status_badge(s)} &nbsp; "
                    f"**{title}**",
                    unsafe_allow_html=True,
                )
                h1.write(f"Lot **{al['previous_batch']}** ({al['previous_count']}) → Lot **{al['batch']}** ({al['current_count']}) · **{pct}**")
                with h2:
                    if st.button("Open in Alerts →", key=f"ov_open_{al['alert_id']}"):
                        st.session_state["nav_page"] = "🚨 Alert Center"
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — AI INVESTIGATOR
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 AI Investigator":
    st.title("AI Defect Investigator")
    st.markdown(
        "Enter a new complaint or field observation. The system embeds it into 384-dimensional "
        "semantic space and retrieves the most similar historical defect records — regardless of "
        "wording differences."
    )

    # Quick-test chips (for live demo)
    st.markdown("**Quick-Test Chips (click to populate):**")
    qc = st.columns(4)
    if qc[0].button("📱 Display blinking during charge"):
        st.session_state["inv_q"] = "Phone display keeps blinking continuously while charging"
    if qc[1].button("🔌 USB port fails to charge"):
        st.session_state["inv_q"] = "USB-C port is loose and fails to charge device unless pressed at angle"
    if qc[2].button("🚗 Door panel gap exceeds spec"):
        st.session_state["inv_q"] = "Inspection noted door panel gap exceeds specified tolerance on driver side"
    if qc[3].button("🧵 Yarn loose across woven surface"):
        st.session_state["inv_q"] = "Loose yarns visible across the woven fabric surface near seam"

    with st.form("inv_form"):
        default_q = st.session_state.get("inv_q", "")
        query = st.text_area(
            "Defect Description or Customer Complaint",
            value=default_q,
            placeholder="e.g. Screen flickers randomly after phone unlocks",
            height=90,
        )
        fc1, fc2, fc3 = st.columns([1, 1, 2])
        top_k = fc1.slider("Top-K Results", 3, 10, 5)
        min_sim = fc2.slider("Min Similarity Filter", 0.0, 0.9, 0.25, 0.05)
        assign_thresh = fc3.slider(
            "Cluster Suggestion Threshold",
            0.0, 0.9, 0.40, 0.05,
            help="Centroid similarity below this marks the report as 'Potential New Group'."
        )
        submitted = st.form_submit_button("🔍 Semantic Vector Search", type="primary")

    if submitted and not query.strip():
        st.warning("Please enter a defect description.")

    if submitted and query.strip():
        try:
            with st.spinner("Embedding query and computing cosine similarity…"):
                results = search_defect(query, df, embeddings, limit=top_k)
                model = load_model()
                q_emb = model.encode([query.strip()], normalize_embeddings=True)
                suggested_cluster, centroid_sim, assign_type = assign_to_cluster(
                    q_emb[0], df, embeddings, min_similarity=assign_thresh
                )

            filtered = [r for r in results if r["similarity"] >= min_sim]

            st.caption(
                "Cosine similarity is a ranking signal in SentenceTransformer embedding space — "
                "not a classification probability, defect prediction, or model accuracy score."
            )

            # ── Cluster Suggestion Banner ──
            if assign_type == "suggested":
                rep = labels.get(suggested_cluster, f"Group {suggested_cluster}")
                st.success(
                    f"**Suggested AI Group:** Group {suggested_cluster} — *{rep}*  \n"
                    f"Centroid similarity: **{centroid_sim:.3f}**  \n"
                    f"*Note: This is a heuristic centroid-distance assignment, not re-clustering.*"
                )
                st.session_state["inv_suggested_cluster"] = suggested_cluster
                st.session_state["inv_centroid_sim"] = centroid_sim
            else:
                st.warning(
                    f"**Needs Review — Potential New Issue**  \n"
                    f"Best centroid similarity: **{centroid_sim:.3f}** (below threshold {assign_thresh:.2f}).  \n"
                    f"This complaint may represent a new failure mode not seen in the current dataset."
                )
                st.session_state["inv_suggested_cluster"] = None
                st.session_state["inv_centroid_sim"] = centroid_sim

            st.session_state["inv_q"] = query

            # ── Results ──
            if not filtered:
                st.info(f"No results met the minimum similarity of {min_sim:.2f}.")
            else:
                st.markdown(f"**{len(filtered)} Semantically Similar Reports:**")
                for rank, res in enumerate(filtered, 1):
                    with st.container(border=True):
                        r1, r2 = st.columns([3, 1])
                        with r1:
                            grp_label = labels.get(res["cluster"], f"Group {res['cluster']}")
                            st.markdown(f"**#{rank} · {grp_label}**")
                            st.write(f'"{res["description"]}"')
                        with r2:
                            st.metric("Cosine Similarity", f"{res['similarity']:.3f}")
                            st.markdown(sim_gauge(res["similarity"]), unsafe_allow_html=True)
                        mc = st.columns(4)
                        mc[0].markdown(f"**Lot:** `{res['batch_id']}`")
                        mc[1].markdown(f"**Report:** `{res['report_id']}`")
                        mc[2].markdown(f"**Category:** `{res['category']}`")
                        mc[3].markdown(f"**Severity:** `{res['severity']}`")
                        with st.expander("Why did AI match this?"):
                            st.markdown(
                                f"Both descriptions share semantic manufacturing fault signatures "
                                f"in the same SentenceTransformer vector neighbourhood "
                                f"(AI Cluster #{res['cluster']})."
                            )

            # ── Confirm & Add Report ──
            st.markdown("---")
            with st.expander("➕ Confirm & Add This Report to the Working Dataset"):
                st.info(
                    "Use this to add a manually verified report. The record will be tagged "
                    "`data_origin = confirmed_manual`. The original uploaded CSV is not modified. "
                    "Re-run analysis to update clusters and alerts."
                )
                sc1, sc2 = st.columns(2)
                new_batch = sc1.text_input("Batch ID *", placeholder="e.g. B005")
                new_report_id = sc2.text_input("Report ID", placeholder="auto-generated if blank")
                sc3, sc4, sc5 = st.columns(3)
                new_product = sc3.text_input("Product", placeholder="Optional")
                new_cat = sc4.text_input("Category", placeholder="Optional")
                new_sev = sc5.selectbox("Severity", ["Not assessed", "Low", "Medium", "High", "Critical"])

                cluster_note = (
                    f"AI suggested: Group {suggested_cluster} — "
                    f"{labels.get(suggested_cluster, '?')}"
                    if suggested_cluster is not None
                    else "No cluster matched threshold — will be assigned on re-analysis"
                )
                st.caption(f"Cluster note: {cluster_note}")

                if st.button("✅ Add to Working Dataset", type="primary", key="confirm_add"):
                    if not new_batch.strip():
                        st.error("Batch ID is required.")
                    else:
                        new_row = {
                            "report_id": new_report_id.strip() or f"MAN-{uuid.uuid4().hex[:6].upper()}",
                            "batch_id": new_batch.strip(),
                            "defect_description": query.strip(),
                            "product": new_product.strip() or "Unknown",
                            "category": new_cat.strip() or "Not assessed",
                            "severity": new_sev if new_sev != "Not assessed" else "",
                            "data_origin": "confirmed_manual",
                        }
                        st.session_state["pending_reports"].append(new_row)
                        st.success(
                            f"Report added to pending list "
                            f"({len(st.session_state['pending_reports'])} pending). "
                            f"Go to **Data Management** and click 'Re-run Analysis with Pending Reports' to update clusters."
                        )
        except Exception as exc:
            st.error(f"Search failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — BATCH INTELLIGENCE
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📊 Batch Intelligence":
    st.title("Batch Intelligence & Defect Distribution")
    st.markdown(
        "Compare defect-group concentrations between two production lots. "
        "Changes in report counts may indicate emerging quality issues — but without "
        "production-volume denominators, these are count changes, not defect rates."
    )

    if len(batch_options) < 2:
        st.warning("At least two distinct batch IDs are required.")
        st.stop()

    b1, b2 = st.columns(2)
    bi_prev = b1.selectbox("Baseline Lot (A)", batch_options, index=0, key="bi_prev")
    bi_curr_opts = [b for b in batch_options if b != bi_prev]
    bi_curr = b2.selectbox("Comparison Lot (B)", bi_curr_opts, index=0, key="bi_curr")

    st.info(
        "Batch IDs are treated as production lot labels only. "
        "Alphabetical or numeric order does not imply chronological order unless your dataset supplies a date/sequence column."
    )

    # Compute per-page analytics
    try:
        bi_summary, bi_alerts = batch_analytics(
            df,
            previous_batch=bi_prev,
            current_batch=bi_curr,
            min_current_count=st.session_state["rule_min_count"],
            increase_threshold=st.session_state["rule_increase_thresh"],
            high_increase_threshold=st.session_state["rule_high_thresh"],
        )
    except ValueError as e:
        st.error(str(e))
        st.stop()

    cnt_a = len(df[df["batch_id"] == bi_prev])
    cnt_b = len(df[df["batch_id"] == bi_curr])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Lot {bi_prev} Reports", cnt_a)
    k2.metric(f"Lot {bi_curr} Reports", cnt_b)
    k3.metric("Net Change", f"{cnt_b - cnt_a:+d}")
    k4.metric("Triggered Alerts", len(bi_alerts))
    st.caption("Counts are not defect rates. Production-lot sizes are not supplied.")

    # Grouped comparison bar chart
    with st.container(border=True):
        st.markdown(f"### Group Comparison: {bi_prev} vs {bi_curr}")
        sel_lots_df = df[df["batch_id"].isin([bi_prev, bi_curr])]
        gc = sel_lots_df.groupby(["batch_id", "ai_cluster"]).size().reset_index(name="Reports")
        gc["Defect Group"] = gc["ai_cluster"].map(labels)
        fig_comp = px.bar(
            gc, x="Defect Group", y="Reports", color="batch_id",
            barmode="group", color_discrete_sequence=["#14B8A6", "#38BDF8"],
            labels={"batch_id": "Production Lot"},
            hover_data={"Reports": True, "Defect Group": True, "batch_id": True},
        )
        st.plotly_chart(themed_bar(fig_comp, 380), use_container_width=True)

    # Donut charts
    dc1, dc2 = st.columns(2)
    for lot, col in [(bi_prev, dc1), (bi_curr, dc2)]:
        lot_df = df[df["batch_id"] == lot]
        lot_gc = lot_df["ai_cluster"].value_counts().reset_index()
        lot_gc.columns = ["ai_cluster", "Count"]
        lot_gc["Group"] = lot_gc["ai_cluster"].map(labels)
        fig_d = px.pie(lot_gc, names="Group", values="Count", hole=0.46, title=f"Defect Mix — Lot {lot}")
        col.plotly_chart(themed_bar(fig_d, 320), use_container_width=True)

    # Delta summary table
    with st.container(border=True):
        st.markdown("### Group-Level Report Delta")
        st.caption("Shows report counts per group for both selected lots.")
        rows = []
        for cluster_id in sorted(bi_summary.columns):
            cnt_baseline = int(bi_summary.loc[bi_prev, cluster_id]) if bi_prev in bi_summary.index else 0
            cnt_compare = int(bi_summary.loc[bi_curr, cluster_id]) if bi_curr in bi_summary.index else 0
            delta = cnt_compare - cnt_baseline
            pct = f"+{(delta / cnt_baseline * 100):.1f}%" if cnt_baseline > 0 and delta > 0 else (
                "New (no baseline)" if cnt_baseline == 0 and cnt_compare > 0 else f"{delta:+d}"
            )
            triggered = any(a["cluster"] == cluster_id for a in bi_alerts)
            rows.append({
                "Group ID": int(cluster_id),
                "Representative Defect": labels.get(int(cluster_id), "—"),
                f"Lot {bi_prev} Count": cnt_baseline,
                f"Lot {bi_curr} Count": cnt_compare,
                "Delta": delta,
                "% Change": pct,
                "⚠ Alert": "Yes" if triggered else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ALERT CENTER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🚨 Alert Center":
    st.title("Evidence-Based Alert Center")
    st.markdown(
        "Rule-based report-count spike detection. Alerts are triggered when configured "
        "thresholds are exceeded. **These are not ML predictions or defect rate alarms.** "
        "An alert should prompt human review, not autonomous action."
    )

    # Configuration panel
    with st.expander("⚙️ Alert Rule Configuration", expanded=False):
        cfg1, cfg2, cfg3 = st.columns(3)
        min_cnt = cfg1.number_input("Min reports in comparison lot", 1, 100,
                                     st.session_state["rule_min_count"])
        inc_t = cfg2.slider("Medium alert threshold (%)", 10, 300,
                             st.session_state["rule_increase_thresh"], 10)
        high_t = cfg3.slider("High alert threshold (%)", 50, 500,
                              st.session_state["rule_high_thresh"], 10)
        if st.button("Apply Rule Changes", type="primary"):
            st.session_state["rule_min_count"] = int(min_cnt)
            st.session_state["rule_increase_thresh"] = int(inc_t)
            st.session_state["rule_high_thresh"] = int(high_t)
            st.success("Rules saved.")
            st.rerun()

    # Recalculate with current rules and selected batches
    if previous_batch and current_batch:
        _, page_alerts = batch_analytics(
            df,
            previous_batch=previous_batch,
            current_batch=current_batch,
            min_current_count=st.session_state["rule_min_count"],
            increase_threshold=st.session_state["rule_increase_thresh"],
            high_increase_threshold=st.session_state["rule_high_thresh"],
        )
    else:
        page_alerts = []
        st.info("Select two different lots in the sidebar to generate alerts.")

    alert_statuses = get_alert_statuses()

    # Status legend
    st.markdown("""
    <div style="background:#F8FAFC;border:1px solid #DDE9F2;border-radius:12px;padding:12px 18px;margin-bottom:12px;font-size:.88rem;color:#3D5A73;">
    <strong>Alert Lifecycle:</strong> &nbsp;
    <span class="badge badge-new">NEW</span> → engineer is notified &nbsp;|&nbsp;
    <span class="badge badge-ack">ACKNOWLEDGED</span> → engineer has seen it (defect not yet resolved) &nbsp;|&nbsp;
    <span class="badge badge-invest">INVESTIGATING</span> → root-cause analysis in progress &nbsp;|&nbsp;
    <span class="badge badge-resolved">RESOLVED</span> → corrective action confirmed
    </div>
    """, unsafe_allow_html=True)

    # KPI strip
    ac1, ac2, ac3, ac4 = st.columns(4)
    sev_high = sum(1 for a in page_alerts if a["severity"] == "High")
    sev_med = sum(1 for a in page_alerts if a["severity"] == "Medium")
    n_resolved = sum(1 for a in page_alerts if alert_statuses.get(a["alert_id"], {}).get("status") == "resolved")
    n_active = len(page_alerts) - n_resolved
    ac1.metric("High Priority", sev_high)
    ac2.metric("Medium Priority", sev_med)
    ac3.metric("Active (unresolved)", n_active)
    ac4.metric("Resolved", n_resolved)

    # Severity distribution chart
    if page_alerts:
        sv_df = pd.DataFrame([{"Severity": a["severity"]} for a in page_alerts])
        sv_counts = sv_df["Severity"].value_counts().reindex(["High", "Medium"], fill_value=0).reset_index()
        sv_counts.columns = ["Severity", "Count"]
        fig_sv = px.bar(sv_counts, x="Severity", y="Count", color="Severity",
                        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B"})
        st.plotly_chart(themed_bar(fig_sv, 220), use_container_width=True)

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    sev_filt = f1.multiselect("Severity", ["High", "Medium"], default=["High", "Medium"])
    status_filt = f2.multiselect(
        "Status", list(ALERT_STATUSES),
        default=["new", "acknowledged", "under_investigation"],
        format_func=lambda s: s.replace("_", " ").title(),
    )
    lot_filt = f3.multiselect("Affected Lot", sorted({a["batch"] for a in page_alerts}),
                               default=sorted({a["batch"] for a in page_alerts}))
    with f4:
        if st.button("Reset All Statuses"):
            reset_alert_statuses()
            st.success("All alert statuses reset.")
            st.rerun()

    visible = [
        a for a in page_alerts
        if a["severity"] in sev_filt
        and alert_statuses.get(a["alert_id"], {}).get("status", "new") in status_filt
        and (not lot_filt or a["batch"] in lot_filt)
    ]

    st.markdown(f"**{len(visible)} alert(s) match current filters:**")

    if not page_alerts:
        st.success("No threshold breaches detected for the selected lot comparison.")
    elif not visible:
        st.info("No alerts match the current filters.")
    else:
        for al in visible:
            title = labels.get(al["cluster"], f"Group {al['cluster']}")
            status = alert_statuses.get(al["alert_id"], {}).get("status", "new")
            notes = alert_statuses.get(al["alert_id"], {}).get("notes", "")
            pct = "No baseline data" if al["increase_percent"] is None else f"+{al['increase_percent']:.1f}%"
            grp_recs = df[
                (df["ai_cluster"] == al["cluster"])
                & (df["batch_id"].isin([al["previous_batch"], al["batch"]]))
            ]

            with st.container(border=True):
                hc1, hc2 = st.columns([3, 1])
                with hc1:
                    st.markdown(
                        f"{severity_badge(al['severity'])} &nbsp; {status_badge(status)} &nbsp; "
                        f"<strong style='font-size:1.08rem;'>{title}</strong>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"Lot **{al['previous_batch']}**: {al['previous_count']} reports → "
                        f"Lot **{al['batch']}**: {al['current_count']} reports · **{pct}**"
                    )
                with hc2:
                    status_options = {
                        "new": "Mark as New",
                        "acknowledged": "Acknowledge",
                        "under_investigation": "Start Investigation",
                        "resolved": "Mark Resolved",
                    }
                    next_actions = {
                        "new": ["acknowledged"],
                        "acknowledged": ["under_investigation", "resolved"],
                        "under_investigation": ["resolved", "acknowledged"],
                        "resolved": ["acknowledged"],
                    }
                    for next_s in next_actions.get(status, []):
                        btn_label = status_options[next_s]
                        btn_type = "primary" if next_s == "resolved" else "secondary"
                        if st.button(btn_label, key=f"al_{al['alert_id']}_{next_s}", type=btn_type):
                            set_alert_status(al["alert_id"], next_s, notes)
                            st.rerun()

                with st.expander("Investigate: Supporting Reports & Recommendations"):
                    st.markdown(f"**Engineering Recommendation:**")
                    st.info(al.get("recommendation", "Conduct gate inspection on affected lot."))
                    cols_show = [c for c in ["report_id", "batch_id", "defect_description", "category", "severity", "data_origin"] if c in grp_recs.columns]
                    st.dataframe(grp_recs[cols_show], use_container_width=True, hide_index=True)

                    new_note = st.text_input("Investigation Notes", value=notes, key=f"note_{al['alert_id']}")
                    if st.button("Save Notes", key=f"savenote_{al['alert_id']}"):
                        set_alert_status(al["alert_id"], status, new_note)
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — DEFECT EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📋 Defect Explorer":
    st.title("Defect Report Explorer")
    st.markdown("Search, filter, and inspect all validated manufacturing defect records.")

    # Column legend
    st.markdown("""
    <div class="col-legend">
    <p><strong>report_id</strong> — Unique identifier for this inspection record.</p>
    <p><strong>batch_id</strong> — Manufacturing lot or production batch label.</p>
    <p><strong>defect_description</strong> — Free-text description of the observed quality issue.</p>
    <p><strong>category</strong> — Reference defect category (from CSV, not inferred by AI).</p>
    <p><strong>severity</strong> — Severity supplied in the source CSV, or "Not assessed" if absent. The semantic model does not assign severity.</p>
    <p><strong>ai_cluster</strong> — Internal group ID assigned by Agglomerative Clustering. Not the number of defects. See group representatives for readable descriptions.</p>
    <p><strong>data_origin</strong> — Provenance tag: <em>synthetic</em> (demo data), <em>uploaded</em> (user CSV), or <em>confirmed_manual</em> (added via AI Investigator).</p>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    search_q = f1.text_input("Search description text", placeholder="e.g. flicker, gap, seal")
    lot_sel = f2.multiselect("Production Lot", batch_options)
    cats = sorted(df["category"].dropna().astype(str).unique()) if "category" in df else []
    cat_sel = f3.multiselect("Category", cats)
    sevs = sorted(df["severity"].dropna().astype(str).unique()) if "severity" in df else []
    sev_sel = f4.multiselect("Severity", sevs)

    f5, f6, f7 = st.columns(3)
    grp_opts = sorted(df["ai_cluster"].unique())
    grp_sel = f5.multiselect(
        "AI Defect Group",
        grp_opts,
        format_func=lambda c: f"Group {c} — {labels.get(int(c), '?')}",
    )
    origins = sorted(df["data_origin"].dropna().astype(str).unique()) if "data_origin" in df else []
    origin_sel = f6.multiselect("Data Origin", origins)

    # Apply filters
    fdf = df.copy()
    if search_q.strip():
        fdf = fdf[fdf["defect_description"].str.contains(search_q.strip(), case=False, na=False)]
    if lot_sel:
        fdf = fdf[fdf["batch_id"].isin(lot_sel)]
    if cat_sel and "category" in fdf:
        fdf = fdf[fdf["category"].astype(str).isin(cat_sel)]
    if sev_sel and "severity" in fdf:
        fdf = fdf[fdf["severity"].astype(str).isin(sev_sel)]
    if grp_sel:
        fdf = fdf[fdf["ai_cluster"].isin(grp_sel)]
    if origin_sel and "data_origin" in fdf:
        fdf = fdf[fdf["data_origin"].astype(str).isin(origin_sel)]

    st.markdown(f"**{len(fdf):,} of {len(df):,} records match current filters:**")
    st.dataframe(fdf, use_container_width=True, hide_index=True, height=380)

    # Single report deep-dive
    if not fdf.empty:
        with st.expander("Single-Report Deep Dive"):
            sel_idx = st.selectbox(
                "Select a record",
                fdf.index.tolist(),
                format_func=lambda i: (
                    f"[{fdf.loc[i, 'report_id'] if 'report_id' in fdf else i+1}] "
                    f"{textwrap.shorten(str(fdf.loc[i, 'defect_description']), 70, placeholder='…')}"
                ),
            )
            item = fdf.loc[sel_idx]
            d1, d2, d3 = st.columns(3)
            d1.metric("Lot", str(item.get("batch_id", "—")))
            d2.metric("Category", str(item.get("category", "—")))
            d3.metric("Severity", str(item.get("severity", "Not assessed")))
            st.markdown(f"**Description:** {item['defect_description']}")
            cid = item.get("ai_cluster", 0)
            st.caption(
                f"AI Cluster: #{cid} — \"{labels.get(int(cid), 'Unknown')}\"  \n"
                f"Data Origin: {item.get('data_origin', 'unknown')}"
            )

    st.download_button(
        "⬇ Download Filtered Records (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="defectintel_filtered.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — DATA MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📁 Data Management":
    st.title("Data Management")
    st.markdown(
        "Switch between multi-industry demo presets, upload your own compatible defect CSV, "
        "or merge manually confirmed reports. Required columns: `defect_description`, `batch_id`."
    )

    # Pending reports banner
    n_pending = len(st.session_state["pending_reports"])
    if n_pending > 0:
        st.info(
            f"**{n_pending} pending confirmed report(s)** added via AI Investigator. "
            f"Click 'Re-run Analysis with Pending Reports' below to merge and recluster."
        )

    # 1-Click Preset Buttons
    st.markdown("### 1-Click Industry Presets")
    st.caption("Each dataset is clearly tagged as synthetic demo data.")
    pc = st.columns(4)
    preset_items = list(PRESET_DATASETS.items())
    icons = ["📱", "🚗", "🍱", "🧵"]
    for col, icon, (pname, ppath) in zip(pc, icons, preset_items):
        if col.button(f"{icon} {pname.split('(')[0].strip()}"):
            if ppath.exists():
                st.session_state["active_csv_bytes"] = ppath.read_bytes()
                st.session_state["active_data_name"] = pname
                st.session_state["pending_reports"] = []
                st.cache_data.clear()
                st.success(f"Loaded: {pname}")
                st.rerun()
            else:
                col.error("Dataset file not found.")

    st.markdown("---")

    # Custom CSV Upload
    st.markdown("### Upload Custom Defect CSV")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        up_bytes = uploaded.getvalue()
        try:
            raw_preview = pd.read_csv(BytesIO(up_bytes))
            validated = validate_dataset(raw_preview)
            st.success(f"Validation passed · {len(validated):,} usable records detected.")
            discarded = len(raw_preview) - len(validated)
            if discarded:
                st.warning(f"{discarded} row(s) with missing description or batch ID will be excluded.")
            st.dataframe(raw_preview.head(5), use_container_width=True, hide_index=True)
            if st.button("Analyse Uploaded Dataset", type="primary"):
                st.session_state["active_csv_bytes"] = up_bytes
                st.session_state["active_data_name"] = uploaded.name
                st.session_state["pending_reports"] = []
                st.cache_data.clear()
                st.success("Dataset activated. Clusters and alerts will update on next page load.")
                st.rerun()
        except Exception as e:
            st.error(f"Validation failed: {e}")

    st.markdown("---")

    # Re-run with pending reports
    st.markdown("### Confirmed Manual Reports")
    if n_pending > 0:
        st.write(f"**{n_pending} report(s) pending merge:**")
        pend_df = pd.DataFrame(st.session_state["pending_reports"])
        st.dataframe(pend_df, use_container_width=True, hide_index=True)
        if st.button("♻ Re-run Analysis with Pending Reports", type="primary"):
            base_df = pd.read_csv(BytesIO(st.session_state["active_csv_bytes"]))
            combined = pd.concat([base_df, pend_df], ignore_index=True)
            buf = BytesIO()
            combined.to_csv(buf, index=False)
            st.session_state["active_csv_bytes"] = buf.getvalue()
            st.session_state["active_data_name"] += " + confirmed"
            st.session_state["pending_reports"] = []
            st.cache_data.clear()
            st.success("Reports merged and analysis re-run.")
            st.rerun()
        if st.button("Discard Pending Reports"):
            st.session_state["pending_reports"] = []
            st.rerun()
    else:
        st.caption("No pending manually confirmed reports.")

    st.markdown("---")

    # Clustering Threshold Tuning
    st.markdown("### Clustering Sensitivity")
    st.markdown(
        "Lower values create **stricter, more specific** defect groups. "
        "Higher values create **broader** groups. This is a grouping-distance control, "
        "not a model accuracy setting."
    )
    new_thresh = st.slider(
        "Grouping Distance Threshold",
        min_value=0.20, max_value=0.85,
        value=float(st.session_state["distance_threshold"]),
        step=0.05,
    )
    if st.button("Update Clustering & Re-analyse"):
        st.session_state["distance_threshold"] = new_thresh
        st.cache_data.clear()
        st.success(f"Threshold set to {new_thresh:.2f}. Re-running analysis…")
        st.rerun()

    st.markdown("---")

    # Current dataset summary + download
    st.markdown("### Active Dataset Summary")
    st.write(f"**Name:** {st.session_state['active_data_name']}")
    st.write(f"**Validated reports:** {len(df):,}")
    st.write(f"**AI-discovered groups:** {df['ai_cluster'].nunique():,}")
    st.write(f"**Clustering distance threshold:** {active_threshold:.2f}")
    st.download_button(
        "⬇ Download Analysed Dataset (CSV with ai_cluster column)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="defectintel_analysed.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🧪 Model Evaluation":
    st.title("Model Evaluation & Metrics")
    st.markdown(
        "Transparent evaluation of the semantic embedding and clustering components. "
        "These metrics characterise how well the unsupervised AI has organised the defect reports. "
        "**They are not classification accuracy scores.**"
    )

    with st.spinner("Computing evaluation metrics…"):
        full_eval = compute_full_evaluation(df, embeddings)

    # ── Cluster Shape Stats ──
    st.markdown("### Cluster Shape Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("AI Groups Discovered", full_eval["n_clusters"])
    s2.metric("Reports Analysed", full_eval["n_samples"])
    s3.metric("Smallest Group", full_eval["min_cluster_size"])
    s4.metric("Largest Group", full_eval["max_cluster_size"])
    st.caption(f"Mean group size: {full_eval['mean_cluster_size']:.1f} reports")

    # Cluster size distribution histogram
    with st.container(border=True):
        st.markdown("#### Cluster Size Distribution")
        sizes = df["ai_cluster"].value_counts().reset_index()
        sizes.columns = ["Group ID", "Reports"]
        sizes["Group"] = sizes["Group ID"].map(labels)
        fig_sz = px.bar(
            sizes.sort_values("Reports", ascending=False),
            x="Group", y="Reports",
            color="Reports",
            color_continuous_scale=[[0, "#EDF6FC"], [1, "#14B8A6"]],
            hover_data={"Group": True, "Reports": True},
        )
        fig_sz.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
        st.plotly_chart(themed_bar(fig_sz, 300), use_container_width=True)

    # ── Silhouette Score ──
    st.markdown("### Silhouette Score (Internal Clustering Quality)")
    with st.expander("What is the Silhouette Score?"):
        st.markdown("""
        - Measures how well each report fits its assigned cluster vs neighbouring clusters.
        - **Range:** -1.0 (misclassified) to +1.0 (dense, well-separated clusters).
        - Values near 0 indicate overlapping clusters.
        - Computed using **cosine distance** to match the metric used by Agglomerative Clustering.
        - Higher is better. No ground truth labels are needed.
        - This does **not** convert to a classification accuracy percentage.
        """)
    sil = full_eval.get("silhouette")
    if sil is not None:
        sil_col, _ = st.columns([1, 2])
        sil_col.metric("Silhouette Score (cosine)", f"{sil:.3f}",
                        help="Internal cluster quality. Range −1 to +1.")
        if sil > 0.5:
            st.success(f"Score {sil:.3f}: Clusters are reasonably dense and separated.")
        elif sil > 0.2:
            st.warning(f"Score {sil:.3f}: Moderate separation. Some overlap between groups.")
        else:
            st.error(f"Score {sil:.3f}: Low separation. Consider adjusting the distance threshold.")
    else:
        st.info("Silhouette score cannot be computed: requires 2 or more clusters with fewer clusters than samples.")

    # ── ARI & NMI vs Reference Labels ──
    st.markdown("### Concordance with Reference Category Labels (Optional)")
    with st.expander("What are ARI and NMI?"):
        st.markdown("""
        - **ARI (Adjusted Rand Index):** Measures pair-wise cluster agreement, adjusted for chance.
          Range: −1 to +1. Perfect agreement = 1.0. Random = ~0.
        - **NMI (Normalised Mutual Information):** Information-theoretic agreement measure.
          Range: 0 to 1. Higher = more shared information between cluster assignments and labels.
        - These compare AI clusters against existing **reference labels** (e.g. `category` column).
        - Reference categories are coarse labels used **only for evaluation**, not as clustering input.
        - Broad reference categories may not exactly match fine-grained AI defect groups —
          a lower score does not necessarily mean incorrect clustering.
        - **Neither ARI nor NMI is a classification accuracy metric.**
        """)
    ari_nmi = full_eval.get("ari_nmi")
    if ari_nmi:
        ref_col = ari_nmi["reference_column"]
        st.caption(
            f"Evaluated against `{ref_col}` labels for {ari_nmi['reports_evaluated']} records."
        )
        am1, am2 = st.columns(2)
        am1.metric("Adjusted Rand Index (ARI)", f"{ari_nmi['adjusted_rand_index']:.3f}")
        am2.metric("Normalised Mutual Info (NMI)", f"{ari_nmi['normalized_mutual_info']:.3f}")

        # Crosstab heatmap
        with st.expander("Show Cluster vs Reference Label Crosstab"):
            ct = pd.crosstab(df["ai_cluster"], df[ref_col])
            ct.index = [f"Group {i} – {labels.get(int(i), '?')[:28]}" for i in ct.index]
            fig_ct = px.imshow(
                ct,
                color_continuous_scale="Teal",
                labels=dict(x="Reference Category", y="AI Cluster", color="Count"),
                title=f"AI Cluster vs {ref_col}",
                text_auto=True,
            )
            fig_ct.update_layout(height=max(300, 40 * len(ct)))
            st.plotly_chart(themed_bar(fig_ct, max(300, 40 * len(ct))), use_container_width=True)
    else:
        st.info(
            "No compatible reference label column (`category`, `product`) found in the active dataset. "
            "ARI and NMI are shown as 'Not evaluated'."
        )

    # ── Similarity Retrieval ──
    st.markdown("### Similarity Retrieval (Precision@5)")
    with st.expander("What is Precision@5?"):
        st.markdown("""
        - Measures whether the top-5 retrieved reports are genuinely relevant to a query.
        - Requires human-judged query-result pairs as ground truth.
        - Without verified ground truth, this metric cannot be computed honestly.
        - Reporting cosine similarity as Precision@5 would be misleading.
        """)
    st.info(
        "Precision@5 = **Not evaluated** for this dataset.  \n"
        "Ground-truth query-result judgements are not available. "
        "Providing a fabricated metric would misrepresent model performance."
    )

    # ── Component Role Explanation ──
    st.markdown("### Component Roles — What Does Each Part Do?")
    st.markdown("""
    | Component | Role | Inputs | Outputs |
    |---|---|---|---|
    | **SentenceTransformer `all-MiniLM-L6-v2`** | Semantic embedding | Raw text | 384-dim float vector |
    | **Agglomerative Clustering** | Unsupervised grouping | Embedding matrix | Cluster ID per report |
    | **Cosine Similarity** | Similarity ranking | Two embedding vectors | Scalar score 0–1 |
    | **Centroid Assignment (new reports)** | Heuristic assignment | New embedding + cluster centroids | Nearest cluster ID + similarity |
    | **Batch Analytics Rules** | Count-spike detection | Report counts per batch & group | Rule-based alert list |

    The embedding model and clustering algorithm perform distinct roles. ARI/NMI evaluate the clustering step
    against reference labels. The embedding model itself is a pretrained general-purpose encoder —
    no domain-specific fine-tuning has been performed on this dataset.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — VISUAL INSPECTION (Beta / Manual Log)
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔬 Visual Inspection (Beta)":
    st.title("Manual Visual Inspection Log (Beta)")
    st.info(
        "**Computer-vision automated defect detection is not implemented.** "
        "This module provides a structured manual inspection log form. "
        "The core DefectIntel platform operates on verified text-based defect reports."
    )
    img = st.file_uploader("Upload component photo (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if img:
        st.image(img, caption="Component under inspection", use_container_width=True)
    with st.form("vi_form"):
        vi1, vi2 = st.columns(2)
        vi_type = vi1.selectbox("Observed Anomaly", [
            "Surface Scratch", "Component Misalignment", "Solder Bridge",
            "Pin Deformity", "Coating Defect", "Seal Failure", "Other"
        ])
        vi_sev = vi2.selectbox("Assessed Severity", ["Low", "Medium", "High", "Critical"])
        vi_notes = st.text_area("Inspector Observations")
        vi_lot = st.text_input("Production Lot ID")
        vi_sub = st.form_submit_button("Record Inspection Entry", type="primary")
    if vi_sub:
        if vi_notes.strip() and vi_lot.strip():
            st.success(f"Entry recorded: {vi_type} ({vi_sev}) for lot {vi_lot}.")
        else:
            st.warning("Notes and Lot ID are required.")
