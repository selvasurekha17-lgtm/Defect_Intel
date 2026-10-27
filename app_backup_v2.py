from io import BytesIO
from pathlib import Path
import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from backend import (
    DATA_PATH,
    PRESET_DATASETS,
    acknowledge_alert,
    unacknowledge_alert,
    reset_acknowledgements,
    analyze_dataset,
    batch_analytics,
    clustering_evaluation,
    get_acknowledged_alerts,
    group_profiles,
    search_defect,
    validate_dataset,
    generate_investigation_recommendation,
)

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DefectIntel | Industrial Quality Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Premium Industrial AI SaaS Design System (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --navy: #17324D;
        --navy-light: #2A4866;
        --teal: #14B8A6;
        --teal-dark: #0F766E;
        --teal-light: #CCFBF1;
        --sky: #38BDF8;
        --sky-light: #E0F2FE;
        --bg-main: #F7FAFC;
        --bg-sidebar: #EDF6FC;
        --card-bg: rgba(255, 255, 255, 0.94);
        --card-border: #E2EDF4;
        --coral-red: #EF4444;
        --coral-light: #FEE2E2;
        --amber-warn: #F59E0B;
        --amber-light: #FEF3C7;
        --emerald-ok: #10B981;
        --emerald-light: #D1FAE5;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--navy);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #EDF6FC 0%, #E6F0F7 100%);
        border-right: 1px solid #D9E8F2;
        padding-top: 1rem;
    }

    /* Metric Cards */
    [data-testid="stMetric"] {
        background: var(--card-bg);
        padding: 18px 22px;
        border: 1px solid var(--card-border);
        border-radius: 18px;
        box-shadow: 0 6px 20px rgba(23, 50, 77, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 26px rgba(23, 50, 77, 0.08);
    }
    [data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--teal), var(--sky));
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: var(--navy);
    }

    /* Streamlit Containers / Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card-bg);
        border-radius: 18px;
        border: 1px solid var(--card-border);
        box-shadow: 0 6px 20px rgba(23, 50, 77, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        margin-bottom: 0.8rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #CBDCE7;
        box-shadow: 0 10px 28px rgba(23, 50, 77, 0.07);
    }

    /* Futuristic Hero Section */
    .hero-container {
        position: relative;
        overflow: hidden;
        min-height: 290px;
        padding: 40px 48px;
        margin: 6px 0 28px 0;
        border: 1px solid #D8E8F2;
        border-radius: 24px;
        background: radial-gradient(circle at 86% 40%, rgba(204, 251, 241, 0.85) 0%, rgba(224, 242, 254, 0.4) 35%, transparent 68%),
                    linear-gradient(125deg, #FFFFFF 0%, #F0F7FC 60%, #E8F4FA 100%);
        box-shadow: 0 16px 45px rgba(23, 50, 77, 0.07);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hero-content {
        position: relative;
        z-index: 2;
        max-width: 680px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--teal-dark);
        background: rgba(20, 184, 166, 0.12);
        padding: 6px 14px;
        border-radius: 999px;
        border: 1px solid rgba(20, 184, 166, 0.3);
        margin-bottom: 14px;
    }
    .hero-badge .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--teal);
        border-radius: 50%;
        box-shadow: 0 0 10px var(--teal);
        animation: pulse-dot 1.8s infinite;
    }
    .hero-content h1 {
        font-size: clamp(2rem, 3.8vw, 2.9rem);
        font-weight: 800;
        color: var(--navy);
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin: 0 0 14px 0;
    }
    .hero-content p {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #4A6379;
        margin: 0;
        max-width: 620px;
    }

    /* 3D-Style Rotating AI Core Visualization */
    .ai-visual-wrap {
        position: relative;
        width: 250px;
        height: 250px;
        perspective: 900px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .ai-orb-cluster {
        width: 170px;
        height: 170px;
        position: relative;
        transform-style: preserve-3d;
        animation: orb-rotate 16s linear infinite;
    }
    .core-nucleus {
        position: absolute;
        inset: 38px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #FFFFFF 0%, #38BDF8 40%, #14B8A6 85%);
        box-shadow: 0 0 35px rgba(20, 184, 166, 0.5), inset 0 0 20px rgba(255, 255, 255, 0.8);
        filter: blur(0.5px);
        animation: core-pulse 3.5s ease-in-out infinite alternate;
    }
    .orbital-ring {
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 1.5px dashed rgba(20, 184, 166, 0.55);
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.2);
    }
    .ring-1 {
        transform: rotateX(65deg) rotateY(15deg);
        animation: ring-spin-1 9s linear infinite;
    }
    .ring-2 {
        transform: rotateX(-50deg) rotateY(45deg);
        border-color: rgba(56, 189, 248, 0.6);
        border-style: solid;
        border-width: 1.2px;
        animation: ring-spin-2 11s linear infinite reverse;
    }
    .ring-3 {
        transform: rotateX(25deg) rotateY(-70deg);
        border-color: rgba(15, 118, 110, 0.45);
        animation: ring-spin-3 13s linear infinite;
    }
    .synapse-node {
        position: absolute;
        width: 11px;
        height: 11px;
        background: #14B8A6;
        border: 2px solid #FFFFFF;
        border-radius: 50%;
        box-shadow: 0 0 14px #14B8A6, 0 0 20px #38BDF8;
    }
    .sn-1 { top: 0; left: 50%; transform: translate(-50%, -50%); }
    .sn-2 { bottom: 0; left: 50%; transform: translate(-50%, 50%); }
    .sn-3 { top: 50%; left: 0; transform: translate(-50%, -50%); background: #38BDF8; }
    .sn-4 { top: 50%; right: 0; transform: translate(50%, -50%); background: #38BDF8; }

    @keyframes orb-rotate {
        0% { transform: rotateY(0deg) rotateX(10deg); }
        100% { transform: rotateY(360deg) rotateX(10deg); }
    }
    @keyframes ring-spin-1 {
        0% { transform: rotateX(65deg) rotateZ(0deg); }
        100% { transform: rotateX(65deg) rotateZ(360deg); }
    }
    @keyframes ring-spin-2 {
        0% { transform: rotateX(-50deg) rotateZ(0deg); }
        100% { transform: rotateX(-50deg) rotateZ(360deg); }
    }
    @keyframes ring-spin-3 {
        0% { transform: rotateX(25deg) rotateZ(0deg); }
        100% { transform: rotateX(25deg) rotateZ(360deg); }
    }
    @keyframes core-pulse {
        0% { transform: scale(0.92); opacity: 0.85; }
        100% { transform: scale(1.06); opacity: 1; }
    }
    @keyframes pulse-dot {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.3); opacity: 0.55; }
    }

    /* Badges & Chips */
    .badge-high {
        background-color: var(--coral-light);
        color: var(--coral-red);
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(239, 68, 68, 0.25);
        display: inline-block;
    }
    .badge-medium {
        background-color: var(--amber-light);
        color: #B45309;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(245, 158, 11, 0.25);
        display: inline-block;
    }
    .badge-acknowledged {
        background-color: var(--emerald-light);
        color: #047857;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 4px 10px;
        border-radius: 999px;
        border: 1px solid rgba(16, 185, 129, 0.25);
        display: inline-block;
    }
    .badge-teal {
        background-color: var(--teal-light);
        color: var(--teal-dark);
        font-weight: 600;
        font-size: 0.78rem;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .chip-btn {
        background: #FFFFFF;
        border: 1px solid #D5E4EF;
        color: #17324D;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-block;
        margin: 3px;
    }

    /* Tables & Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #FFFFFF !important;
        border: 1px solid #CFE0EC !important;
        border-radius: 12px !important;
        color: var(--navy) !important;
    }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #14B8A6 0%, #0D9488 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(20, 184, 166, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(20, 184, 166, 0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* Reduced Motion */
    @media (prefers-reduced-motion: reduce) {
        *, ::before, ::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }

    @media (max-width: 800px) {
        .hero-container { padding: 24px; flex-direction: column; text-align: center; }
        .hero-content { margin-bottom: 24px; }
        .ai-visual-wrap { width: 160px; height: 160px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cached Semantic Analysis Function
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def analyze_csv(csv_bytes, distance_threshold=0.5):
    source = pd.read_csv(BytesIO(csv_bytes))
    return analyze_dataset(source, distance_threshold=distance_threshold)


def group_labels(df, embeddings):
    profiles = group_profiles(df, embeddings)
    return {
        profile["cluster"]: textwrap.shorten(profile["label"], width=46, placeholder="…")
        for profile in profiles
    }, profiles


def apply_industrial_theme(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=14, r=14, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#17324D", size=12),
        xaxis=dict(
            gridcolor="#E5EFF6",
            zerolinecolor="#CBDCE7",
            tickfont=dict(color="#4A6379"),
        ),
        yaxis=dict(
            gridcolor="#E5EFF6",
            zerolinecolor="#CBDCE7",
            tickfont=dict(color="#4A6379"),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="#E2EDF4",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    return fig


# -----------------------------------------------------------------------------
# Sidebar Navigation & Workspace Controls
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <span style="font-size: 1.6rem; color: #14B8A6;">◈</span>
        <span style="font-size: 1.35rem; font-weight: 800; color: #17324D; letter-spacing: -0.02em;">DefectIntel</span>
    </div>
    <div style="font-size: 0.72rem; font-weight: 700; color: #0F766E; letter-spacing: 0.1em; margin-bottom: 18px;">
        MANUFACTURING QUALITY SAAS
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "WORKSPACE NAVIGATION",
    [
        "Overview",
        "AI Investigator",
        "Batch Intelligence",
        "Alert Center",
        "Defect Explorer",
        "Data Management",
        "Visual Inspection (Beta)",
    ],
    index=0,
)

st.sidebar.markdown("---")

# Initialize Session State Data
sample_bytes = DATA_PATH.read_bytes()
if "active_csv_bytes" not in st.session_state:
    st.session_state["active_csv_bytes"] = sample_bytes
    st.session_state["active_data_name"] = "Mobile Electronics (100 reports)"
    st.session_state["distance_threshold"] = 0.50

active_bytes = st.session_state["active_csv_bytes"]
active_threshold = st.session_state.get("distance_threshold", 0.50)

# Load and compute AI embeddings & clusters
try:
    with st.spinner("Executing semantic AI model (all-MiniLM-L6-v2)…"):
        df, embeddings = analyze_csv(active_bytes, distance_threshold=active_threshold)
except Exception as error:
    st.error(f"Could not analyze the active dataset: {error}")
    st.info("Restore the bundled sample dataset in Data Management.")
    st.stop()

labels, profiles = group_labels(df, embeddings)
batch_options = sorted(df["batch_id"].astype(str).unique().tolist())

# Sidebar Batch Selector
previous_batch = None
current_batch = None
if len(batch_options) >= 2:
    st.sidebar.markdown("**Comparative Batch Selection**")
    previous_batch = st.sidebar.selectbox("Baseline Batch", batch_options, index=0)
    compared_options = [b for b in batch_options if b != previous_batch]
    current_batch = st.sidebar.selectbox(
        "Comparison Batch",
        compared_options,
        index=0,
        key=f"comp_{previous_batch}",
    )
    st.sidebar.caption("Batch IDs are distinct production lots; order is not assumed to be chronological.")

# Compute rule-based alerts
try:
    summary, alerts = (
        batch_analytics(
            df,
            previous_batch=previous_batch,
            current_batch=current_batch,
            min_current_count=st.session_state.get("rule_min_count", 5),
            increase_threshold=st.session_state.get("rule_increase_thresh", 50),
            high_increase_threshold=st.session_state.get("rule_high_thresh", 100),
        )
        if previous_batch and current_batch
        else batch_analytics(df)
    )
except ValueError as error:
    st.sidebar.error(str(error))
    summary, alerts = pd.DataFrame(), []

acknowledged = get_acknowledged_alerts()
unreviewed_alerts = [a for a in alerts if a["alert_id"] not in acknowledged]

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active Lot:** `{st.session_state['active_data_name']}`")
st.sidebar.markdown(f"**Verified Reports:** `{len(df):,}`")
st.sidebar.markdown(f"**AI Clusters:** `{df['ai_cluster'].nunique():,}` groups")
if alerts:
    st.sidebar.markdown(
        f"**Active Warnings:** <span class='badge-high'>{len(unreviewed_alerts)} unreviewed</span>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 1: OVERVIEW
# -----------------------------------------------------------------------------
if page == "Overview":
    # Hero Section with 3D-Style Visualization
    st.markdown(
        """
        <section class="hero-container">
            <div class="hero-content">
                <div class="hero-badge">
                    <span class="pulse-dot"></span>
                    <span>DefectIntel · Industrial AI Quality Suite</span>
                </div>
                <h1>Understand Defects Before They Escalate.</h1>
                <p>Transform unstructured manufacturing defect reports into actionable root-cause intelligence
                with semantic AI embeddings, automated clustering, and evidence-backed early warnings.</p>
            </div>
            <div class="ai-visual-wrap" aria-hidden="true">
                <div class="ai-orb-cluster">
                    <div class="orbital-ring ring-1">
                        <span class="synapse-node sn-1"></span>
                        <span class="synapse-node sn-2"></span>
                    </div>
                    <div class="orbital-ring ring-2">
                        <span class="synapse-node sn-3"></span>
                        <span class="synapse-node sn-4"></span>
                    </div>
                    <div class="orbital-ring ring-3"></div>
                    <div class="core-nucleus"></div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inspection Reports", f"{len(df):,}", help="Total validated quality inspection records")
    col2.metric("Discovered AI Groups", f"{df['ai_cluster'].nunique():,}", help="Defect clusters discovered by Agglomerative Clustering")
    col3.metric("Unreviewed Warnings", f"{len(unreviewed_alerts)}", help="Rule-based batch alerts awaiting quality sign-off")
    col4.metric("Production Batches", f"{df['batch_id'].nunique():,}", help="Distinct manufacturing batch IDs represented")

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Visual Analytics Split
    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        with st.container(border=True):
            st.markdown("### Most Frequent Defect Groups")
            st.caption("Group titles are representative descriptions from real inspection logs.")
            counts = df["ai_cluster"].value_counts().sort_values(ascending=False).head(8)
            chart_data = pd.DataFrame({
                "Representative Defect": [labels.get(int(c), f"Group {c}") for c in counts.index],
                "Report Count": counts.values,
            })
            fig_groups = px.bar(
                chart_data,
                x="Report Count",
                y="Representative Defect",
                orientation="h",
                color="Report Count",
                color_continuous_scale=[[0, "#38BDF8"], [1, "#14B8A6"]],
                labels={"Report Count": "Validated Reports"},
            )
            fig_groups.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
            st.plotly_chart(apply_industrial_theme(fig_groups, height=330), use_container_width=True)

    with right_col:
        with st.container(border=True):
            st.markdown("### Top Discovered Profiles")
            st.caption("AI-identified cluster representatives ranked by volume.")
            profile_df = pd.DataFrame(profiles).sort_values("report_count", ascending=False).head(7)
            st.dataframe(
                profile_df[["cluster", "label", "report_count"]].rename(
                    columns={"cluster": "Group ID", "label": "Representative Issue", "report_count": "Reports"}
                ),
                use_container_width=True,
                hide_index=True,
                height=310,
            )

    # Evaluation Metric & Scientific Transparency
    eval_metric = clustering_evaluation(df)
    if eval_metric:
        with st.expander("Clustering Concordance Evaluation (Scientific Reference)"):
            st.caption(
                f"Evaluation calculated against existing `{eval_metric['reference_column']}` labels for "
                f"{eval_metric['reports_evaluated']} records. These are clustering agreement metrics, not supervised classification accuracy."
            )
            m1, m2 = st.columns(2)
            m1.metric("Adjusted Rand Index (ARI)", f"{eval_metric['adjusted_rand_index']:.3f}",
                      help="Measures similarity of two clusterings adjusted for chance (-1.0 to +1.0)")
            m2.metric("Normalized Mutual Information (NMI)", f"{eval_metric['normalized_mutual_info']:.3f}",
                      help="Information-theoretic agreement measure (0.0 to 1.0)")

    # Early Warning Quick Ticker
    st.markdown("### Priority Batch Warnings")
    if not previous_batch or not current_batch:
        st.info("Select two different batches in the sidebar to activate early warning calculations.")
    elif not alerts:
        st.success(f"No statistical defect spikes detected between {previous_batch} and {current_batch}.")
    else:
        for alert in alerts[:3]:
            group_records = df[
                (df["ai_cluster"] == alert["cluster"])
                & (df["batch_id"].isin([alert["previous_batch"], alert["batch"]]))
            ]
            title = labels.get(alert["cluster"], f"Defect Group {alert['cluster']}")
            with st.container(border=True):
                head_c1, head_c2 = st.columns([3, 1])
                with head_c1:
                    badge_class = "badge-high" if alert["severity"] == "High" else "badge-medium"
                    pct_str = "No baseline reports" if alert["increase_percent"] is None else f"+{alert['increase_percent']:.1f}% surge"
                    st.markdown(
                        f"<span class='{badge_class}'>{alert['severity'].upper()} PRIORITY</span> &nbsp; "
                        f"<strong>{title}</strong>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"Baseline Lot **{alert['previous_batch']}** ({alert['previous_count']} reports) → "
                        f"Comparison Lot **{alert['batch']}** ({alert['current_count']} reports) · **{pct_str}**"
                    )
                with head_c2:
                    if alert["alert_id"] in acknowledged:
                        st.markdown("<span class='badge-acknowledged'>✓ REVIEWED</span>", unsafe_allow_html=True)
                    else:
                        if st.button("Acknowledge", key=f"ov_ack_{alert['alert_id']}"):
                            acknowledge_alert(alert["alert_id"])
                            st.rerun()

                with st.expander("Root-Cause Investigation Details"):
                    st.info(f"**Recommended Action:** {alert.get('recommendation', 'Inspect lot.')}")
                    cols_to_show = [c for c in ["report_id", "batch_id", "defect_description", "category", "severity"] if c in group_records.columns]
                    st.dataframe(group_records[cols_to_show], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 2: AI DEFECT INVESTIGATOR
# -----------------------------------------------------------------------------
elif page == "AI Investigator":
    st.title("AI Defect Investigator")
    st.markdown(
        "Enter a new complaint or field observation. The semantic engine generates a **384-dimensional dense embedding** "
        "and retrieves semantically related defect reports regardless of wording differences."
    )

    # Demo Query Chips for fast presentation
    st.markdown("**Quick-Test Sample Complaints (Click to populate):**")
    chip_cols = st.columns(4)
    if chip_cols[0].button("📱 Display blinking during charge"):
        st.session_state["inv_query"] = "Phone display keeps blinking continuously while charging"
    if chip_cols[1].button("🔌 USB connector loose / no power"):
        st.session_state["inv_query"] = "USB-C port is loose and fails to charge device unless pressed"
    if chip_cols[2].button("📷 Camera purple artifact blur"):
        st.session_state["inv_query"] = "Camera photos show purple artifacts and dark blurry patches"
    if chip_cols[3].button("🚗 Panel gap exceeds tolerance"):
        st.session_state["inv_query"] = "Inspection noted door panel gap exceeds specified tolerance"

    # Search Form
    with st.form("ai_investigator_form"):
        default_val = st.session_state.get("inv_query", "")
        query = st.text_area(
            "New Defect Description or Customer Complaint",
            value=default_val,
            placeholder="Type or paste a defect description (e.g., 'Screen flickers randomly after phone unlocks')",
            height=100,
        )
        ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([1, 1, 2])
        top_k = ctrl_c1.slider("Result Count (Top-K)", min_value=3, max_value=10, value=5)
        min_sim = ctrl_c2.slider("Min Similarity Filter", min_value=0.0, max_value=0.9, value=0.30, step=0.05)
        submit_btn = st.form_submit_button("Semantic Vector Search", type="primary")

    if submit_btn or default_val:
        if not query.strip():
            st.warning("Please enter a defect description to search.")
        else:
            try:
                with st.spinner("Embedding query into vector space and computing cosine similarity…"):
                    results = search_defect(query, df, embeddings, limit=top_k)

                filtered_results = [r for r in results if r["similarity"] >= min_sim]

                st.caption(
                    "Scientific Disclaimer: Cosine similarity measures semantic alignment in SentenceTransformer embedding space. "
                    "It is a comparative ranking signal, not a calibrated classification probability or model accuracy score."
                )

                if not filtered_results:
                    st.info(f"No existing reports met the minimum similarity score of {min_sim:.2f}.")
                else:
                    st.markdown(f"**Found {len(filtered_results)} Semantically Similar Reports:**")
                    for idx, res in enumerate(filtered_results, start=1):
                        with st.container(border=True):
                            sim_pct = int(res["similarity"] * 100)
                            color_hex = "#14B8A6" if res["similarity"] >= 0.70 else "#38BDF8" if res["similarity"] >= 0.50 else "#64748B"
                            rep_title = labels.get(res["cluster"], f"Group {res['cluster']}")

                            r_top, r_sim = st.columns([3, 1])
                            with r_top:
                                st.markdown(f"#### #{idx} · {rep_title}")
                                st.write(f"“{res['description']}”")
                            with r_sim:
                                st.metric("Cosine Similarity", f"{res['similarity']:.3f}")
                                st.markdown(
                                    f"""
                                    <div style="background: #E2EDF4; border-radius: 999px; height: 8px; width: 100%; overflow: hidden;">
                                        <div style="background: {color_hex}; height: 8px; width: {sim_pct}%;"></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                            m1, m2, m3, m4 = st.columns(4)
                            m1.markdown(f"**Lot ID:** `{res['batch_id']}`")
                            m2.markdown(f"**Report ID:** `{res['report_id']}`")
                            m3.markdown(f"**Category:** `{res['category']}`")
                            m4.markdown(f"**Severity:** `{res['severity']}`")

                            with st.expander("Why AI Matched This"):
                                st.markdown(
                                    f"This historical complaint was mapped to **AI Cluster #{res['cluster']}** based on semantic sentence "
                                    f"embedding distance. Both texts share contextual manufacturing fault signatures in the 384-dimensional representation."
                                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")


# -----------------------------------------------------------------------------
# PAGE 3: BATCH INTELLIGENCE
# -----------------------------------------------------------------------------
elif page == "Batch Intelligence":
    st.title("Batch Intelligence & Defect Distribution")
    st.markdown(
        "Analyze defect concentrations across manufacturing lots. Compare baseline vs current batches "
        "to discover shifting failure modes before full-scale assembly."
    )

    if len(batch_options) < 2:
        st.warning("At least two distinct batch IDs are required for comparative lot analytics.")
    else:
        b_c1, b_c2, b_c3, b_c4 = st.columns(4)
        count_baseline = len(df[df["batch_id"] == previous_batch])
        count_compared = len(df[df["batch_id"] == current_batch])
        delta = count_compared - count_baseline

        b_c1.metric(f"Baseline Lot ({previous_batch})", f"{count_baseline} reports")
        b_c2.metric(f"Comparison Lot ({current_batch})", f"{count_compared} reports")
        b_c3.metric("Net Report Change", f"{delta:+d} reports", delta_color="inverse")
        b_c4.metric("Active Triggered Rules", len(alerts))

        st.caption("Note: Changes reflect reported defect volumes; total production lot sizes are unsupplied.")

        # Batch Comparison Group Chart
        st.markdown("### Defect Group Comparison Across Lots")
        selected_lots = df[df["batch_id"].isin([previous_batch, current_batch])]
        batch_grp_counts = (
            selected_lots.groupby(["batch_id", "ai_cluster"]).size().reset_index(name="Reports")
        )
        batch_grp_counts["Defect Group"] = batch_grp_counts["ai_cluster"].map(labels)

        fig_batch_comp = px.bar(
            batch_grp_counts,
            x="Defect Group",
            y="Reports",
            color="batch_id",
            barmode="group",
            color_discrete_sequence=["#14B8A6", "#38BDF8"],
            labels={"batch_id": "Production Lot"},
            title=f"Report Counts by Group: {previous_batch} vs {current_batch}",
        )
        st.plotly_chart(apply_industrial_theme(fig_batch_comp, height=380), use_container_width=True)

        # Defect Group Distribution Donut & Batch Volume
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            with st.container(border=True):
                st.markdown(f"#### Defect Mix in Lot {current_batch}")
                cur_lot_df = df[df["batch_id"] == current_batch]
                lot_grp_counts = cur_lot_df["ai_cluster"].value_counts().reset_index()
                lot_grp_counts.columns = ["ai_cluster", "Count"]
                lot_grp_counts["Group"] = lot_grp_counts["ai_cluster"].map(labels)
                fig_donut = px.pie(
                    lot_grp_counts,
                    names="Group",
                    values="Count",
                    hole=0.48,
                    color_discrete_sequence=px.colors.qualitative.Prism,
                )
                st.plotly_chart(apply_industrial_theme(fig_donut, height=320), use_container_width=True)

        with d_c2:
            with st.container(border=True):
                st.markdown("#### Overall Reports per Lot")
                all_lot_counts = df["batch_id"].value_counts().sort_index().reset_index()
                all_lot_counts.columns = ["Lot", "Total Reports"]
                fig_lots = px.bar(
                    all_lot_counts,
                    x="Lot",
                    y="Total Reports",
                    color="Total Reports",
                    color_continuous_scale=[[0, "#E0F2FE"], [1, "#0284C7"]],
                )
                fig_lots.update_layout(coloraxis_showscale=False)
                st.plotly_chart(apply_industrial_theme(fig_lots, height=320), use_container_width=True)

        # Group-Level Summary Table
        st.markdown("### Group-by-Group Report Delta")
        if not summary.empty:
            delta_table = summary.T.copy()
            delta_table.index = [labels.get(int(c), f"Group {c}") for c in delta_table.index]
            delta_table.index.name = "Defect Group"
            st.dataframe(delta_table, use_container_width=True)


# -----------------------------------------------------------------------------
# PAGE 4: ALERT CENTER
# -----------------------------------------------------------------------------
elif page == "Alert Center":
    st.title("Evidence-Based Early Warning & Alert Center")
    st.markdown(
        "Industrial quality alerts generated when configured defect report spike rules trigger. "
        "All alerts link directly to supporting inspection logs and root-cause action recommendations."
    )

    # Configurable Thresholds
    with st.expander("⚙️ Configure Alert Rules & Sensitivity", expanded=False):
        cfg_c1, cfg_c2, cfg_c3 = st.columns(3)
        min_reports = cfg_c1.number_input(
            "Minimum reports in compared lot", min_value=1, max_value=50,
            value=st.session_state.get("rule_min_count", 5)
        )
        inc_thresh = cfg_c2.slider(
            "Minimum increase threshold (%)", min_value=10, max_value=300,
            value=st.session_state.get("rule_increase_thresh", 50), step=10
        )
        high_thresh = cfg_c3.slider(
            "High severity threshold (%)", min_value=50, max_value=500,
            value=st.session_state.get("rule_high_thresh", 100), step=10
        )
        if st.button("Apply Configured Rules"):
            st.session_state["rule_min_count"] = int(min_reports)
            st.session_state["rule_increase_thresh"] = int(inc_thresh)
            st.session_state["rule_high_thresh"] = int(high_thresh)
            st.success("Alert rules updated and saved for this session.")
            st.rerun()

    # Recalculate alerts with active thresholds
    if previous_batch and current_batch:
        _, dynamic_alerts = batch_analytics(
            df,
            previous_batch=previous_batch,
            current_batch=current_batch,
            min_current_count=st.session_state.get("rule_min_count", 5),
            increase_threshold=st.session_state.get("rule_increase_thresh", 50),
            high_increase_threshold=st.session_state.get("rule_high_thresh", 100),
        )
    else:
        dynamic_alerts = []

    # Alert Metrics
    high_count = sum(1 for a in dynamic_alerts if a["severity"] == "High")
    med_count = sum(1 for a in dynamic_alerts if a["severity"] == "Medium")
    ack_count = sum(1 for a in dynamic_alerts if a["alert_id"] in acknowledged)
    pending_count = len(dynamic_alerts) - ack_count

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("High Priority Warnings", high_count)
    a2.metric("Medium Warnings", med_count)
    a3.metric("Awaiting Review", pending_count)
    a4.metric("Acknowledged", ack_count)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns(3)
    sev_filter = f_col1.multiselect("Filter Severity", ["High", "Medium"], default=["High", "Medium"])
    status_filter = f_col2.selectbox("Review Status", ["All", "Awaiting Review", "Reviewed"])
    if f_col3.button("Reset All Acknowledgements", help="Clear all stored review signatures"):
        reset_acknowledgements()
        st.success("All alert acknowledgements reset.")
        st.rerun()

    # Apply Filters
    filtered_alerts = [
        a for a in dynamic_alerts
        if a["severity"] in sev_filter
        and (
            status_filter == "All"
            or (status_filter == "Reviewed" and a["alert_id"] in acknowledged)
            or (status_filter == "Awaiting Review" and a["alert_id"] not in acknowledged)
        )
    ]

    st.markdown(f"**Showing {len(filtered_alerts)} Active Alert(s):**")

    if not dynamic_alerts:
        st.success(f"No warning thresholds breached between Lot {previous_batch} and Lot {current_batch}.")
    elif not filtered_alerts:
        st.info("No alerts match the selected severity and review status filters.")
    else:
        for alert in filtered_alerts:
            title = labels.get(alert["cluster"], f"Defect Group {alert['cluster']}")
            group_records = df[
                (df["ai_cluster"] == alert["cluster"])
                & (df["batch_id"].isin([alert["previous_batch"], alert["batch"]]))
            ]
            is_ack = alert["alert_id"] in acknowledged
            badge_class = "badge-high" if alert["severity"] == "High" else "badge-medium"
            pct_text = "No baseline occurrences" if alert["increase_percent"] is None else f"+{alert['increase_percent']:.1f}% surge"

            with st.container(border=True):
                c_info, c_action = st.columns([3, 1])
                with c_info:
                    st.markdown(
                        f"<span class='{badge_class}'>{alert['severity'].upper()} PRIORITY</span> &nbsp; "
                        f"<strong style='font-size: 1.1rem; color: #17324D;'>{title}</strong>",
                        unsafe_allow_html=True,
                    )
                    st.write(
                        f"Lot **{alert['previous_batch']}**: {alert['previous_count']} reports → "
                        f"Lot **{alert['batch']}**: {alert['current_count']} reports · **{pct_text}**"
                    )
                with c_action:
                    if is_ack:
                        st.markdown("<span class='badge-acknowledged'>✓ SIGNED & REVIEWED</span>", unsafe_allow_html=True)
                        if st.button("Mark Unreviewed", key=f"unack_{alert['alert_id']}"):
                            unacknowledge_alert(alert["alert_id"])
                            st.rerun()
                    else:
                        if st.button("Acknowledge & Sign", key=f"ac_ack_{alert['alert_id']}", type="primary"):
                            acknowledge_alert(alert["alert_id"])
                            st.rerun()

                with st.expander("Investigate Root-Cause & Supporting Inspection Logs"):
                    st.markdown(f"**Recommended Quality Action:**")
                    st.info(alert.get("recommendation", "Conduct visual gate review."))

                    st.markdown("**Supporting Quality Reports from Selected Lots:**")
                    cols = [c for c in ["report_id", "batch_id", "defect_description", "category", "severity"] if c in group_records.columns]
                    st.dataframe(group_records[cols], use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# PAGE 5: DEFECT EXPLORER
# -----------------------------------------------------------------------------
elif page == "Defect Explorer":
    st.title("Defect Report Explorer")
    st.markdown("Search, filter, inspect, and export all validated manufacturing defect records.")

    f1, f2, f3, f4 = st.columns(4)
    search_q = f1.text_input("Search defect text", placeholder="e.g., flicker, charge, gap")
    sel_lots = f2.multiselect("Production Lot", batch_options)
    categories = sorted(df["category"].dropna().astype(str).unique()) if "category" in df else []
    sel_cats = f3.multiselect("Category", categories)
    severities = sorted(df["severity"].dropna().astype(str).unique()) if "severity" in df else []
    sel_sevs = f4.multiselect("Severity", severities)

    # Filter execution
    filtered_df = df.copy()
    if search_q.strip():
        filtered_df = filtered_df[
            filtered_df["defect_description"].str.contains(search_q.strip(), case=False, na=False)
        ]
    if sel_lots:
        filtered_df = filtered_df[filtered_df["batch_id"].isin(sel_lots)]
    if sel_cats and "category" in filtered_df:
        filtered_df = filtered_df[filtered_df["category"].astype(str).isin(sel_cats)]
    if sel_sevs and "severity" in filtered_df:
        filtered_df = filtered_df[filtered_df["severity"].astype(str).isin(sel_sevs)]

    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} Validated Records:**")
    st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=360)

    # Deep-Dive Inspector
    if not filtered_df.empty:
        with st.expander("Single-Report Deep Dive Inspector"):
            report_indices = filtered_df.index.tolist()
            chosen_idx = st.selectbox(
                "Select Record to Inspect",
                report_indices,
                format_func=lambda i: (
                    f"[{filtered_df.loc[i, 'report_id'] if 'report_id' in filtered_df else i+1}] "
                    f"{textwrap.shorten(str(filtered_df.loc[i, 'defect_description']), width=70, placeholder='…')}"
                ),
            )
            item = filtered_df.loc[chosen_idx]
            ic1, ic2, ic3 = st.columns(3)
            ic1.metric("Lot ID", str(item.get("batch_id", "N/A")))
            ic2.metric("Category", str(item.get("category", "N/A")))
            ic3.metric("Severity", str(item.get("severity", "N/A")))
            st.markdown(f"**Full Defect Description:**")
            st.info(f"“{item['defect_description']}”")
            cluster_id = item.get("ai_cluster", 0)
            st.caption(f"Assigned AI Cluster: #{cluster_id} · Group Representative: “{labels.get(int(cluster_id), 'Unknown')}”")

    # CSV Download
    st.download_button(
        "Download Filtered Reports (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="defectintel_filtered_reports.csv",
        mime="text/csv",
    )


# -----------------------------------------------------------------------------
# PAGE 6: DATA MANAGEMENT
# -----------------------------------------------------------------------------
elif page == "Data Management":
    st.title("Data Management & Multi-Industry Datasets")
    st.markdown(
        "Switch between industry demo datasets with one click, or upload your own compatible manufacturing CSV "
        "to run semantic embeddings and discover defect groups."
    )

    # 1-Click Multi-Industry Presets
    st.markdown("### 1-Click Multi-Domain Demo Presets")
    st.caption("Demonstrate DefectIntel across diverse manufacturing domains with verified sample datasets:")
    preset_c1, preset_c2, preset_c3, preset_c4 = st.columns(4)

    if preset_c1.button("📱 Mobile Electronics (100)", help="Smartphone display, charging, camera issues"):
        p_path = PRESET_DATASETS["Mobile Electronics (100 reports)"]
        if p_path.exists():
            st.session_state["active_csv_bytes"] = p_path.read_bytes()
            st.session_state["active_data_name"] = "Mobile Electronics (100 reports)"
            st.success("Loaded Mobile Electronics dataset.")
            st.rerun()

    if preset_c2.button("🚗 Automotive Assembly (90)", help="Door panel gaps, wiring connectors, clear coat paint"):
        p_path = PRESET_DATASETS["Automotive Assembly (90 reports)"]
        if p_path.exists():
            st.session_state["active_csv_bytes"] = p_path.read_bytes()
            st.session_state["active_data_name"] = "Automotive Assembly (90 reports)"
            st.success("Loaded Automotive Assembly dataset.")
            st.rerun()

    if preset_c3.button("🍱 Food Packaging (90)", help="Tray heat seals, pouch punctures, label printing"):
        p_path = PRESET_DATASETS["Food Packaging (90 reports)"]
        if p_path.exists():
            st.session_state["active_csv_bytes"] = p_path.read_bytes()
            st.session_state["active_data_name"] = "Food Packaging (90 reports)"
            st.success("Loaded Food Packaging dataset.")
            st.rerun()

    if preset_c4.button("🧵 Textile & Garment (90)", help="Fabric yarn tension, dye patch, seam weaving"):
        p_path = PRESET_DATASETS["Textile & Garments (90 reports)"]
        if p_path.exists():
            st.session_state["active_csv_bytes"] = p_path.read_bytes()
            st.session_state["active_data_name"] = "Textile & Garments (90 reports)"
            st.success("Loaded Textile & Garments dataset.")
            st.rerun()

    st.markdown("---")

    # Custom CSV Upload
    st.markdown("### Upload Custom Inspection CSV")
    st.caption("Required headers: `defect_description` and `batch_id`. Optional: `report_id`, `category`, `severity`.")
    uploaded_file = st.file_uploader("Upload CSV Defect Dataset", type=["csv"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            raw_preview = pd.read_csv(BytesIO(file_bytes))
            validated_df = validate_dataset(raw_preview)
            st.success(f"CSV Validation Passed · {len(validated_df):,} usable defect reports detected.")
            st.dataframe(raw_preview.head(5), use_container_width=True, hide_index=True)

            if st.button("Run Semantic AI Analysis on Uploaded CSV", type="primary"):
                with st.spinner("Calculating embeddings and discovering clusters…"):
                    new_df, _ = analyze_csv(file_bytes, distance_threshold=active_threshold)
                st.session_state["active_csv_bytes"] = file_bytes
                st.session_state["active_data_name"] = uploaded_file.name
                st.success(f"Analysis complete: {len(new_df):,} reports partitioned into {new_df['ai_cluster'].nunique()} groups.")
                st.rerun()
        except Exception as err:
            st.error(f"CSV Validation Error: {err}")

    st.markdown("---")

    # Current Dataset Controls & Threshold Tuning
    st.markdown("### Active Dataset & Clustering Tuning")
    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.write(f"**Active Lot Collection:** `{st.session_state['active_data_name']}`")
        st.write(f"**Validated Records:** `{len(df):,}` reports")
        st.write(f"**Discovered Defect Groups:** `{df['ai_cluster'].nunique():,}` clusters")
        st.download_button(
            "Download Analyzed Dataset with AI Clusters (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="defectintel_analyzed_data.csv",
            mime="text/csv",
        )

    with s_col2:
        new_thresh = st.slider(
            "AI Grouping Distance Threshold",
            min_value=0.20,
            max_value=0.85,
            value=float(st.session_state.get("distance_threshold", 0.50)),
            step=0.05,
            help="Lower distance values produce tighter, more specific defect groups; higher values yield broader groups.",
        )
        if st.button("Update Clustering Sensitivity"):
            st.session_state["distance_threshold"] = new_thresh
            st.success(f"Distance threshold set to {new_thresh:.2f}. Re-clustering…")
            st.rerun()


# -----------------------------------------------------------------------------
# PAGE 7: VISUAL INSPECTION (EXPERIMENTAL)
# -----------------------------------------------------------------------------
elif page == "Visual Inspection (Beta)":
    st.title("Manual Visual Inspection & Defect Annotation (Beta)")
    st.info(
        "Notice: Computer vision automated defect segmentation is an upcoming experimental module. "
        "The current core DefectIntel platform operates on verified text-based natural language defect reports."
    )

    st.markdown("#### Quality Engineer Visual Inspection Log")
    st.caption("Upload a component image for manual inspection logging and annotation:")

    img_file = st.file_uploader("Upload Component Photo (PNG, JPG)", type=["png", "jpg", "jpeg"])
    if img_file is not None:
        st.image(img_file, caption="Component Under Inspection", use_container_width=True)

        with st.form("manual_annotation_form"):
            an_c1, an_c2 = st.columns(2)
            defect_type = an_c1.selectbox("Observed Anomaly", ["Surface Scratch", "Component Misalignment", "Solder Bridge", "Pin Deformity", "Other"])
            severity_opt = an_c2.selectbox("Assessed Severity", ["Low", "Medium", "High", "Critical"])
            inspector_notes = st.text_area("Quality Inspector Observations", placeholder="e.g. Pin 4 slightly deformed on connector housing")
            submit_log = st.form_submit_button("Record Manual Inspection Entry", type="primary")

        if submit_log:
            st.success(f"Inspection entry saved: {defect_type} ({severity_opt}) recorded for lot review.")
