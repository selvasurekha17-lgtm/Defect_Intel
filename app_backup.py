
import streamlit as st
import pandas as pd

from backend import analyze_defects, search_defect, batch_analytics

st.set_page_config(
    page_title="DefectIntel",
    page_icon="🔍",
    layout="wide"
)

st.title("DefectIntel")
st.caption("AI-Powered Manufacturing Quality Intelligence")

@st.cache_resource
def load_analysis():
    return analyze_defects()

try:
    df, embeddings = load_analysis()
    summary, alerts = batch_analytics(df)
except Exception as error:
    st.error(f"Backend error: {error}")
    st.stop()

st.subheader("Quality Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Reports", len(df))
col2.metric("AI Defect Groups", df["ai_cluster"].nunique())
col3.metric("Triggered Alerts", len(alerts))

st.divider()

st.subheader("AI Defect Distribution")

cluster_counts = (
    df["ai_cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

st.subheader("Batch Defect Analysis")

st.dataframe(summary, use_container_width=True)

st.subheader("Early Warning Alerts")

if alerts:
    for alert in alerts:
        st.warning(
            f"Cluster {alert['cluster']} increased by "
            f"{alert['increase_percent']}% "
            f"in batch {alert['batch']}"
        )
else:
    st.success("No alerts triggered by the current rules.")

st.divider()

st.subheader("AI Defect Similarity Search")

query = st.text_input(
    "Describe a new manufacturing defect",
    placeholder="Example: Phone display keeps flashing"
)

if st.button("Analyze Defect"):
    if not query.strip():
        st.warning("Please enter a defect description.")
    else:
        results = search_defect(query, df, embeddings)

        for result in results:
            st.write("**Description:**", result["description"])
            st.write("**AI Cluster:**", result["cluster"])
            st.write("**Similarity:**", result["similarity"])
            st.divider()

st.subheader("Analyzed Defect Reports")

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)
