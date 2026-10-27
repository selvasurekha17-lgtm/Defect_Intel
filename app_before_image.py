
import streamlit as st
import pandas as pd

from backend import analyze_defects, search_defect, batch_analytics

st.set_page_config(
    page_title="DefectIntel",
    page_icon="🔍",
    layout="wide"
)

st.sidebar.title("◈ DefectIntel")
st.sidebar.caption("Manufacturing Intelligence")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "AI Investigator",
        "Batch Intelligence",
        "Defect Reports"
    ]
)

st.title("◈ DefectIntel")
st.caption("AI-Powered Manufacturing Quality Intelligence")
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


if page == "Overview":

    st.header("Quality Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Reports", len(df))
    col2.metric("AI Defect Groups", df["ai_cluster"].nunique())
    col3.metric("Triggered Alerts", len(alerts))

    st.divider()

    st.subheader("AI Defect Distribution")

    counts = df["ai_cluster"].value_counts()
    st.bar_chart(counts.head(10))

    st.subheader("Priority Alerts")

    if alerts:
        for alert in alerts:
            st.warning(
                f"Cluster {alert['cluster']} increased by "
                f"{alert['increase_percent']}% "
                f"in batch {alert['batch']}"
            )
    else:
        st.success("No alerts triggered.")


elif page == "AI Investigator":

    st.header("AI Defect Investigator")

    st.write(
        "Describe a new manufacturing defect. "
        "AI will identify similar existing reports."
    )

    query = st.text_input(
        "Enter defect description",
        placeholder="Example: Phone display keeps blinking"
    )

    if st.button("Analyze with AI"):

        if not query.strip():
            st.warning("Please enter a defect description.")
        else:
            results = search_defect(query, df, embeddings)

            for result in results:
                with st.container(border=True):
                    st.write("**Similar Report:**", result["description"])
                    st.write("**AI Cluster:**", result["cluster"])
                    st.write("**Similarity:**", result["similarity"])


elif page == "Batch Intelligence":

    st.header("Batch Intelligence")

    st.write("Compare defect reports across manufacturing batches.")

    st.subheader("Batch-wise Defect Counts")

    batch_counts = df["batch_id"].value_counts().sort_index()
    st.bar_chart(batch_counts)

    st.subheader("Early Warning Alerts")

    if alerts:
        for alert in alerts:
            st.warning(
                f"Cluster {alert['cluster']} increased by "
                f"{alert['increase_percent']}% "
                f"in batch {alert['batch']}"
            )
    else:
        st.success("No alerts triggered.")

    with st.expander("View detailed cluster counts"):
        st.dataframe(summary, use_container_width=True)


elif page == "Defect Reports":

    st.header("Analyzed Defect Reports")

    selected_batch = st.selectbox(
        "Filter by batch",
        ["All"] + sorted(df["batch_id"].unique().tolist())
    )

    filtered_df = df.copy()

    if selected_batch != "All":
        filtered_df = df[df["batch_id"] == selected_batch]

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Analyzed Reports",
        data=filtered_df.to_csv(index=False),
        file_name="defectintel_report.csv",
        mime="text/csv"
    )


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
