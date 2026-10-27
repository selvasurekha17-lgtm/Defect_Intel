import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title='DefectIntel | Quality Intelligence', page_icon='🔎', layout='wide')

st.markdown('''<style>
.block-container {padding-top: 1.5rem; max-width: 1450px;}
[data-testid="stMetric"] {background: #172235; border: 1px solid #2a4055; border-radius: 12px; padding: 15px;}
[data-testid="stSidebar"] {border-right: 1px solid #2a4055;}
h1, h2, h3 {letter-spacing: -.025em;}
</style>''', unsafe_allow_html=True)

REQUIRED = {'report_id', 'batch_id', 'defect_description'}

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data(show_spinner=False)
def analyze(csv_bytes, threshold):
    from io import BytesIO
    data = pd.read_csv(BytesIO(csv_bytes))
    data.columns = data.columns.str.strip()
    missing = REQUIRED - set(data.columns)
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(missing)))
    data = data.dropna(subset=['defect_description', 'batch_id']).copy()
    data['defect_description'] = data['defect_description'].astype(str).str.strip()
    data['batch_id'] = data['batch_id'].astype(str).str.strip()
    data = data[data['defect_description'] != ''].reset_index(drop=True)
    if data.empty:
        raise ValueError('No valid defect descriptions found in the CSV.')
    model = get_model()
    vectors = model.encode(data['defect_description'].tolist(), normalize_embeddings=True, show_progress_bar=False)
    if len(data) == 1:
        data['ai_cluster'] = 0
    else:
        grouping = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric='cosine', linkage='average')
        data['ai_cluster'] = grouping.fit_predict(vectors)
    return data, vectors

st.sidebar.title('◈ DefectIntel')
st.sidebar.caption('Manufacturing Quality Intelligence')
page = st.sidebar.radio('NAVIGATION', ['Overview', 'AI Investigator', 'Batch Intelligence', 'Alert Center', 'Defect Reports'])
st.sidebar.divider()
st.sidebar.subheader('Data Source')
file = st.sidebar.file_uploader('Upload defect reports CSV', type=['csv'])
if file is not None:
    csv_bytes = file.getvalue()
    st.sidebar.success('Uploaded file selected')
else:
    try:
        with open('data/defectintel_sample_defects_100.csv', 'rb') as f:
            csv_bytes = f.read()
        st.sidebar.info('Using sample dataset')
    except FileNotFoundError:
        st.error('Upload a CSV with report_id, batch_id and defect_description columns.')
        st.stop()
threshold = st.sidebar.slider('AI grouping distance', 0.20, 0.85, 0.50, 0.05,
    help='Lower values make smaller, more specific clusters. This is a tuning control, not model accuracy.')
st.sidebar.caption('Demo uses reported defects; it does not inspect physical products.')

try:
    with st.spinner('Analyzing defect descriptions...'):
        df, embeddings = analyze(csv_bytes, threshold)
except Exception as exc:
    st.error(f'Could not analyze CSV: {exc}')
    st.stop()

batch_order = sorted(df['batch_id'].unique().tolist())
cluster_sizes = df['ai_cluster'].value_counts()
examples = df.groupby('ai_cluster')['defect_description'].first().to_dict()
counts = pd.crosstab(df['batch_id'], df['ai_cluster']).reindex(batch_order, fill_value=0)

# Count-based alerts, not normalized defect rates.
alerts = []
for i in range(1, len(batch_order)):
    previous, current = batch_order[i - 1], batch_order[i]
    for cluster in counts.columns:
        before = int(counts.loc[previous, cluster])
        after = int(counts.loc[current, cluster])
        if before > 0 and after >= 5 and after >= 1.5 * before:
            alerts.append({'batch': current, 'previous_batch': previous,
                           'cluster': int(cluster), 'before': before, 'after': after,
                           'increase': round((after - before) * 100 / before, 1)})

st.title('◈ DefectIntel')
st.caption('AI-powered defect report intelligence · Interactive demonstration')

if page == 'Overview':
    a, b, c, d = st.columns(4)
    a.metric('Analyzed reports', len(df))
    b.metric('AI groups', df['ai_cluster'].nunique())
    c.metric('Production batches', len(batch_order))
    d.metric('Rule-based alerts', len(alerts))
    st.subheader('Most frequent AI groups')
    top = cluster_sizes.head(10)
    display = pd.DataFrame({'Group': [f'Group {int(k)}: {str(examples[k])[:48]}' for k in top.index],
                            'Reports': top.values}).set_index('Group')
    st.bar_chart(display, horizontal=True)
    st.caption('Group names are example reports, not verified defect classifications.')
    st.subheader('Reports per batch')
    st.bar_chart(df.groupby('batch_id').size().reindex(batch_order).rename('Reports'))
    if alerts:
        st.warning(f'{len(alerts)} report-count increase alert(s) found. Open Alert Center for evidence.')
    st.info('Data represents defect reports, not production defect rates. No inspected-unit denominators are available.')

elif page == 'AI Investigator':
    st.subheader('Find similar defect reports')
    query = st.text_area('Describe the observed defect', placeholder='Example: Display flashes while charging')
    top_n = st.slider('Number of matches', 1, min(10, len(df)), min(5, len(df)))
    if st.button('Analyze with AI', type='primary'):
        if not query.strip():
            st.warning('Enter a defect description first.')
        else:
            q = get_model().encode([query.strip()], normalize_embeddings=True)
            scores = cosine_similarity(q, embeddings)[0]
            indices = np.argsort(scores)[::-1][:top_n]
            for rank, idx in enumerate(indices, start=1):
                r = df.iloc[int(idx)]
                with st.container(border=True):
                    st.markdown(f'**Match {rank} · Similarity {scores[idx]:.3f}**')
                    st.write(r['defect_description'])
                    st.caption(f"Batch {r['batch_id']} · AI Group {int(r['ai_cluster'])} · Report {r['report_id']}")
            st.caption('Similarity measures semantic closeness, not defect certainty or prediction accuracy.')

elif page == 'Batch Intelligence':
    st.subheader('Compare production batches')
    if len(batch_order) < 2:
        st.info('At least two batches are required for comparison.')
    else:
        left, right = st.columns(2)
        previous = left.selectbox('First batch', batch_order, index=0)
        current = right.selectbox('Second batch', batch_order, index=min(1, len(batch_order)-1))
        if previous == current:
            st.info('Select different batches to compare.')
        else:
            comparison = pd.DataFrame({'AI Group': counts.columns.astype(int),
                previous: counts.loc[previous].values, current: counts.loc[current].values})
            comparison['Change in reports'] = comparison[current] - comparison[previous]
            comparison['Example defect'] = comparison['AI Group'].map(examples)
            st.dataframe(comparison.sort_values('Change in reports', ascending=False), hide_index=True, use_container_width=True)
    st.subheader('All batch counts')
    with st.expander('Open batch-by-group count table'):
        st.dataframe(counts, use_container_width=True)
    st.caption('Batch IDs are sorted alphabetically; confirm their actual chronological order before interpreting changes as trends.')

elif page == 'Alert Center':
    st.subheader('Rule-based Early Warning Center')
    st.caption('Rule: current count ≥5, previous count >0, and count increases by at least 50% between adjacent sorted batch IDs.')
    if not alerts:
        st.success('No alerts meet the current rule.')
    else:
        for n, alert in enumerate(alerts):
            title = f"Batch {alert['batch']} · Group {alert['cluster']} · +{alert['increase']}% ({alert['before']} → {alert['after']})"
            with st.expander('⚠️ ' + title, expanded=(n == 0)):
                st.write('**Example issue:**', examples.get(alert['cluster'], 'Unknown'))
                st.write(f"Compared with batch {alert['previous_batch']}.")
                evidence = df[(df['batch_id'] == alert['batch']) & (df['ai_cluster'] == alert['cluster'])]
                st.dataframe(evidence[['report_id', 'batch_id', 'defect_description']], hide_index=True, use_container_width=True)
                ack_key = f"ack_{alert['batch']}_{alert['cluster']}_{alert['previous_batch']}"
                if st.button('Acknowledge for this session', key=ack_key):
                    st.session_state[ack_key + '_done'] = True
                if st.session_state.get(ack_key + '_done'):
                    st.success('Acknowledged in this browser session (not stored permanently).')
    st.info('Alerts compare report counts, not production-normalized defect rates. They do not establish root cause or severity.')

elif page == 'Defect Reports':
    st.subheader('Explore analyzed reports')
    batch = st.selectbox('Filter batch', ['All'] + batch_order)
    groups = sorted(int(v) for v in df['ai_cluster'].unique())
    group = st.selectbox('Filter AI group', ['All'] + groups)
    view = df.copy()
    if batch != 'All':
        view = view[view['batch_id'] == batch]
    if group != 'All':
        view = view[view['ai_cluster'] == group]
    st.write(f'{len(view)} matching report(s)')
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button('Download filtered CSV', view.to_csv(index=False).encode('utf-8'),
                       file_name='defectintel_analyzed.csv', mime='text/csv')

st.divider()
st.caption('Prototype for research/demo use. Sample data may be synthetic; validate model groups and alerts before operational deployment.')
