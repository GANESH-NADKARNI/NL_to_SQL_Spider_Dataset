"""
NL2SQL Research - Streamlit Application
=========================================
Run after training: streamlit run app.py
"""
import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="NL2SQL Research",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

TRAINED_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
DATA_SPLITS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'splits')

MODEL_COLORS = {
    'Rule-Based NLP': '#FF6B9D',
    'Random Forest': '#4ECDC4',
    'LSTM': '#45B7D1',
    'T5 Transformer': '#FFA07A',
}

MODEL_ICONS = {
    'Rule-Based NLP': '🎯',
    'Random Forest': '🌲',
    'LSTM': '🔁',
    'T5 Transformer': '⚡',
}

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d3159;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* Model cards in sidebar */
    .model-card {
        background: #252840;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        border-left: 4px solid;
    }
    .model-name { font-size: 14px; font-weight: 600; color: #fff; }
    .model-metric { font-size: 12px; color: #aaa; }
    .metric-value { font-weight: 700; }

    /* Query output boxes */
    .sql-box {
        background: #1e2235;
        border: 1px solid #3d4166;
        border-radius: 8px;
        padding: 14px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        color: #c9d1d9;
        white-space: pre-wrap;
        word-break: break-word;
    }
    .sql-box.expected { border-left: 4px solid #4CAF50; }
    .sql-box.predicted { border-left: 4px solid #2196F3; }
    .sql-box.match { border-left: 4px solid #4CAF50 !important; background: #1a2e1a; }
    .sql-box.no-match { border-left: 4px solid #f44336 !important; background: #2e1a1a; }

    /* Section headers */
    .section-title {
        font-size: 22px;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1d2e;
        border-radius: 8px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #fff !important;
        background-color: #2d3159 !important;
        border-radius: 6px;
    }

    /* Match badge */
    .badge-match { color: #4CAF50; font-weight: 700; }
    .badge-no-match { color: #f44336; font-weight: 700; }
    .badge-partial { color: #FF9800; font-weight: 700; }

    /* Table results */
    .result-table {
        font-size: 12px;
        color: #c9d1d9;
    }
    
    /* Status indicators */
    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all trained models."""
    models = {}
    errors = {}

    # Rule-Based NLP
    rb_path = os.path.join(TRAINED_DIR, 'rule_based.pkl')
    if os.path.exists(rb_path):
        try:
            from models.rule_based import RuleBasedNLP
            m = RuleBasedNLP()
            m.load(rb_path)
            models['Rule-Based NLP'] = m
        except Exception as e:
            errors['Rule-Based NLP'] = str(e)

    # Random Forest
    rf_path = os.path.join(TRAINED_DIR, 'random_forest.pkl')
    if os.path.exists(rf_path):
        try:
            from models.random_forest_model import RandomForestNL2SQL
            m = RandomForestNL2SQL()
            m.load(rf_path)
            models['Random Forest'] = m
        except Exception as e:
            errors['Random Forest'] = str(e)

    # LSTM
    lstm_dir = os.path.join(TRAINED_DIR, 'lstm')
    if os.path.exists(lstm_dir):
        try:
            from models.lstm_model import LSTMSeq2Seq
            from data.preprocess import Vocabulary
            nl_vocab = Vocabulary.load(os.path.join(TRAINED_DIR, 'nl_vocab.pkl'))
            sql_vocab = Vocabulary.load(os.path.join(TRAINED_DIR, 'sql_vocab.pkl'))
            m = LSTMSeq2Seq()
            m.load(lstm_dir, nl_vocab, sql_vocab)
            models['LSTM'] = m
        except Exception as e:
            errors['LSTM'] = str(e)

    # T5
    t5_dir = os.path.join(TRAINED_DIR, 't5')
    if os.path.exists(t5_dir):
        try:
            from models.t5_model import T5NL2SQL
            m = T5NL2SQL()
            m.load(t5_dir)
            models['T5 Transformer'] = m
        except Exception as e:
            errors['T5 Transformer'] = str(e)

    return models, errors


@st.cache_data(show_spinner=False)
def load_data_splits():
    """Load train/test splits."""
    try:
        train_df = pd.read_csv(os.path.join(DATA_SPLITS_DIR, 'train.csv'))
        familiar_df = pd.read_csv(os.path.join(DATA_SPLITS_DIR, 'familiar_test.csv'))
        unseen_df = pd.read_csv(os.path.join(DATA_SPLITS_DIR, 'unseen_test.csv'))
        return train_df, familiar_df, unseen_df
    except Exception as e:
        return None, None, None


@st.cache_data(show_spinner=False)
def load_all_metrics():
    """Load saved metrics from training."""
    path = os.path.join(TRAINED_DIR, 'all_metrics.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def get_db_connection():
    """Get SQLite DB connection."""
    from utils.db_utils import initialize_db
    db_path = os.path.join(os.path.dirname(__file__), 'nl2sql_demo.db')
    return initialize_db(db_path)


# ─── Helper Functions ───────────────────────────────────────────
def normalize_sql_for_compare(sql):
    import re
    sql = sql.upper().strip().rstrip(';')
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r"'[^']*'", "'<STR>'", sql)
    sql = re.sub(r'\b\d+(\.\d+)?\b', '<NUM>', sql)
    return sql


def is_exact_match(pred, true):
    return normalize_sql_for_compare(pred) == normalize_sql_for_compare(true)


def is_template_match(pred, true):
    import re
    def get_template(sql):
        sql_up = sql.upper()
        kws = [kw for kw in ['SELECT', 'FROM', 'WHERE', 'JOIN', 'GROUP BY', 
                              'ORDER BY', 'HAVING', 'LIMIT', 'DISTINCT']
               if kw in sql_up]
        tables = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.IGNORECASE)
        tnames = sorted(set(t for tup in tables for t in tup if t))
        return '|'.join(kws) + '||' + '|'.join(tnames)
    return get_template(pred) == get_template(true)


def render_sql_comparison(model_name, predicted_sql, expected_sql):
    """Render side-by-side SQL comparison."""
    em = is_exact_match(predicted_sql, expected_sql)
    tm = is_template_match(predicted_sql, expected_sql)

    color = MODEL_COLORS.get(model_name, '#888')
    icon = MODEL_ICONS.get(model_name, '🤖')

    match_class = 'match' if em else ('no-match')
    badge = '✅ Exact Match' if em else ('🟡 Template Match' if tm else '❌ No Match')
    badge_class = 'badge-match' if em else ('badge-partial' if tm else 'badge-no-match')

    st.markdown(f"""
    <div style="border-left: 3px solid {color}; padding-left: 12px; margin-bottom: 16px;">
        <div style="font-size: 15px; font-weight: 600; color: {color}; margin-bottom: 8px;">
            {icon} {model_name} &nbsp; <span class="{badge_class}">{badge}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Expected SQL**")
        st.markdown(f'<div class="sql-box expected">{expected_sql}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("**Model Output**")
        st.markdown(f'<div class="sql-box predicted {match_class}">{predicted_sql}</div>', unsafe_allow_html=True)


def execute_and_show_table(sql, label, color):
    """Execute SQL and show results table."""
    from utils.db_utils import execute_sql_safe
    conn = get_db_connection()
    cols, rows, err = execute_sql_safe(conn, sql)

    st.markdown(f"**{label} Results**")
    if err:
        st.caption(f"⚠️ Query execution note: {err[:100]}")
    elif cols and rows is not None:
        if len(rows) == 0:
            st.caption("*(Query executed successfully — 0 rows returned)*")
        else:
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True, height=200)
            st.caption(f"{len(rows)} row(s) returned")
    else:
        st.caption("*(Could not execute on demo database)*")


def run_batch_evaluation(models, nl_list, sql_list, split_name: str):
    """Run batch evaluation across all models and return results."""
    results = {}
    progress = st.progress(0)
    status = st.empty()

    model_names = list(models.keys())
    for i, (mname, model) in enumerate(models.items()):
        status.text(f"Evaluating {mname}...")
        progress.progress((i) / len(model_names))

        try:
            t0 = time.time()
            preds = model.predict_batch(list(nl_list))
            elapsed = time.time() - t0

            exact_hits = sum(is_exact_match(p, t) for p, t in zip(preds, sql_list))
            template_hits = sum(is_template_match(p, t) for p, t in zip(preds, sql_list))
            n = len(sql_list)

            per_query = []
            for nl, true, pred in zip(nl_list, sql_list, preds):
                per_query.append({
                    'NL Query': nl,
                    'Expected SQL': true,
                    'Predicted SQL': pred,
                    'Exact Match': '✅' if is_exact_match(pred, true) else '❌',
                    'Template Match': '✅' if is_template_match(pred, true) else '❌',
                })

            results[mname] = {
                'exact_acc': exact_hits / n,
                'template_acc': template_hits / n,
                'exact_hits': exact_hits,
                'template_hits': template_hits,
                'total': n,
                'time_sec': elapsed,
                'per_query': per_query,
                'predictions': preds,
            }
        except Exception as e:
            st.warning(f"Error evaluating {mname}: {e}")
            results[mname] = {
                'exact_acc': 0, 'template_acc': 0,
                'exact_hits': 0, 'template_hits': 0,
                'total': len(sql_list), 'time_sec': 0,
                'per_query': [], 'predictions': []
            }
        progress.progress((i + 1) / len(model_names))

    progress.empty()
    status.empty()
    return results


def render_batch_results(results):
    """Render batch test results with charts and tables."""
    model_names = list(results.keys())
    exact_accs = [results[m]['exact_acc'] * 100 for m in model_names]
    template_accs = [results[m]['template_acc'] * 100 for m in model_names]
    colors = [MODEL_COLORS.get(m, '#888') for m in model_names]

    # ── Accuracy Chart ──
    fig = go.Figure()
    x = list(range(len(model_names)))
    width = 0.35

    fig.add_trace(go.Bar(
        name='Template Accuracy',
        x=[n - width/2 for n in x],
        y=template_accs,
        width=width,
        marker_color=colors,
        opacity=0.85,
        text=[f"{a:.1f}%" for a in template_accs],
        textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name='Exact Match',
        x=[n + width/2 for n in x],
        y=exact_accs,
        width=width,
        marker_color=colors,
        opacity=0.55,
        text=[f"{a:.1f}%" for a in exact_accs],
        textposition='outside',
    ))

    fig.update_layout(
        title='Model Accuracy Comparison',
        xaxis=dict(tickvals=x, ticktext=model_names, color='#ccc'),
        yaxis=dict(range=[0, 110], ticksuffix='%', color='#ccc', gridcolor='#2d3159'),
        paper_bgcolor='#1a1d2e',
        plot_bgcolor='#1a1d2e',
        font=dict(color='#ccc'),
        legend=dict(bgcolor='#252840', bordercolor='#3d4166'),
        bargap=0.2,
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary Metrics Table ──
    st.markdown("#### Summary")
    summary_data = []
    for mname in model_names:
        r = results[mname]
        summary_data.append({
            'Model': f"{MODEL_ICONS.get(mname, '🤖')} {mname}",
            'Template Accuracy': f"{r['template_acc']*100:.1f}%",
            'Exact Match': f"{r['exact_acc']*100:.1f}%",
            'Correct (Template)': f"{r['template_hits']}/{r['total']}",
            'Correct (Exact)': f"{r['exact_hits']}/{r['total']}",
            'Inference Time': f"{r['time_sec']:.1f}s",
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Per-Query Table per Model ──
    st.markdown("#### Per-Query Results")
    model_tabs = st.tabs([f"{MODEL_ICONS.get(m, '🤖')} {m}" for m in model_names])
    for tab, mname in zip(model_tabs, model_names):
        with tab:
            per_q = results[mname]['per_query']
            if per_q:
                df = pd.DataFrame(per_q)
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.info("No results available")


# ─── SIDEBAR ────────────────────────────────────────────────────
def render_sidebar(models, all_metrics):
    with st.sidebar:
        st.markdown("## 🧠 NL2SQL Research")
        st.markdown(f"**{len(models)} MODELS LOADED**" if models else "⚠️ No models loaded")
        st.divider()

        st.markdown("### 📊 Model Metrics")
        st.caption("Evaluated on familiar + unseen queries combined")

        all_model_names = ['Rule-Based NLP', 'Random Forest', 'LSTM', 'T5 Transformer']
        for mname in all_model_names:
            color = MODEL_COLORS[mname]
            icon = MODEL_ICONS[mname]
            loaded = mname in models
            m = all_metrics.get(mname, {})

            template_acc = m.get('familiar_template', 0) * 100
            exact_acc = m.get('familiar_exact', 0) * 100

            status_color = color if loaded else '#555'
            status_text = 'loaded' if loaded else 'not trained'

            st.markdown(f"""
            <div class="model-card" style="border-color: {status_color};">
                <div class="model-name">{icon} {mname}</div>
                <div class="model-metric">
                    <span style="color:{status_color}">●</span> {status_text}<br>
                    <span class="metric-value" style="color:{color}">{template_acc:.1f}%</span>
                    <span style="color:#888"> template · </span>
                    <span class="metric-value" style="color:{color}">{exact_acc:.1f}%</span>
                    <span style="color:#888"> exact</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🗄️ Database Status")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            n_tables = cursor.fetchone()[0]
            st.success(f"✅ SQLite ready — {n_tables} tables")
            st.caption("Source: auto-generated schema")
            with st.expander("Tables in DB"):
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
                tables = [r[0] for r in cursor.fetchall()]
                for t in tables:
                    st.markdown(f"- `{t}`")
        except Exception as e:
            st.error(f"DB Error: {e}")

        st.divider()
        if not models:
            st.warning("⚠️ No trained models found.\n\nRun training first:\n```\npython train.py\n```")


# ─── MAIN APP ───────────────────────────────────────────────────
def main():
    with st.spinner("Loading models..."):
        models, load_errors = load_models()
    all_metrics = load_all_metrics()
    train_df, familiar_df, unseen_df = load_data_splits()

    # Sidebar
    render_sidebar(models, all_metrics)

    if load_errors:
        with st.expander("⚠️ Model Load Errors"):
            for m, e in load_errors.items():
                st.error(f"**{m}**: {e}")

    # ── TABS ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 Familiar Query",
        "🟣 Unseen Query",
        "🟢 Run Familiar Test",
        "🔴 Run Unseen Test",
    ])

    # ── TAB 1: Familiar Query ──────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-title">Familiar Query — Model-Trained Phrasings</div>',
                    unsafe_allow_html=True)
        st.caption("Select a query the model has seen during training. Compare expected vs predicted SQL.")

        if familiar_df is None:
            st.warning("Data splits not found. Please run `python train.py` first.")
        elif not models:
            st.warning("No models loaded. Please run `python train.py` first.")
        else:
            # Query selector
            options = familiar_df['Prompt'].tolist()
            options_display = [f"[{i+1}] {q[:80]}{'...' if len(q)>80 else ''}"
                               for i, q in enumerate(options)]

            selected_idx = st.selectbox(
                "Select a familiar NL query:",
                range(len(options)),
                format_func=lambda i: options_display[i],
                key="familiar_select"
            )

            selected_nl = familiar_df.iloc[selected_idx]['Prompt']
            expected_sql = familiar_df.iloc[selected_idx]['Query']

            st.markdown(f"**Selected Query:** `{selected_nl}`")
            st.divider()

            show_tables = st.checkbox("Show table results (demo DB)", value=True, key="fam_show_tables")

            if st.button("▶ Run All Models", key="fam_run"):
                st.markdown("### Model Outputs")
                for mname, model in models.items():
                    try:
                        pred_sql = model.predict(selected_nl)
                    except Exception as e:
                        pred_sql = f"-- Error: {e}"

                    render_sql_comparison(mname, pred_sql, expected_sql)

                    if show_tables:
                        with st.expander(f"Table Results — {mname}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                execute_and_show_table(expected_sql, "Expected", '#4CAF50')
                            with col2:
                                execute_and_show_table(pred_sql, "Predicted", '#2196F3')
                    st.divider()

    # ── TAB 2: Unseen Query ────────────────────────────────────────
    with tab2:
        st.markdown('<div class="section-title">Unseen Query — Never-Seen NL Phrasings</div>',
                    unsafe_allow_html=True)
        st.caption("Select a query the model has NOT seen during training. The model knows the SQL structure but not this NL wording.")

        if unseen_df is None:
            st.warning("Data splits not found. Please run `python train.py` first.")
        elif not models:
            st.warning("No models loaded. Please run `python train.py` first.")
        else:
            options = unseen_df['Prompt'].tolist()
            options_display = [f"[{i+1}] {q[:80]}{'...' if len(q)>80 else ''}"
                               for i, q in enumerate(options)]

            selected_idx = st.selectbox(
                "Select an unseen NL query:",
                range(len(options)),
                format_func=lambda i: options_display[i],
                key="unseen_select"
            )

            selected_nl = unseen_df.iloc[selected_idx]['Prompt']
            expected_sql = unseen_df.iloc[selected_idx]['Query']

            st.markdown(f"**Selected Query:** `{selected_nl}`")
            st.divider()

            show_tables = st.checkbox("Show table results (demo DB)", value=True, key="unseen_show_tables")

            if st.button("▶ Run All Models", key="unseen_run"):
                st.markdown("### Model Outputs")
                for mname, model in models.items():
                    try:
                        pred_sql = model.predict(selected_nl)
                    except Exception as e:
                        pred_sql = f"-- Error: {e}"

                    render_sql_comparison(mname, pred_sql, expected_sql)

                    if show_tables:
                        with st.expander(f"Table Results — {mname}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                execute_and_show_table(expected_sql, "Expected", '#4CAF50')
                            with col2:
                                execute_and_show_table(pred_sql, "Predicted", '#2196F3')
                    st.divider()

    # ── TAB 3: Run Familiar Test ───────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title">Batch Test — Familiar Queries</div>',
                    unsafe_allow_html=True)
        st.caption("Run all models on the familiar test set and compare performance.")

        if familiar_df is None:
            st.warning("Data splits not found. Please run `python train.py` first.")
        elif not models:
            st.warning("No models loaded. Please run `python train.py` first.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                n_samples = st.slider(
                    "Number of queries to test",
                    min_value=10, max_value=min(200, len(familiar_df)),
                    value=min(100, len(familiar_df)),
                    step=10, key="fam_n"
                )
            with col2:
                st.metric("Available Familiar Queries", len(familiar_df))

            if st.button("▶ Run Familiar Test", key="fam_batch_run", type="primary"):
                sample_df = familiar_df.sample(n=min(n_samples, len(familiar_df)),
                                               random_state=42)
                st.info(f"Running {len(sample_df)} queries across {len(models)} models...")

                with st.spinner("Evaluating all models..."):
                    results = run_batch_evaluation(
                        models,
                        sample_df['Prompt'].tolist(),
                        sample_df['Query'].tolist(),
                        'familiar'
                    )

                st.success("✅ Evaluation complete!")
                render_batch_results(results)

                # Store results in session state
                st.session_state['fam_results'] = results

            elif 'fam_results' in st.session_state:
                st.info("Showing previous results. Click button to re-run.")
                render_batch_results(st.session_state['fam_results'])

    # ── TAB 4: Run Unseen Test ─────────────────────────────────────
    with tab4:
        st.markdown('<div class="section-title">Batch Test — Unseen Queries</div>',
                    unsafe_allow_html=True)
        st.caption("Run all models on never-seen NL phrasings. Tests generalization ability.")

        if unseen_df is None:
            st.warning("Data splits not found. Please run `python train.py` first.")
        elif not models:
            st.warning("No models loaded. Please run `python train.py` first.")
        else:
            col1, col2 = st.columns([2, 1])
            with col1:
                n_samples = st.slider(
                    "Number of queries to test",
                    min_value=10, max_value=min(200, len(unseen_df)),
                    value=min(100, len(unseen_df)),
                    step=10, key="unseen_n"
                )
            with col2:
                st.metric("Available Unseen Queries", len(unseen_df))

            if st.button("▶ Run Unseen Test", key="unseen_batch_run", type="primary"):
                sample_df = unseen_df.sample(n=min(n_samples, len(unseen_df)),
                                             random_state=42)
                st.info(f"Running {len(sample_df)} queries across {len(models)} models...")

                with st.spinner("Evaluating all models..."):
                    results = run_batch_evaluation(
                        models,
                        sample_df['Prompt'].tolist(),
                        sample_df['Query'].tolist(),
                        'unseen'
                    )

                st.success("✅ Evaluation complete!")
                render_batch_results(results)
                st.session_state['unseen_results'] = results

            elif 'unseen_results' in st.session_state:
                st.info("Showing previous results. Click button to re-run.")
                render_batch_results(st.session_state['unseen_results'])

            # Show accuracy drop chart (familiar vs unseen comparison)
            if 'fam_results' in st.session_state and 'unseen_results' in st.session_state:
                st.markdown("---")
                st.markdown("### 📉 Familiar vs Unseen — Accuracy Drop Analysis")
                fam_r = st.session_state['fam_results']
                unseen_r = st.session_state['unseen_results']
                common_models = [m for m in fam_r if m in unseen_r]

                fig2 = go.Figure()
                for mname in common_models:
                    color = MODEL_COLORS.get(mname, '#888')
                    fig2.add_trace(go.Scatter(
                        x=['Familiar', 'Unseen'],
                        y=[fam_r[mname]['template_acc'] * 100,
                           unseen_r[mname]['template_acc'] * 100],
                        mode='lines+markers',
                        name=mname,
                        line=dict(color=color, width=2),
                        marker=dict(size=10, color=color),
                    ))

                fig2.update_layout(
                    title='Template Accuracy: Familiar → Unseen',
                    xaxis=dict(color='#ccc'),
                    yaxis=dict(range=[0, 105], ticksuffix='%', color='#ccc',
                               gridcolor='#2d3159'),
                    paper_bgcolor='#1a1d2e',
                    plot_bgcolor='#1a1d2e',
                    font=dict(color='#ccc'),
                    legend=dict(bgcolor='#252840'),
                    height=380,
                )
                st.plotly_chart(fig2, use_container_width=True)


if __name__ == '__main__':
    main()
