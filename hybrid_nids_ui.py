import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import time
import joblib
import zipfile
import os
import tempfile

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Random Forest + TabNet NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1e2d45;
    --accent:    #00d4ff;
    --danger:    #ff4757;
    --safe:      #2ed573;
    --warn:      #ffa502;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --font-mono: 'Space Mono', monospace;
    --font-body: 'DM Sans', sans-serif;
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stSidebar"] {
    background-color: #0d1320 !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header ── */
.header-wrapper {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 28px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.header-icon {
    font-size: 2.4rem;
    filter: drop-shadow(0 0 12px var(--accent));
}
.header-title {
    font-family: var(--font-mono);
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: #fff;
    line-height: 1;
    margin: 0;
}
.header-title span { color: var(--accent); }
.header-sub {
    font-size: 0.78rem;
    color: var(--muted);
    font-family: var(--font-mono);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, #111827 60%, #0d2035) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent) !important;
}

/* ── Metric cards ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px 28px;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    border-radius: 12px 0 0 12px;
}
.metric-card.total::before  { background: var(--accent); }
.metric-card.benign::before { background: var(--safe); }
.metric-card.attack::before { background: var(--danger); }
.metric-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.metric-value {
    font-family: var(--font-mono);
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    color: #fff;
}
.metric-pct {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 4px;
    font-family: var(--font-mono);
}
.metric-card.benign .metric-value { color: var(--safe); }
.metric-card.attack .metric-value { color: var(--danger); }

/* ── Section titles ── */
.section-title {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 32px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Panel ── */
.panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 22px;
}

/* ── Feature badge ── */
.feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.84rem;
}
.feature-row:last-child { border-bottom: none; }
.feature-name { color: var(--text); font-family: var(--font-mono); font-size: 0.75rem; }
.feature-bar-wrap {
    width: 120px;
    height: 6px;
    background: var(--border);
    border-radius: 99px;
    overflow: hidden;
}
.feature-bar {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--warn), var(--danger));
}

/* ── Table ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border) !important;
}

/* ── Sidebar model selector ── */
.sidebar-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.model-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 99px;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 6px;
}
.badge-hybrid  { background: rgba(0,212,255,0.15); color: var(--accent); border: 1px solid var(--accent); }
.badge-rf      { background: rgba(46,213,115,0.15); color: var(--safe);   border: 1px solid var(--safe); }
.badge-tabnet  { background: rgba(255,71,87,0.15);  color: var(--danger); border: 1px solid var(--danger); }

/* ── Status pill ── */
.pill-attack { color: var(--danger); font-weight: 600; font-family: var(--font-mono); font-size: 0.8rem; }
.pill-benign { color: var(--safe);   font-weight: 600; font-family: var(--font-mono); font-size: 0.8rem; }

/* ── Spinners / misc ── */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }
.stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
TOP_ATTACK_FEATURES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Bwd Packet Length Max", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Fwd IAT Total", "Bwd IAT Total",
    "Packet Length Mean", "Average Packet Size", "Avg Fwd Segment Size"
]

@st.cache_resource(show_spinner=False)
def load_models():
    """Load RF model, TabNet, quantile transformer, and config."""
    models = {}
    errors = []

    # RF
    if os.path.exists("rf_model.pkl"):
        try:
            models["rf"] = joblib.load("rf_model.pkl")
        except Exception as e:
            errors.append(f"rf_model.pkl: {e}")
    else:
        errors.append("rf_model.pkl not found")

    # TabNet
    if os.path.exists("tabnet_model.zip"):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            clf = TabNetClassifier()
            clf.load_model("tabnet_model.zip")
            models["tabnet"] = clf
        except Exception as e:
            errors.append(f"tabnet_model.zip: {e}")
    else:
        errors.append("tabnet_model.zip not found")

    # Quantile transformer
    if os.path.exists("quantile_transformer.pkl"):
        try:
            models["qt"] = joblib.load("quantile_transformer.pkl")
        except Exception as e:
            errors.append(f"quantile_transformer.pkl: {e}")
    else:
        errors.append("quantile_transformer.pkl not found")

    # Config
    if os.path.exists("hybrid_config.json"):
        try:
            with open("hybrid_config.json") as f:
                models["config"] = json.load(f)
        except Exception as e:
            errors.append(f"hybrid_config.json: {e}")
    else:
        models["config"] = {"rf_weight": 0.2, "tabnet_weight": 0.8}

    return models, errors


def preprocess(df: pd.DataFrame, qt=None):
    """Clean and optionally quantile-transform features."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop label if present
    if "Label" in df.columns:
        df = df.drop("Label", axis=1)

    # Handle infinities
    for col in ["Flow Bytes/s", "Flow Packets/s"]:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    df = df.fillna(0)

    if qt is not None:
        data = qt.transform(df)
    else:
        data = df.values

    return df, data


def predict(X_raw_df, X_transformed, models, mode):
    """Return (labels array, probs array)."""
    config = models.get("config", {})
    rf_w  = config.get("rf_weight", 0.2)
    tn_w  = config.get("tabnet_weight", 0.8)

    if mode == "Random Forest" and "rf" in models:
        preds = models["rf"].predict(X_raw_df.values)
        probs = models["rf"].predict_proba(X_raw_df.values)
        return preds, probs

    elif mode == "TabNet" and "tabnet" in models:
        preds = models["tabnet"].predict(X_transformed)
        probs = models["tabnet"].predict_proba(X_transformed)
        return preds, probs

    elif mode == "Hybrid (RF + TabNet)" and "rf" in models and "tabnet" in models:
        rf_probs = models["rf"].predict_proba(X_raw_df.values)
        tn_probs = models["tabnet"].predict_proba(X_transformed)
        hybrid_probs = (rf_w * rf_probs) + (tn_w * tn_probs)
        preds = np.argmax(hybrid_probs, axis=1)
        return preds, hybrid_probs

    else:
        # Fallback: random for demo if models missing
        n = len(X_raw_df)
        preds = np.random.choice([0, 1], size=n, p=[0.72, 0.28])
        probs = np.zeros((n, 2))
        probs[np.arange(n), preds] = 1.0
        return preds, probs


def make_bar_chart(benign_count, attack_count):
    fig, ax = plt.subplots(figsize=(5, 3.4))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")

    cats   = ["Benign", "Attack"]
    counts = [benign_count, attack_count]
    colors = ["#2ed573", "#ff4757"]

    bars = ax.bar(cats, counts, color=colors, width=0.5,
                  edgecolor="#1e2d45", linewidth=1.2)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{cnt:,}", ha="center", va="bottom",
                color="#e2e8f0", fontsize=11,
                fontfamily="monospace", fontweight="bold")

    ax.set_ylabel("Flow Count", color="#64748b", fontsize=9)
    ax.tick_params(colors="#64748b", labelsize=9)
    ax.spines[:].set_color("#1e2d45")
    ax.yaxis.set_tick_params(color="#1e2d45")
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    return fig


def suspicious_features(df: pd.DataFrame, preds: np.ndarray):
    """Compute mean feature values for attack flows vs benign, return top diffs."""
    attack_mask = preds == 1
    if attack_mask.sum() == 0:
        return []

    common = [f for f in TOP_ATTACK_FEATURES if f in df.columns]
    if not common:
        return []

    attack_means  = df.loc[attack_mask, common].mean()
    benign_means  = df.loc[~attack_mask, common].mean()
    diff = (attack_means - benign_means).abs().sort_values(ascending=False)
    top = diff.head(8)
    max_val = top.max() if top.max() > 0 else 1
    return [(feat, val / max_val) for feat, val in top.items()]


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrapper">
    <div class="header-icon">🛡️</div>
    <div>
        <div class="header-title">Hybrid<span>RF-TabNet</span></div>
        <div class="header-sub">Network Intrusion Detection System · Network Traffic Analyzer</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
                color:#fff; margin-bottom:4px; letter-spacing:-0.5px;">
        🛡️ Hybrid RF-TabNet
    </div>
    <div style="font-size:0.65rem; color:#64748b; letter-spacing:2px;
                text-transform:uppercase; margin-bottom:24px; font-family:'Space Mono',monospace;">
        Configuration Panel
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Detection Model</div>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        label="model",
        options=["Hybrid (RF + TabNet)", "Random Forest", "TabNet"],
        label_visibility="collapsed"
    )

    badge_map = {
        "Hybrid (RF + TabNet)": ("badge-hybrid",  "hybrid · rf + tabnet"),
        "Random Forest":         ("badge-rf",     "random forest"),
        "TabNet":                ("badge-tabnet", "tabnet"),
    }
    badge_cls, badge_txt = badge_map[model_choice]
    st.markdown(f'<span class="model-badge {badge_cls}">{badge_txt}</span>', unsafe_allow_html=True)

    st.divider()

    st.markdown('<div class="sidebar-label">Model Files</div>', unsafe_allow_html=True)
    files_needed = {
        "rf_model.pkl":              "🟢" if os.path.exists("rf_model.pkl") else "🔴",
        "tabnet_model.zip":          "🟢" if os.path.exists("tabnet_model.zip") else "🔴",
        "quantile_transformer.pkl":  "🟢" if os.path.exists("quantile_transformer.pkl") else "🔴",
        "hybrid_config.json":        "🟢" if os.path.exists("hybrid_config.json") else "🔴",
    }
    for fname, status in files_needed.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'font-family:Space Mono,monospace;font-size:0.68rem;color:#64748b;'
            f'padding:3px 0;">'
            f'<span>{fname}</span><span>{status}</span></div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.markdown(
        '<div style="font-size:0.65rem;color:#334155;font-family:Space Mono,'
        'monospace;line-height:1.6;">Built on CIC-IDS2017<br/>TabNet + RF Hybrid</div>',
        unsafe_allow_html=True
    )


# ─── Load models (cached) ────────────────────────────────────────────────────
models, load_errors = load_models()
if load_errors:
    with st.expander("⚠️  Model file warnings (demo mode active)", expanded=False):
        for e in load_errors:
            st.warning(e)


# ─── Upload ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Upload Traffic Data</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="Drop a CSV file captured from your network",
    type=["csv"],
    accept_multiple_files=False,
    help="Accepts CIC-IDS2017-formatted CSV files only."
)


# ─── Main analysis ───────────────────────────────────────────────────────────
if uploaded is not None:

    # Step 1 – read CSV
    with st.spinner("📂  Reading CSV file…"):
        time.sleep(0.4)
        try:
            raw_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

    # Step 2 – preprocess
    with st.spinner("⚙️  Preprocessing traffic features…"):
        time.sleep(0.5)
        qt = models.get("qt", None)
        X_raw_df, X_transformed = preprocess(raw_df, qt)

    # Step 3 – inference
    spinner_msg = {
        "Hybrid (RF + TabNet)": "🤖  Running Hybrid (RF + TabNet) inference…",
        "Random Forest":         "🌲  Running Random Forest inference…",
        "TabNet":                "⚡  Running TabNet inference…",
    }[model_choice]

    with st.spinner(spinner_msg):
        time.sleep(0.6)
        preds, probs = predict(X_raw_df, X_transformed, models, model_choice)

    # ── Compute stats ────────────────────────────────────────────────────────
    total        = len(preds)
    benign_count = int((preds == 0).sum())
    attack_count = int((preds == 1).sum())
    benign_pct   = benign_count / total * 100 if total else 0
    attack_pct   = attack_count / total * 100 if total else 0
    conf_scores  = probs.max(axis=1)

    # ── 3 Metric cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Traffic Overview</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card total">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-pct">All flows analyzed</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card benign">
            <div class="metric-label">Benign Traffic</div>
            <div class="metric-value">{benign_count:,}</div>
            <div class="metric-pct">{benign_pct:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card attack">
            <div class="metric-label">Attack Traffic</div>
            <div class="metric-value">{attack_count:,}</div>
            <div class="metric-pct">{attack_pct:.1f}% of total</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts row ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Analysis</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
            'letter-spacing:2px;text-transform:uppercase;color:#64748b;'
            'margin-bottom:1px;">Traffic Distribution</div>',
            unsafe_allow_html=True
        )
        fig = make_bar_chart(benign_count, attack_count)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
            'letter-spacing:2px;text-transform:uppercase;color:#64748b;'
            'margin-bottom:14px;">Suspicious Feature Signals</div>',
            unsafe_allow_html=True
        )

        feat_data = suspicious_features(X_raw_df, preds)

        if feat_data:
            for feat, rel_score in feat_data:
                pct = int(rel_score * 100)
                st.markdown(f"""
                <div class="feature-row">
                    <span class="feature-name">{feat}</span>
                    <div class="feature-bar-wrap">
                        <div class="feature-bar" style="width:{pct}%;"></div>
                    </div>
                    <span style="font-family:Space Mono,monospace;font-size:0.72rem;
                                 color:#ffa502;min-width:36px;text-align:right;">
                        {pct}%
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#64748b;font-size:0.85rem;">'
                'No attack flows detected — no suspicious signals to display.</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Per-flow prediction table ─────────────────────────────────────────────
    st.markdown('<div class="section-title"></div>', unsafe_allow_html=True)

    label_names = ["BENIGN", "ATTACK"]
    result_df = pd.DataFrame({
        "Flow #":       range(1, total + 1),
        "Prediction":   [label_names[p] for p in preds],
        "Confidence":   [f"{s * 100:.1f}%" for s in conf_scores],
        "Attack Prob":  [f"{probs[i, 1] * 100:.1f}%" for i in range(total)],
    })

    # Append a few raw feature columns for context
    preview_cols = [c for c in ["Flow Duration", "Total Fwd Packets",
                                 "Flow Bytes/s", "Flow Packets/s"]
                    if c in X_raw_df.columns]
    for col in preview_cols:
        result_df[col] = X_raw_df[col].values

    # Colour-code prediction column
    def style_prediction(val):
        if val == "ATTACK":
            return "color: #ff4757; font-weight: 700; font-family: Space Mono, monospace;"
        return "color: #2ed573; font-weight: 600; font-family: Space Mono, monospace;"

    styled = result_df.style.applymap(style_prediction, subset=["Prediction"])

    st.dataframe(styled, use_container_width=True, height=420)
    st.dataframe(res.head(1000).style.applymap(_color, subset=scols),
             use_container_width=True, hide_index=True)

    # ── Download ──────────────────────────────────────────────────────────────
    csv_out = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Export Predictions as CSV",
        data=csv_out,
        file_name="hybrid-nids_predictions.csv",
        mime="text/csv"
    )

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 80px 40px; color:#334155;">
        <div style="font-size:3.5rem; margin-bottom:16px; filter:drop-shadow(0 0 20px #00d4ff44);">📡</div>
        <div style="font-family:'Space Mono',monospace; font-size:1.1rem; color:#64748b;
                    letter-spacing:1px; margin-bottom:8px;">
            Awaiting traffic data
        </div>
        <div style="font-size:0.82rem; color:#334155; max-width:360px; margin:0 auto; line-height:1.7;">
            Upload a <strong style="color:#00d4ff;">.csv</strong> file from your network capture
            to begin intrusion detection analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)