"""
Hybrid NIDS (Network Intrusion Detection System) - Streamlit UI
Combines Random Forest + TabNet for traffic classification
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-dark:    #0b0f1a;
    --bg-card:    #111827;
    --bg-panel:   #1a2235;
    --accent:     #00e5ff;
    --accent2:    #ff4d6d;
    --accent3:    #a3e635;
    --text:       #e2e8f0;
    --muted:      #64748b;
    --border:     #1e2d45;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text);
}

/* ── Header ── */
.nids-header {
    background: linear-gradient(135deg, #0b0f1a 0%, #0d1b2a 50%, #0b1622 100%);
    border-bottom: 1px solid var(--border);
    padding: 2rem 2.5rem 1.5rem;
    margin: -1rem -1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.nids-header .logo {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -1px;
    text-shadow: 0 0 20px rgba(0,229,255,0.4);
}
.nids-header .subtitle {
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Stat Tiles ── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.stat-tile {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.stat-tile .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
}
.stat-tile .lbl {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.2rem;
}

/* ── Verdict Banner ── */
.verdict-benign {
    background: linear-gradient(135deg, #052e16, #14532d);
    border: 2px solid #22c55e;
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.verdict-attack {
    background: linear-gradient(135deg, #2d0a0a, #450a0a);
    border: 2px solid var(--accent2);
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.verdict-label {
    font-family: 'Space Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 4px;
}
.verdict-sub {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.5rem;
}

/* ── Progress ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0;
    font-size: 0.9rem;
}
.step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.step-done  { background: #22c55e; }
.step-run   { background: var(--accent); animation: pulse 1s infinite; }
.step-wait  { background: var(--muted); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Misc ── */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 1px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0,229,255,0.35) !important;
}
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nids-header">
  <div>
    <div class="logo">🛡 HYBRID NIDS</div>
    <div class="subtitle">Random Forest + TabNet · Network Intrusion Detection</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    st.markdown("---")

    st.markdown("**Model Files**")
    rf_path      = st.text_input("RF Model",          value="rf_model.pkl")
    tabnet_path  = st.text_input("TabNet Model",       value="tabnet_model.zip")
    config_path  = st.text_input("Hybrid Config",      value="hybrid_config.json")
    scaler_path  = st.text_input("Quantile Transformer", value="quantile_transformer.pkl")

    st.markdown("---")
    st.markdown("**Processing Options**")
    max_flows = st.slider("Max flows to analyse", 10, 5000, 500)
    show_raw  = st.checkbox("Show raw feature table", value=False)
    show_conf = st.checkbox("Show confidence scores", value=True)

    st.markdown("---")
    st.markdown(
        "<small style='color:#64748b'>Upload your `.pcap` to begin. "
        "Models are loaded from the paths above.</small>",
        unsafe_allow_html=True,
    )

# ─── Feature names expected by the model ───────────────────────────────────────
EXPECTED_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio",
    "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets",
    "Subflow Bwd Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std",
    "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

# ─── Helper: load models ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models(rf_p, tab_p, cfg_p, scl_p):
    errors = []
    rf = tabnet = scaler = config = None

    # Random Forest
    if os.path.exists(rf_p):
        try:
            import joblib
            rf = joblib.load(rf_p)
        except Exception as e:
            errors.append(f"RF model: {e}")
    else:
        errors.append(f"RF model file not found: {rf_p}")

    # TabNet
    if os.path.exists(tab_p):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet = TabNetClassifier()
            tabnet.load_model(tab_p)
        except Exception as e:
            errors.append(f"TabNet model: {e}")
    else:
        errors.append(f"TabNet file not found: {tab_p}")

    # Hybrid config
    if os.path.exists(cfg_p):
        try:
            with open(cfg_p) as f:
                config = json.load(f)
        except Exception as e:
            errors.append(f"Hybrid config: {e}")
    else:
        errors.append(f"Hybrid config not found: {cfg_p}")

    # Quantile Transformer
    if os.path.exists(scl_p):
        try:
            import joblib
            scaler = joblib.load(scl_p)
        except Exception as e:
            errors.append(f"Scaler: {e}")
    else:
        errors.append(f"Scaler not found: {scl_p}")

    return rf, tabnet, config, scaler, errors


# ─── Helper: PCAP parsing with Scapy ───────────────────────────────────────────
def parse_pcap(filepath: str, max_flows: int = 500):
    """
    Parse a pcap file using Scapy and reconstruct pseudo-flows.
    Returns (flows_df, raw_packets_info, error_msg).
    """
    try:
        from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    except ImportError:
        return None, None, "Scapy is not installed. Run: pip install scapy"

    try:
        packets = rdpcap(filepath)
    except Exception as e:
        return None, None, f"Could not read PCAP: {e}"

    if len(packets) == 0:
        return None, None, "PCAP file contains no packets."

    # ── Raw packet-level info for the traffic overview ──
    raw_info = {
        "total_packets": len(packets),
        "src_ips": Counter(),
        "dst_ips": Counter(),
        "protocols": Counter(),
        "total_bytes": 0,
        "src_ports": Counter(),
        "dst_ports": Counter(),
    }

    # ── Flow tracking ──
    flows = {}  # key → list of (ts, size, flags, sport, dport)

    for pkt in packets:
        ts   = float(pkt.time)
        size = len(pkt)
        raw_info["total_bytes"] += size

        if IP not in pkt:
            raw_info["protocols"]["Other"] += 1
            continue

        src = pkt[IP].src
        dst = pkt[IP].dst
        raw_info["src_ips"][src] += 1
        raw_info["dst_ips"][dst] += 1

        sport = dport = 0
        flags = 0
        proto = "Other"

        if TCP in pkt:
            proto  = "TCP"
            sport  = pkt[TCP].sport
            dport  = pkt[TCP].dport
            flags  = int(pkt[TCP].flags)
            raw_info["src_ports"][sport] += 1
            raw_info["dst_ports"][dport] += 1
        elif UDP in pkt:
            proto  = "UDP"
            sport  = pkt[UDP].sport
            dport  = pkt[UDP].dport
            raw_info["src_ports"][sport] += 1
            raw_info["dst_ports"][dport] += 1
        elif ICMP in pkt:
            proto  = "ICMP"

        raw_info["protocols"][proto] += 1

        fkey = (src, dst, sport, dport, proto)
        if fkey not in flows:
            flows[fkey] = {"ts": [], "sizes": [], "flags": [], "proto": proto}
        flows[fkey]["ts"].append(ts)
        flows[fkey]["sizes"].append(size)
        flows[fkey]["flags"].append(flags)

    if not flows:
        return None, raw_info, "No IP flows could be extracted from the PCAP."

    # ── Compute flow features ──
    records = []
    for (src, dst, sport, dport, proto), data in list(flows.items())[:max_flows]:
        ts_arr  = np.array(sorted(data["ts"]))
        sz_arr  = np.array(data["sizes"])
        fl_arr  = np.array(data["flags"])
        n       = len(ts_arr)
        duration = float(ts_arr[-1] - ts_arr[0]) if n > 1 else 0.0
        dur_safe = duration if duration > 0 else 1e-6

        iats     = np.diff(ts_arr) if n > 1 else np.array([0.0])
        fwd_sz   = sz_arr[::2]   # odd indices → forward
        bwd_sz   = sz_arr[1::2]  # even indices → backward

        def safe(arr, fn):
            try:
                return float(fn(arr)) if len(arr) > 0 else 0.0
            except Exception:
                return 0.0

        # TCP flag counts (bit positions: FIN=0,SYN=1,RST=2,PSH=3,ACK=4,URG=5,ECE=6,CWE=7)
        tcp_flags = fl_arr if proto == "TCP" else np.zeros(n, dtype=int)
        fin_cnt = int(np.sum((tcp_flags & 0x01) > 0))
        syn_cnt = int(np.sum((tcp_flags & 0x02) > 0))
        rst_cnt = int(np.sum((tcp_flags & 0x04) > 0))
        psh_cnt = int(np.sum((tcp_flags & 0x08) > 0))
        ack_cnt = int(np.sum((tcp_flags & 0x10) > 0))
        urg_cnt = int(np.sum((tcp_flags & 0x20) > 0))
        ece_cnt = int(np.sum((tcp_flags & 0x40) > 0))
        cwe_cnt = int(np.sum((tcp_flags & 0x80) > 0))

        total_fwd = max(len(fwd_sz), 1)
        total_bwd = max(len(bwd_sz), 1)

        rec = {
            "Destination Port":           dport,
            "Flow Duration":              duration * 1e6,   # microseconds
            "Total Fwd Packets":          total_fwd,
            "Total Backward Packets":     total_bwd,
            "Total Length of Fwd Packets": safe(fwd_sz, np.sum),
            "Total Length of Bwd Packets": safe(bwd_sz, np.sum),
            "Fwd Packet Length Max":      safe(fwd_sz, np.max),
            "Fwd Packet Length Min":      safe(fwd_sz, np.min),
            "Fwd Packet Length Mean":     safe(fwd_sz, np.mean),
            "Fwd Packet Length Std":      safe(fwd_sz, np.std),
            "Bwd Packet Length Max":      safe(bwd_sz, np.max),
            "Bwd Packet Length Min":      safe(bwd_sz, np.min),
            "Bwd Packet Length Mean":     safe(bwd_sz, np.mean),
            "Bwd Packet Length Std":      safe(bwd_sz, np.std),
            "Flow Bytes/s":               safe(sz_arr, np.sum) / dur_safe,
            "Flow Packets/s":             n / dur_safe,
            "Flow IAT Mean":              safe(iats, np.mean),
            "Flow IAT Std":               safe(iats, np.std),
            "Flow IAT Max":               safe(iats, np.max),
            "Flow IAT Min":               safe(iats, np.min),
            "Fwd IAT Total":              safe(iats, np.sum),
            "Fwd IAT Mean":               safe(iats, np.mean),
            "Fwd IAT Std":                safe(iats, np.std),
            "Fwd IAT Max":                safe(iats, np.max),
            "Fwd IAT Min":                safe(iats, np.min),
            "Bwd IAT Total":              safe(iats, np.sum),
            "Bwd IAT Mean":               safe(iats, np.mean),
            "Bwd IAT Std":                safe(iats, np.std),
            "Bwd IAT Max":                safe(iats, np.max),
            "Bwd IAT Min":                safe(iats, np.min),
            "Fwd PSH Flags":              psh_cnt,
            "Bwd PSH Flags":              0,
            "Fwd URG Flags":              urg_cnt,
            "Bwd URG Flags":              0,
            "Fwd Header Length":          total_fwd * 20,
            "Bwd Header Length":          total_bwd * 20,
            "Fwd Packets/s":              total_fwd / dur_safe,
            "Bwd Packets/s":              total_bwd / dur_safe,
            "Min Packet Length":          safe(sz_arr, np.min),
            "Max Packet Length":          safe(sz_arr, np.max),
            "Packet Length Mean":         safe(sz_arr, np.mean),
            "Packet Length Std":          safe(sz_arr, np.std),
            "Packet Length Variance":     safe(sz_arr, np.var),
            "FIN Flag Count":             fin_cnt,
            "SYN Flag Count":             syn_cnt,
            "RST Flag Count":             rst_cnt,
            "PSH Flag Count":             psh_cnt,
            "ACK Flag Count":             ack_cnt,
            "URG Flag Count":             urg_cnt,
            "CWE Flag Count":             cwe_cnt,
            "ECE Flag Count":             ece_cnt,
            "Down/Up Ratio":              total_bwd / total_fwd,
            "Average Packet Size":        safe(sz_arr, np.mean),
            "Avg Fwd Segment Size":       safe(fwd_sz, np.mean),
            "Avg Bwd Segment Size":       safe(bwd_sz, np.mean),
            "Subflow Fwd Packets":        total_fwd,
            "Subflow Fwd Bytes":          safe(fwd_sz, np.sum),
            "Subflow Bwd Packets":        total_bwd,
            "Subflow Bwd Bytes":          safe(bwd_sz, np.sum),
            "Init_Win_bytes_forward":     65535,
            "Init_Win_bytes_backward":    65535,
            "act_data_pkt_fwd":           total_fwd,
            "min_seg_size_forward":       safe(fwd_sz, np.min),
            "Active Mean":                safe(iats, np.mean),
            "Active Std":                 safe(iats, np.std),
            "Active Max":                 safe(iats, np.max),
            "Active Min":                 safe(iats, np.min),
            "Idle Mean":                  safe(iats, np.mean),
            "Idle Std":                   safe(iats, np.std),
            "Idle Max":                   safe(iats, np.max),
            "Idle Min":                   safe(iats, np.min),
            # metadata (not fed to model)
            "_src_ip":  src,
            "_dst_ip":  dst,
            "_proto":   proto,
        }
        records.append(rec)

    if not records:
        return None, raw_info, "No flow records could be built."

    df = pd.DataFrame(records)
    return df, raw_info, None

def preprocess(df: pd.DataFrame, scaler, expected_features: list):
    # Drop metadata columns
    meta_cols = [c for c in df.columns if c.startswith("_")]
    feat_df = df.drop(columns=meta_cols, errors="ignore")

    # Ensure all expected features exist
    for col in expected_features:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    # Reorder columns
    feat_df = feat_df[expected_features].copy()

    # Replace invalid values
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    # Apply QuantileTransformer if available
    if scaler is not None:
        # ⚡ This ensures shape exactly matches scaler
        X = feat_df.values
        if X.shape[1] != scaler.n_features_in_:
            # pad or trim
            n_diff = scaler.n_features_in_ - X.shape[1]
            if n_diff > 0:
                X = np.hstack([X, np.zeros((X.shape[0], n_diff), dtype=np.float32)])
            else:
                X = X[:, :scaler.n_features_in_]
        arr = scaler.transform(X)
    else:
        arr = feat_df.values

    return arr

# ─── Helper: hybrid predict ────────────────────────────────────────────────────
def hybrid_predict(X, rf_model, tabnet_model, config):
    rf_weight     = config.get("rf_weight", 0.5)
    tabnet_weight = config.get("tabnet_weight", 0.5)

    rf_probs     = rf_model.predict_proba(X)
    tabnet_probs = tabnet_model.predict_proba(X)

    # align class counts
    n_classes = max(rf_probs.shape[1], tabnet_probs.shape[1])
    def pad(p):
        if p.shape[1] < n_classes:
            pad_w = np.zeros((p.shape[0], n_classes - p.shape[1]))
            return np.hstack([p, pad_w])
        return p
    rf_probs     = pad(rf_probs)
    tabnet_probs = pad(tabnet_probs)

    hybrid_probs = (rf_weight * rf_probs) + (tabnet_weight * tabnet_probs)
    hybrid_preds = np.argmax(hybrid_probs, axis=1)
    return hybrid_preds, hybrid_probs


# ─── Plots ─────────────────────────────────────────────────────────────────────
def plot_protocol_pie(protocols: Counter):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    labels = list(protocols.keys())
    sizes  = list(protocols.values())

    COLORS = ["#00e5ff", "#ff4d6d", "#a3e635", "#f59e0b", "#8b5cf6", "#ec4899"]
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="#111827")
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        colors=COLORS[:len(labels)],
        pctdistance=0.8,
        wedgeprops=dict(linewidth=2, edgecolor="#0b0f1a"),
        startangle=140,
    )
    for at in autotexts:
        at.set_color("#0b0f1a")
        at.set_fontsize(9)
        at.set_fontweight("bold")

    patches = [mpatches.Patch(color=COLORS[i % len(COLORS)], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, frameon=False,
              labelcolor="#e2e8f0", fontsize=8)
    ax.set_title("Protocol Distribution", color="#00e5ff", fontfamily="monospace", fontsize=10, pad=12)
    fig.tight_layout()
    return fig


def plot_top_talkers(src_ips: Counter, n: int = 8):
    import matplotlib.pyplot as plt

    top = src_ips.most_common(n)
    ips, counts = zip(*top) if top else ([], [])

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#111827")
    bars = ax.barh(list(ips)[::-1], list(counts)[::-1],
                   color="#00e5ff", alpha=0.85, height=0.6)
    ax.set_facecolor("#111827")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.tick_params(colors="#e2e8f0", labelsize=8)
    ax.xaxis.label.set_color("#64748b")
    ax.set_xlabel("Packet Count", color="#64748b", fontsize=8)
    ax.set_title("Top Source IPs", color="#00e5ff", fontfamily="monospace", fontsize=10, pad=10)
    fig.tight_layout()
    return fig


def plot_prediction_bar(hybrid_probs: np.ndarray, class_names=("BENIGN", "ATTACK")):
    import matplotlib.pyplot as plt

    mean_probs = hybrid_probs.mean(axis=0)
    # clip to available classes
    mean_probs = mean_probs[:len(class_names)]

    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor="#111827")
    colors = ["#22c55e", "#ff4d6d"]
    bars = ax.bar(class_names[:len(mean_probs)], mean_probs,
                  color=colors[:len(mean_probs)], width=0.4)
    ax.set_facecolor("#111827")
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d45")
    ax.tick_params(colors="#e2e8f0", labelsize=9)
    ax.set_ylabel("Mean Probability", color="#64748b", fontsize=8)
    ax.set_title("Hybrid Model Confidence", color="#00e5ff", fontfamily="monospace", fontsize=10, pad=10)
    for bar, val in zip(bars, mean_probs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom", color="#e2e8f0", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ─── Main UI ────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section 1 : Upload ─────────────────────────────────────────────────────────
st.markdown('<div class="card-title">📂 &nbsp;STEP 1 · UPLOAD PCAP</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drop a `.pcap` file to begin analysis",
    type=["pcap", "pcapng"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("👆  Upload a `.pcap` or `.pcapng` file to start.", icon="ℹ️")
    st.stop()

# ── Save to temp file ──────────────────────────────────────────────────────────
with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

run_btn = st.button("🔍  Analyse Traffic", use_container_width=False)

if not run_btn:
    st.markdown(
        f"<small style='color:#64748b'>File ready: <b>{uploaded.name}</b> "
        f"({uploaded.size / 1024:.1f} KB) — click <b>Analyse Traffic</b> to proceed.</small>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Pipeline execution ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="card-title">⚡ &nbsp;PROCESSING PIPELINE</div>', unsafe_allow_html=True)

steps = [
    "Parse PCAP & extract flows",
    "Compute network features",
    "Load & apply quantile transformer",
    "Run Random Forest",
    "Run TabNet",
    "Compute hybrid prediction",
]
step_placeholder = st.empty()

def render_steps(done: int, running: int):
    html = ""
    for i, s in enumerate(steps):
        if i < done:
            dot  = "step-dot step-done"
            clr  = "#22c55e"
        elif i == running:
            dot  = "step-dot step-run"
            clr  = "#00e5ff"
        else:
            dot  = "step-dot step-wait"
            clr  = "#64748b"
        html += f'<div class="step-row"><div class="{dot}"></div><span style="color:{clr}">{s}</span></div>'
    step_placeholder.markdown(html, unsafe_allow_html=True)

# ── Step 0 & 1: Parse PCAP ────────────────────────────────────────────────────
render_steps(0, 0)
time.sleep(0.3)
flow_df, raw_info, err = parse_pcap(tmp_path, max_flows=max_flows)

if err:
    st.error(f"❌ PCAP Parsing Error: {err}")
    os.unlink(tmp_path)
    st.stop()

render_steps(2, 2)
time.sleep(0.2)

# ── Step 2: Load models & scaler ──────────────────────────────────────────────
render_steps(2, 2)
rf_model, tabnet_model, hybrid_config, scaler, model_errors = load_models(
    rf_path, tabnet_path, config_path, scaler_path
)

if scaler is None:
    st.warning("⚠️ Quantile transformer not loaded — raw features will be used.")

# Apply transformer
render_steps(2, 2)
try:
    X = preprocess(flow_df, scaler, EXPECTED_FEATURES)
except Exception as e:
    st.error(f"❌ Feature preprocessing failed: {e}")
    os.unlink(tmp_path)
    st.stop()

render_steps(3, 3)
time.sleep(0.2)

# ── Step 3-5: Predict ─────────────────────────────────────────────────────────
predictions = None
hybrid_probs = None
prediction_error = None

if rf_model is None or tabnet_model is None or hybrid_config is None:
    prediction_error = "One or more model files could not be loaded:\n" + "\n".join(model_errors)
else:
    render_steps(3, 3)
    try:
        predictions, hybrid_probs = hybrid_predict(X, rf_model, tabnet_model, hybrid_config)
        render_steps(6, -1)
    except Exception as e:
        prediction_error = f"Prediction failed: {e}"

os.unlink(tmp_path)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── Section 2 : Traffic Overview ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="card-title">📊 &nbsp;STEP 2 · TRAFFIC OVERVIEW</div>', unsafe_allow_html=True)

# Stat tiles
n_flows    = len(flow_df)
unique_src = len(raw_info["src_ips"])
unique_dst = len(raw_info["dst_ips"])
total_pkts = raw_info["total_packets"]
total_mb   = raw_info["total_bytes"] / 1_048_576

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-tile"><div class="val">{total_pkts:,}</div><div class="lbl">Total Packets</div></div>
  <div class="stat-tile"><div class="val">{n_flows:,}</div><div class="lbl">Flows Analysed</div></div>
  <div class="stat-tile"><div class="val">{unique_src}</div><div class="lbl">Unique Src IPs</div></div>
  <div class="stat-tile"><div class="val">{unique_dst}</div><div class="lbl">Unique Dst IPs</div></div>
  <div class="stat-tile"><div class="val">{total_mb:.2f} MB</div><div class="lbl">Total Bytes</div></div>
  <div class="stat-tile"><div class="val">{len(raw_info['protocols'])}</div><div class="lbl">Protocols</div></div>
</div>
""", unsafe_allow_html=True)

# Charts
col1, col2 = st.columns([1, 1.6])

with col1:
    fig_pie = plot_protocol_pie(raw_info["protocols"])
    st.pyplot(fig_pie, use_container_width=True)

with col2:
    fig_bar = plot_top_talkers(raw_info["src_ips"])
    st.pyplot(fig_bar, use_container_width=True)

# Top talkers table
with st.expander("📋  Top 10 Source → Destination Pairs"):
    pairs = Counter()
    for _, row in flow_df.iterrows():
        pairs[(row.get("_src_ip", "?"), row.get("_dst_ip", "?"), row.get("_proto", "?"))] += 1
    # Build DataFrame from top 10 source→destination→protocol flows
    top_pairs = pairs.most_common(10)
    pair_df = pd.DataFrame(
        [(src, dst, proto, count) for (src, dst, proto), count in top_pairs],
        columns=["Src IP", "Dst IP", "Protocol", "Flows"]
    )
    st.dataframe(pair_df, use_container_width=True, hide_index=True)
    
if show_raw:
    with st.expander("🔬  Raw Feature Table (first 50 rows)"):
        feat_cols = [c for c in flow_df.columns if not c.startswith("_")]
        st.dataframe(flow_df[feat_cols].head(50), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ─── Section 3 : Prediction Results ────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="card-title">🎯 &nbsp;STEP 3 · PREDICTION RESULTS</div>', unsafe_allow_html=True)

if prediction_error:
    st.error(f"❌ {prediction_error}")
    st.info("📝 Tip: Make sure your model files are in the working directory and paths in the sidebar are correct.")
    st.stop()

# ── Aggregate verdict ──────────────────────────────────────────────────────────
attack_ratio = float(np.mean(predictions == 1))
benign_ratio = 1.0 - attack_ratio

overall = "ATTACK" if attack_ratio >= 0.5 else "BENIGN"
n_attack = int(np.sum(predictions == 1))
n_benign = int(np.sum(predictions == 0))

if overall == "BENIGN":
    st.markdown(f"""
    <div class="verdict-benign">
      <div class="verdict-label" style="color:#22c55e">✅ &nbsp; BENIGN</div>
      <div class="verdict-sub">Traffic classified as normal · {n_benign} benign / {n_attack} suspicious flows</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-attack">
      <div class="verdict-label" style="color:#ff4d6d">🚨 &nbsp; ATTACK DETECTED</div>
      <div class="verdict-sub">Malicious traffic identified · {n_attack} attack / {n_benign} benign flows</div>
    </div>
    """, unsafe_allow_html=True)

# ── Confidence breakdown ───────────────────────────────────────────────────────
if show_conf and hybrid_probs is not None:
    col_c1, col_c2 = st.columns([1.4, 1])
    with col_c1:
        fig_conf = plot_prediction_bar(hybrid_probs)
        st.pyplot(fig_conf, use_container_width=True)
    with col_c2:
        rf_w  = hybrid_config.get("rf_weight", 0.5)
        tab_w = hybrid_config.get("tabnet_weight", 0.5)
        st.markdown(f"""
        <div class="card">
          <div class="card-title">🔧 Hybrid Weights</div>
          <div style="margin-bottom:0.6rem">
            <span style="color:#64748b;font-size:0.8rem">RANDOM FOREST</span><br>
            <span style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#00e5ff">{rf_w:.2f}</span>
          </div>
          <div>
            <span style="color:#64748b;font-size:0.8rem">TABNET</span><br>
            <span style="font-family:'Space Mono',monospace;font-size:1.4rem;color:#a3e635">{tab_w:.2f}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="card">
          <div class="card-title">📈 Flow Summary</div>
          <div style="display:flex;gap:1.5rem">
            <div>
              <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:#22c55e">{n_benign}</div>
              <div style="font-size:0.75rem;color:#64748b">BENIGN</div>
            </div>
            <div>
              <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:#ff4d6d">{n_attack}</div>
              <div style="font-size:0.75rem;color:#64748b">ATTACK</div>
            </div>
            <div>
              <div style="font-family:'Space Mono',monospace;font-size:1.3rem;color:#f59e0b">{attack_ratio*100:.1f}%</div>
              <div style="font-size:0.75rem;color:#64748b">THREAT RATE</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Per-flow table ─────────────────────────────────────────────────────────────
with st.expander("📋  Per-Flow Prediction Detail"):
    label_map = {0: "BENIGN", 1: "ATTACK"}
    flow_results = flow_df[["_src_ip", "_dst_ip", "_proto"]].copy()
    flow_results.columns = ["Src IP", "Dst IP", "Protocol"]
    flow_results["Prediction"] = [label_map.get(p, str(p)) for p in predictions]
    if hybrid_probs is not None:
        flow_results["Benign Prob"]  = hybrid_probs[:len(predictions), 0].round(4)
        flow_results["Attack Prob"]  = hybrid_probs[:len(predictions), 1].round(4) if hybrid_probs.shape[1] > 1 else 1 - hybrid_probs[:len(predictions), 0].round(4)

    def color_pred(val):
        if val == "ATTACK":
            return "background-color: #450a0a; color: #ff4d6d; font-weight: bold;"
        return "background-color: #052e16; color: #22c55e; font-weight: bold;"

    styled = flow_results.style.applymap(color_pred, subset=["Prediction"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#374151'>Hybrid NIDS · Random Forest + TabNet · "
    "Feature extraction via Scapy · Quantile Transformer preprocessing</small>",
    unsafe_allow_html=True,
)