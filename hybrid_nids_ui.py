"""
Hybrid NIDS — Streamlit UI
Random Forest + TabNet · Network Intrusion Detection System
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import tempfile
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ─── Constants ─────────────────────────────────────────────────────────────────
RF_PATH     = "rf_model.pkl"
TABNET_PATH = "tabnet_model.zip"
CONFIG_PATH = "hybrid_config.json"
SCALER_PATH = "quantile_transformer.pkl"

EXPECTED_FEATURES = [
    "Destination Port","Flow Duration","Total Fwd Packets","Total Backward Packets",
    "Total Length of Fwd Packets","Total Length of Bwd Packets","Fwd Packet Length Max",
    "Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std",
    "Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean",
    "Bwd Packet Length Std","Flow Bytes/s","Flow Packets/s","Flow IAT Mean",
    "Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean",
    "Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean",
    "Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd PSH Flags","Bwd PSH Flags",
    "Fwd URG Flags","Bwd URG Flags","Fwd Header Length","Bwd Header Length",
    "Fwd Packets/s","Bwd Packets/s","Min Packet Length","Max Packet Length",
    "Packet Length Mean","Packet Length Std","Packet Length Variance","FIN Flag Count",
    "SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count",
    "CWE Flag Count","ECE Flag Count","Down/Up Ratio","Average Packet Size",
    "Avg Fwd Segment Size","Avg Bwd Segment Size","Subflow Fwd Packets","Subflow Fwd Bytes",
    "Subflow Bwd Packets","Subflow Bwd Bytes","Init_Win_bytes_forward",
    "Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward",
    "Active Mean","Active Std","Active Max","Active Min",
    "Idle Mean","Idle Std","Idle Max","Idle Min",
]
LABEL_COLUMNS = ["Label","label"," Label","Class","class","Attack"]

# ── Design tokens ──────────────────────────────────────────────────────────────
C_BG       = "#0b0f1a"
C_CARD     = "#111827"
C_PANEL    = "#1a2235"
C_BORDER   = "#1e2d45"
C_ACCENT   = "#00e5ff"
C_GREEN    = "#22c55e"
C_RED      = "#ff4d6d"
C_AMBER    = "#f59e0b"
C_TEXT     = "#e2e8f0"
C_MUTED    = "#64748b"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Tokens ── */
:root {{
    --bg:      {C_BG};
    --card:    {C_CARD};
    --panel:   {C_PANEL};
    --border:  {C_BORDER};
    --accent:  {C_ACCENT};
    --green:   {C_GREEN};
    --red:     {C_RED};
    --amber:   {C_AMBER};
    --text:    {C_TEXT};
    --muted:   {C_MUTED};
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
    --r:       10px;
    --r-lg:    14px;
}}

/* ── Base ── */
html, body, [class*="css"] {{
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
}}

/* ════════ HEADER ════════ */
.header {{
    padding: 1.75rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
}}
.header-wordmark {{
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 0.65rem;
}}
.header-pulse {{
    width: 9px; height: 9px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
    flex-shrink: 0;
}}
.header-sub {{
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 0.3rem;
    letter-spacing: 0.2px;
}}
.header-tag {{
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

/* ════════ SECTION HEADING ════════ */
.sh {{
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.75rem 0 1.1rem;
}}
.sh-text {{
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    white-space: nowrap;
}}
.sh-rule {{
    flex: 1;
    height: 1px;
    background: var(--border);
}}

/* ════════ STAT CARDS ════════ */
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}}
.stat-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.2rem 1.25rem 1rem;
    position: relative;
    overflow: hidden;
}}
.stat-accent-bar {{
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}}
.stat-label {{
    font-size: 0.68rem;
    font-weight: 500;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.55rem;
}}
.stat-value {{
    font-family: var(--mono);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text);
}}
.stat-hint {{
    font-size: 0.7rem;
    color: var(--muted);
    margin-top: 0.4rem;
    line-height: 1.3;
}}

/* ════════ CHART CARD ════════ */
.chart-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1.25rem 1.25rem 0.5rem;
    margin-bottom: 0.75rem;
}}
.chart-label {{
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
}}

/* ════════ IP TABLE ════════ */
.ip-section {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--r);
    overflow: hidden;
    margin-bottom: 1.5rem;
}}
.ip-section-header {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    border-bottom: 1px solid var(--border);
}}
.ip-col-head {{
    padding: 0.75rem 1.25rem;
    font-family: var(--mono);
    font-size: 0.6rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: var(--panel);
}}
.ip-col-head:first-child {{ border-right: 1px solid var(--border); }}
.ip-dot {{
    width: 6px; height: 6px;
    border-radius: 50%;
    flex-shrink: 0;
}}
.ip-body {{
    display: grid;
    grid-template-columns: 1fr 1fr;
}}
.ip-col {{}}
.ip-col:first-child {{ border-right: 1px solid var(--border); }}
.ip-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 1.25rem;
    border-bottom: 1px solid rgba(30,45,69,0.5);
    transition: background 0.1s;
}}
.ip-row:last-child {{ border-bottom: none; }}
.ip-row:hover {{ background: var(--panel); }}
.ip-addr {{
    font-family: var(--mono);
    font-size: 0.76rem;
    color: var(--text);
}}
.ip-count {{
    font-family: var(--mono);
    font-size: 0.68rem;
    padding: 0.15rem 0.55rem;
    border-radius: 5px;
    font-weight: 700;
}}
.ip-count.red   {{ background: rgba(255,77,109,0.12); color: var(--red);   border: 1px solid rgba(255,77,109,0.2); }}
.ip-count.amber {{ background: rgba(245,158,11,0.12); color: var(--amber); border: 1px solid rgba(245,158,11,0.2); }}
.ip-empty {{
    padding: 1.5rem 1.25rem;
    font-size: 0.8rem;
    color: var(--muted);
    text-align: center;
}}
.no-attacks-banner {{
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green);
    border-radius: var(--r);
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: var(--green);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}}

/* ════════ VERDICT ════════ */
.verdict {{
    border-radius: var(--r-lg);
    padding: 1.75rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    margin-bottom: 1.25rem;
}}
.verdict-benign {{
    background: linear-gradient(135deg, #040e08 0%, #071510 100%);
    border: 1px solid rgba(34,197,94,0.3);
}}
.verdict-attack {{
    background: linear-gradient(135deg, #0f0407 0%, #160608 100%);
    border: 1px solid rgba(255,77,109,0.3);
}}
.verdict-left {{ display: flex; align-items: center; gap: 1.1rem; }}
.verdict-icon {{ font-size: 1.8rem; line-height: 1; }}
.verdict-title {{
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 2px;
}}
.verdict-desc {{ font-size: 0.78rem; color: var(--muted); margin-top: 0.3rem; }}
.verdict-meta {{
    display: flex;
    align-items: center;
    gap: 1.5rem;
}}
.vm-item {{ text-align: right; }}
.vm-val {{
    font-family: var(--mono);
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1;
}}
.vm-lbl {{
    font-size: 0.62rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.25rem;
}}
.vm-sep {{ width:1px; height:36px; background: var(--border); }}

/* ════════ SPINNER ════════ */
.spin-outer {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3.5rem 0;
    gap: 1.1rem;
}}
.spin-ring {{
    width: 38px; height: 38px;
    border: 2.5px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: _spin 0.75s linear infinite;
}}
@keyframes _spin {{ to {{ transform: rotate(360deg); }} }}
.spin-text {{
    font-family: var(--mono);
    font-size: 0.68rem;
    color: var(--accent);
    letter-spacing: 2px;
    text-transform: uppercase;
}}

/* ════════ SIDEBAR ════════ */
section[data-testid="stSidebar"] {{
    background: var(--card) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] label {{ color: var(--text) !important; }}
.sb-title {{
    font-family: var(--mono);
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.25rem;
}}
.sb-group {{
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.2rem 0 0.5rem;
}}

/* ════════ BUTTON ════════ */
.stButton > button {{
    background: var(--accent) !important;
    color: #020b12 !important;
    font-family: var(--mono) !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: var(--r) !important;
    padding: 0.65rem 1.75rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.15s !important;
}}
.stButton > button:hover {{
    background: #7de8ff !important;
    box-shadow: 0 0 20px rgba(0,229,255,0.2) !important;
    transform: translateY(-1px) !important;
}}

/* ════════ EXPANDER ════════ */
div[data-testid="stExpander"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden;
}}
div[data-testid="stExpander"] summary {{
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--text) !important;
    letter-spacing: 0.5px;
    padding: 0.85rem 1rem !important;
}}
div[data-testid="stExpander"] > div[role="region"] {{
    border-top: 1px solid var(--border);
    padding: 1rem !important;
}}

/* ════════ ALERTS ════════ */
.stAlert {{ border-radius: var(--r) !important; }}

/* ════════ MISC ════════ */
hr {{ border-color: var(--border) !important; margin: 1.5rem 0 !important; }}
[data-testid="stFileUploader"] {{
    background: var(--card) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--r-lg) !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: var(--accent) !important; }}
</style>
""", unsafe_allow_html=True)


# ─── Chart helper: set matplotlib dark theme ──────────────────────────────────
def _mpl_theme():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": C_CARD, "axes.facecolor": C_CARD,
        "axes.edgecolor": C_BORDER, "axes.labelcolor": C_MUTED,
        "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        "text.color": C_TEXT, "grid.color": C_PANEL,
        "font.family": "sans-serif",
    })


# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div>
    <div class="header-wordmark">
      <div class="header-pulse"></div>
      Hybrid NIDS
    </div>
    <div class="header-sub">Network Intrusion Detection &nbsp;·&nbsp; Random Forest + TabNet Ensemble</div>
  </div>
  <div class="header-tag">v2.0 · ML Classifier</div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">⚙ &nbsp;Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-group">Detection Model</div>', unsafe_allow_html=True)
    model_choice = st.radio(
        "", options=["Hybrid RF-TabNet"],
        index=0, label_visibility="collapsed",
    )
    st.markdown('<div class="sb-group">Display</div>', unsafe_allow_html=True)
    show_raw  = st.checkbox("Show raw feature table", value=False)
    show_conf = st.checkbox("Show confidence chart", value=True)
    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.72rem;color:#4d6480'>"
        "Accepts <b style='color:#8ba3be'>.pcap</b> / <b style='color:#8ba3be'>.pcapng</b></span>",
        unsafe_allow_html=True,
    )


# ─── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    errs, rf, tabnet, scaler, cfg = [], None, None, None, None
    if os.path.exists(RF_PATH):
        try:
            import joblib; rf = joblib.load(RF_PATH)
        except Exception as e: errs.append(f"RF: {e}")
    else: errs.append(f"RF not found ({RF_PATH})")

    if os.path.exists(TABNET_PATH):
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
            tabnet = TabNetClassifier(); tabnet.load_model(TABNET_PATH)
        except Exception as e: errs.append(f"TabNet: {e}")
    else: errs.append(f"TabNet not found ({TABNET_PATH})")

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f: cfg = json.load(f)
        except Exception as e: errs.append(f"Config: {e}")
    else: errs.append(f"Config not found ({CONFIG_PATH})")

    if os.path.exists(SCALER_PATH):
        try:
            import joblib; scaler = joblib.load(SCALER_PATH)
        except Exception as e: errs.append(f"Scaler: {e}")
    else: errs.append(f"Scaler not found ({SCALER_PATH})")
    return rf, tabnet, cfg, scaler, errs


# ─── PCAP Parsing ──────────────────────────────────────────────────────────────
def parse_pcap(filepath):
    try: from scapy.all import rdpcap, IP, TCP, UDP, ICMP
    except ImportError: return None, None, "Scapy not installed — run: pip install scapy"
    try: packets = rdpcap(filepath)
    except Exception as e: return None, None, f"Cannot read PCAP: {e}"
    if not packets: return None, None, "PCAP contains no packets."

    raw = {"total_packets": len(packets), "src_ips": Counter(),
           "dst_ips": Counter(), "protocols": Counter(), "total_bytes": 0}
    flows = {}
    for pkt in packets:
        ts, sz = float(pkt.time), len(pkt)
        raw["total_bytes"] += sz
        if IP not in pkt: raw["protocols"]["Other"] += 1; continue
        src, dst = pkt[IP].src, pkt[IP].dst
        raw["src_ips"][src] += 1; raw["dst_ips"][dst] += 1
        sp = dp = fl = 0; proto = "Other"
        if TCP in pkt:   proto, sp, dp, fl = "TCP", pkt[TCP].sport, pkt[TCP].dport, int(pkt[TCP].flags)
        elif UDP in pkt: proto, sp, dp    = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif ICMP in pkt: proto = "ICMP"
        raw["protocols"][proto] += 1
        k = (src, dst, sp, dp, proto)
        if k not in flows: flows[k] = {"ts":[],"sizes":[],"flags":[],"proto":proto}
        flows[k]["ts"].append(ts); flows[k]["sizes"].append(sz); flows[k]["flags"].append(fl)
    if not flows: return None, raw, "No IP flows extracted."

    records = []
    for (src, dst, sp, dp, proto), d in flows.items():
        ts_a = np.array(sorted(d["ts"])); sz_a = np.array(d["sizes"]); fl_a = np.array(d["flags"])
        n = len(ts_a)
        dur = float(ts_a[-1]-ts_a[0]) if n>1 else 0.0
        ds = dur if dur>0 else 1e-6
        iat = np.diff(ts_a) if n>1 else np.array([0.0])
        fw, bw = sz_a[::2], sz_a[1::2]
        def s(a, fn):
            try: return float(fn(a)) if len(a)>0 else 0.0
            except: return 0.0
        tf = fl_a if proto=="TCP" else np.zeros(n,int)
        fin=int(np.sum((tf&1)>0)); syn=int(np.sum((tf&2)>0)); rst=int(np.sum((tf&4)>0))
        psh=int(np.sum((tf&8)>0)); ack=int(np.sum((tf&16)>0)); urg=int(np.sum((tf&32)>0))
        ece=int(np.sum((tf&64)>0)); cwe=int(np.sum((tf&128)>0))
        nfw=max(len(fw),1); nbw=max(len(bw),1)
        records.append({
            "Destination Port":dp,"Flow Duration":dur*1e6,"Total Fwd Packets":nfw,
            "Total Backward Packets":nbw,"Total Length of Fwd Packets":s(fw,np.sum),
            "Total Length of Bwd Packets":s(bw,np.sum),
            "Fwd Packet Length Max":s(fw,np.max),"Fwd Packet Length Min":s(fw,np.min),
            "Fwd Packet Length Mean":s(fw,np.mean),"Fwd Packet Length Std":s(fw,np.std),
            "Bwd Packet Length Max":s(bw,np.max),"Bwd Packet Length Min":s(bw,np.min),
            "Bwd Packet Length Mean":s(bw,np.mean),"Bwd Packet Length Std":s(bw,np.std),
            "Flow Bytes/s":s(sz_a,np.sum)/ds,"Flow Packets/s":n/ds,
            "Flow IAT Mean":s(iat,np.mean),"Flow IAT Std":s(iat,np.std),
            "Flow IAT Max":s(iat,np.max),"Flow IAT Min":s(iat,np.min),
            "Fwd IAT Total":s(iat,np.sum),"Fwd IAT Mean":s(iat,np.mean),
            "Fwd IAT Std":s(iat,np.std),"Fwd IAT Max":s(iat,np.max),"Fwd IAT Min":s(iat,np.min),
            "Bwd IAT Total":s(iat,np.sum),"Bwd IAT Mean":s(iat,np.mean),
            "Bwd IAT Std":s(iat,np.std),"Bwd IAT Max":s(iat,np.max),"Bwd IAT Min":s(iat,np.min),
            "Fwd PSH Flags":psh,"Bwd PSH Flags":0,"Fwd URG Flags":urg,"Bwd URG Flags":0,
            "Fwd Header Length":nfw*20,"Bwd Header Length":nbw*20,
            "Fwd Packets/s":nfw/ds,"Bwd Packets/s":nbw/ds,
            "Min Packet Length":s(sz_a,np.min),"Max Packet Length":s(sz_a,np.max),
            "Packet Length Mean":s(sz_a,np.mean),"Packet Length Std":s(sz_a,np.std),
            "Packet Length Variance":s(sz_a,np.var),
            "FIN Flag Count":fin,"SYN Flag Count":syn,"RST Flag Count":rst,
            "PSH Flag Count":psh,"ACK Flag Count":ack,"URG Flag Count":urg,
            "CWE Flag Count":cwe,"ECE Flag Count":ece,"Down/Up Ratio":nbw/nfw,
            "Average Packet Size":s(sz_a,np.mean),
            "Avg Fwd Segment Size":s(fw,np.mean),"Avg Bwd Segment Size":s(bw,np.mean),
            "Subflow Fwd Packets":nfw,"Subflow Fwd Bytes":s(fw,np.sum),
            "Subflow Bwd Packets":nbw,"Subflow Bwd Bytes":s(bw,np.sum),
            "Init_Win_bytes_forward":65535,"Init_Win_bytes_backward":65535,
            "act_data_pkt_fwd":nfw,"min_seg_size_forward":s(fw,np.min),
            "Active Mean":s(iat,np.mean),"Active Std":s(iat,np.std),
            "Active Max":s(iat,np.max),"Active Min":s(iat,np.min),
            "Idle Mean":s(iat,np.mean),"Idle Std":s(iat,np.std),
            "Idle Max":s(iat,np.max),"Idle Min":s(iat,np.min),
            "_src_ip":src,"_dst_ip":dst,"_proto":proto,"_true_label":"?",
        })
    return pd.DataFrame(records), raw, None


# ─── CSV Parsing ───────────────────────────────────────────────────────────────
def parse_csv(f):
    try: df = pd.read_csv(f)
    except Exception as e: return None, None, f"Cannot read CSV: {e}"
    df.columns = df.columns.str.strip()
    lc = next((l.strip() for l in LABEL_COLUMNS if l.strip() in df.columns), None)
    tl = None
    if lc: tl = df[lc].astype(str).str.strip(); df = df.drop(columns=[lc])

    raw = {"total_packets":len(df),"src_ips":Counter(),"dst_ips":Counter(),
           "protocols":Counter(),"total_bytes":0}
    for c in ["Src IP","Source IP"]:
        if c in df.columns: raw["src_ips"]=Counter(df[c].dropna().astype(str).tolist()); break
    for c in ["Dst IP","Destination IP"]:
        if c in df.columns: raw["dst_ips"]=Counter(df[c].dropna().astype(str).tolist()); break
    for c in ["Protocol"]:
        if c in df.columns: raw["protocols"]=Counter(df[c].dropna().astype(str).tolist()); break
    for c in ["Total Length of Fwd Packets","Total Fwd Packets"]:
        if c in df.columns:
            raw["total_bytes"]=int(df[c].replace([np.inf,-np.inf],0).fillna(0).sum()); break

    sc  = next((c for c in ["Src IP","Source IP"] if c in df.columns), None)
    dc  = next((c for c in ["Dst IP","Destination IP"] if c in df.columns), None)
    pc  = next((c for c in ["Protocol"] if c in df.columns), None)
    df["_src_ip"]     = df[sc].astype(str)  if sc  else "N/A"
    df["_dst_ip"]     = df[dc].astype(str)  if dc  else "N/A"
    df["_proto"]      = df[pc].astype(str)  if pc  else "N/A"
    df["_true_label"] = tl.values if tl is not None else ["?"]*len(df)
    return df, raw, None


# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df, scaler):
    meta = [c for c in df.columns if c.startswith("_")]
    feat = df.drop(columns=meta, errors="ignore")
    for col in EXPECTED_FEATURES:
        if col not in feat.columns: feat[col] = 0.0
    feat = feat[EXPECTED_FEATURES].copy()
    feat = feat.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype(np.float32)
    if scaler is not None:
        X = feat.values
        n = getattr(scaler,"n_features_in_",X.shape[1])
        if X.shape[1]<n: X=np.hstack([X,np.zeros((X.shape[0],n-X.shape[1]),dtype=np.float32)])
        elif X.shape[1]>n: X=X[:,:n]
        return scaler.transform(X)
    return feat.values


# ─── Prediction ────────────────────────────────────────────────────────────────
def run_prediction(X, rf, tabnet, cfg, mode):
    if mode == "Random Forest":
        if rf is None: raise ValueError("Random Forest model not loaded.")
        p = rf.predict_proba(X); return np.argmax(p,1), p
    if mode == "TabNet":
        if tabnet is None: raise ValueError("TabNet model not loaded.")
        p = tabnet.predict_proba(X); return np.argmax(p,1), p
    if rf is None or tabnet is None or cfg is None:
        raise ValueError("Hybrid mode requires RF, TabNet, and hybrid_config.json.")
    rw, tw = cfg.get("rf_weight",0.5), cfg.get("tabnet_weight",0.5)
    rp, tp = rf.predict_proba(X), tabnet.predict_proba(X)
    nc = max(rp.shape[1],tp.shape[1])
    def pad(p): return np.hstack([p,np.zeros((p.shape[0],nc-p.shape[1]))]) if p.shape[1]<nc else p
    hp = (rw*pad(rp))+(tw*pad(tp)); return np.argmax(hp,1), hp


# ─── Charts ────────────────────────────────────────────────────────────────────
def chart_class_dist(n_benign, n_attack):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _mpl_theme()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    cats   = ["Benign", "Attack"]
    vals   = [n_benign, n_attack]
    colors = [C_GREEN, C_RED]
    bars = ax.bar(cats, vals, color=colors, width=0.42, edgecolor=C_BG, linewidth=1.5,
                  zorder=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_PANEL, linewidth=0.8, linestyle="--", zorder=0)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9)
    ax.set_ylabel("Flow Count", fontsize=8, color=C_MUTED)
    top = max(vals) if max(vals) > 0 else 1
    for bar, val, col in zip(bars, vals, colors):
        ax.text(bar.get_x()+bar.get_width()/2, val+top*0.025,
                f"{val:,}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col,
                fontfamily="monospace")
    fig.tight_layout(pad=1.2)
    return fig


def chart_protocol_dist(protocols: Counter, flow_df):
    """Bar chart of protocol counts from _proto column."""
    import matplotlib.pyplot as plt
    _mpl_theme()

    # Use _proto if available, fallback to raw protocols
    if "_proto" in flow_df.columns:
        proto_counts = Counter(flow_df["_proto"].astype(str).tolist())
    else:
        proto_counts = protocols

    if not proto_counts:
        return None

    items = proto_counts.most_common(8)
    labels, vals = zip(*items)

    COLS = [C_ACCENT, C_GREEN, C_AMBER, C_RED, "#a78bfa", "#f472b6", "#fb923c", "#38bdf8"]
    colors = [COLS[i % len(COLS)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(labels, vals, color=colors, width=0.45,
                  edgecolor=C_BG, linewidth=1.5, zorder=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_PANEL, linewidth=0.8, linestyle="--", zorder=0)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9)
    ax.set_ylabel("Flow Count", fontsize=8, color=C_MUTED)
    top = max(vals) if max(vals) > 0 else 1
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+top*0.025,
                f"{val:,}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_TEXT,
                fontfamily="monospace")
    fig.tight_layout(pad=1.2)
    return fig


def chart_confidence(probs):
    import matplotlib.pyplot as plt
    _mpl_theme()
    mean_p = probs.mean(axis=0)[:2]
    cats   = ["Benign", "Attack"]
    colors = [C_GREEN, C_RED]
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(cats[:len(mean_p)], mean_p, color=colors[:len(mean_p)],
           width=0.42, edgecolor=C_BG, linewidth=1.5, zorder=3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C_PANEL, linewidth=0.8, linestyle="--", zorder=0)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Avg Probability", fontsize=8, color=C_MUTED)
    for i, (v, col) in enumerate(zip(mean_p, colors)):
        ax.text(i, v+0.02, f"{v:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=col,
                fontfamily="monospace")
    fig.tight_layout(pad=1.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ─── MAIN UI ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="card-title">📂 &nbsp;UPLOAD TRAFFIC DATA</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload file",
    type=["pcap"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload a .pcap file to begin.", icon="ℹ️")
    st.stop()

run_btn = st.button("🔍  Analyse Traffic")

if not run_btn:
    ext  = uploaded.name.rsplit(".", 1)[-1].upper()
    size = uploaded.size / 1024
    st.markdown(
        f"<small style='color:#64748b'>Ready: <b>{uploaded.name}</b> "
        f"&nbsp;·&nbsp; {size:.1f} KB &nbsp;·&nbsp; {ext} "
        f"&nbsp;—&nbsp; click <b>Analyse Traffic</b></small>",
        unsafe_allow_html=True,
    )
    st.stop()

# ── Pipeline ───────────────────────────────────────────────────────────────────
is_csv   = uploaded.name.lower().endswith(".csv")
spin_sl  = st.empty()
warn_sl  = st.empty()

def show_spin(msg):
    spin_sl.markdown(
        f'<div class="spin-outer"><div class="spin-ring"></div>'
        f'<div class="spin-text">{msg}</div></div>',
        unsafe_allow_html=True,
    )

# 1 · Parse
show_spin("Parsing traffic data…")
flow_df = raw_info = parse_err = None
if is_csv:
    flow_df, raw_info, parse_err = parse_csv(uploaded)
else:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
        tmp.write(uploaded.read()); tp = tmp.name
    flow_df, raw_info, parse_err = parse_pcap(tp)
    try: os.unlink(tp)
    except: pass
if parse_err:
    spin_sl.empty(); st.error(f"Parse error: {parse_err}"); st.stop()

# 2 · Load models
show_spin("Loading models…")
rf_m, tab_m, cfg, scaler, merrs = load_models()
if scaler is None:
    warn_sl.warning("Quantile transformer not found — raw features will be used.")

# 3 · Preprocess
show_spin("Transforming features…")
try: X = preprocess(flow_df, scaler)
except Exception as e: spin_sl.empty(); st.error(f"Preprocessing failed: {e}"); st.stop()

# 4 · Predict
show_spin("Running inference…")
preds = hybrid_probs = pred_err = None
try:
    preds, hybrid_probs = run_prediction(X, rf_m, tab_m, cfg, model_choice)
except Exception as e:
    pred_err = str(e)

spin_sl.empty()

# ── Derived values ─────────────────────────────────────────────────────────────
n_flows   = len(flow_df)
n_feats   = len(EXPECTED_FEATURES)
n_atk     = int(np.sum(preds == 1)) if preds is not None else 0
n_ben     = int(np.sum(preds == 0)) if preds is not None else n_flows
atk_pct   = (n_atk / n_flows * 100) if n_flows > 0 else 0.0
overall   = "ATTACK" if atk_pct >= 50.0 else "BENIGN"


# ══════════════════════════════════════════════════════════════════════════════
# ── TRAFFIC OVERVIEW ──────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sh"><span class="sh-text">Traffic Overview</span><span class="sh-rule"></span></div>
""", unsafe_allow_html=True)

# ── 4 stat cards ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-accent-bar" style="background:{C_ACCENT}"></div>
    <div class="stat-label">Total Records</div>
    <div class="stat-value" style="color:{C_ACCENT}">{n_flows:,}</div>
    <div class="stat-hint">Total rows in uploaded dataset</div>
  </div>
  <div class="stat-card">
    <div class="stat-accent-bar" style="background:{C_ACCENT}"></div>
    <div class="stat-label">Features</div>
    <div class="stat-value" style="color:{C_ACCENT}">{n_feats}</div>
    <div class="stat-hint">Model input dimensions</div>
  </div>
  <div class="stat-card">
    <div class="stat-accent-bar" style="background:{C_GREEN}"></div>
    <div class="stat-label">Benign Flows</div>
    <div class="stat-value" style="color:{C_GREEN}">{n_ben:,}</div>
    <div class="stat-hint">{100-atk_pct:.1f}% of total &nbsp;·&nbsp; classified normal</div>
  </div>
  <div class="stat-card">
    <div class="stat-accent-bar" style="background:{C_RED}"></div>
    <div class="stat-label">Attack Flows</div>
    <div class="stat-value" style="color:{C_RED}">{n_atk:,}</div>
    <div class="stat-hint">{atk_pct:.1f}% of total &nbsp;·&nbsp; classified malicious</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tooltips via st.caption (below grid) ──────────────────────────────────────
# (Stat hints are embedded in the HTML above)

# ── Two charts side by side ───────────────────────────────────────────────────
if preds is not None:
    col_l, col_r = st.columns(2, gap="medium")

    with col_l:
        st.markdown('<div class="chart-card"><div class="chart-label">Class Distribution</div>', unsafe_allow_html=True)
        st.pyplot(chart_class_dist(n_ben, n_atk), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="chart-card"><div class="chart-label">Traffic Protocols</div>', unsafe_allow_html=True)
        fig_proto = chart_protocol_dist(raw_info["protocols"], flow_df)
        if fig_proto:
            st.pyplot(fig_proto, use_container_width=True)
        else:
            st.markdown("<p style='color:#4d6480;font-size:0.82rem;padding:1rem 0'>No protocol data available.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── TOP MALICIOUS IPs (always visible, full-width table) ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sh"><span class="sh-text">Top Malicious IPs</span><span class="sh-rule"></span></div>
""", unsafe_allow_html=True)

if preds is not None and n_atk > 0:
    atk_mask = preds == 1
    mal_src  = Counter(flow_df["_src_ip"].values[atk_mask]).most_common(10)
    mal_dst  = Counter(flow_df["_dst_ip"].values[atk_mask]).most_common(10)

    # Pad to same length
    max_len = max(len(mal_src), len(mal_dst))
    while len(mal_src) < max_len: mal_src.append(("—", 0))
    while len(mal_dst) < max_len: mal_dst.append(("—", 0))

    # Build HTML rows for source IPs
    src_rows = "".join([
        '<div class="ip-row">'
        + f'<span class="ip-addr">{ip}</span>'
        + (f'<span class="ip-count red">{cnt} flows</span>' if cnt > 0 else '')
        + '</div>'
        for ip, cnt in mal_src
    ])

    # Build HTML rows for destination IPs
    dst_rows = "".join([
        '<div class="ip-row">'
        + f'<span class="ip-addr">{ip}</span>'
        + (f'<span class="ip-count amber">{cnt} flows</span>' if cnt > 0 else '')
        + '</div>'
        for ip, cnt in mal_dst
    ])

    st.markdown(f"""
    <div class="ip-section">
      <div class="ip-section-header">
        <div class="ip-col-head">
          <div class="ip-dot" style="background:{C_RED}"></div>
          Attack Source IPs
        </div>
        <div class="ip-col-head">
          <div class="ip-dot" style="background:{C_AMBER}"></div>
          Attack Target Destinations
        </div>
      </div>
      <div class="ip-body">
        <div class="ip-col">{src_rows}</div>
        <div class="ip-col">{dst_rows}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown(
        f'<div class="no-attacks-banner">'
        f'<span style="font-size:1rem">✓</span>'
        f' No malicious IPs detected in this capture.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Raw feature table (optional) ───────────────────────────────────────────────
if show_raw:
    with st.expander("Raw Feature Table — first 50 rows"):
        fc = [c for c in flow_df.columns if not c.startswith("_")]
        st.dataframe(flow_df[fc].head(50), use_container_width=True)

# ── Per-flow prediction table — expander (dropdown) ───────────────────────────
with st.expander("Per-Flow Prediction Detail", expanded=False):

    
    lmap = {0:"BENIGN", 1:"ATTACK"}
    res  = pd.DataFrame({
        "Src IP":     flow_df["_src_ip"].values,
        "Dst IP":     flow_df["_dst_ip"].values,
        "Protocol":   flow_df["_proto"].values,
        "Prediction": [lmap.get(int(p), str(p)) for p in preds],
    })
    if "_true_label" in flow_df.columns and any(v != "?" for v in flow_df["_true_label"].values):
        res["True Label"] = flow_df["_true_label"].values
    if hybrid_probs is not None:
        res["Benign Prob"] = hybrid_probs[:len(preds), 0].round(4)
        if hybrid_probs.shape[1] > 1:
            res["Attack Prob"] = hybrid_probs[:len(preds), 1].round(4)

    def _color(val):
        v = str(val).upper()
        if "ATTACK" in v: return f"background-color:#1c0508;color:{C_RED};font-weight:600"
        if v == "BENIGN": return f"background-color:#041409;color:{C_GREEN};font-weight:600"
        return ""

    scols = ["Prediction"] + (["True Label"] if "True Label" in res.columns else [])
    st.dataframe(res.style.applymap(_color, subset=scols),
                 use_container_width=True, hide_index=True)
    st.caption(f"Showing {len(res):,} flows · Coloured by predicted class")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;font-size:0.7rem;color:#2a3f5a'>"
    f"Hybrid NIDS &nbsp;·&nbsp; {model_choice} &nbsp;·&nbsp; "
    f"Scapy PCAP parser &nbsp;·&nbsp;·&nbsp; Quantile Transformer"
    f"</div>",
    unsafe_allow_html=True,
)