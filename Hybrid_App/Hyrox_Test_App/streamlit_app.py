import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import date, datetime
import matplotlib.pyplot as plt
import matplotlib
import io
from fpdf import FPDF
from streamlit_gsheets import GSheetsConnection

# Force Matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# --- SETTINGS ---
st.set_page_config(page_title="GRITYARD x HYROX AI", layout="wide")
WORKSHEET_NAME = "Sheet1"

# --- DATASET ---
HYROX_STATS = {
    "Sub 65min": {"male": {"run_1": (3.72, 0.81), "work_1": (4.12, 0.45), "run_2": (4.01, 0.4), "work_2": (2.1, 0.7), "run_3": (4.25, 0.45), "work_3": (3.5, 0.9), "run_4": (4.2, 0.45), "work_4": (2.8, 0.6), "run_5": (4.3, 0.45), "work_5": (3.8, 0.4), "run_6": (4.2, 0.45), "work_6": (1.6, 0.3), "run_7": (4.2, 0.45), "work_7": (3.2, 0.6), "run_8": (4.5, 0.5), "work_8": (3.9, 0.8)}, "female": {"run_1": (4.1, 0.8), "work_1": (4.3, 0.5), "run_2": (4.4, 0.5), "work_2": (2.4, 0.8), "run_3": (4.7, 0.5), "work_3": (3.9, 1.0), "run_4": (4.6, 0.5), "work_4": (3.2, 0.7), "run_5": (4.7, 0.5), "work_5": (4.1, 0.5), "run_6": (4.6, 0.5), "work_6": (1.8, 0.4), "run_7": (4.6, 0.5), "work_7": (3.6, 0.7), "run_8": (4.9, 0.6), "work_8": (4.2, 0.9)}},
    "65-75min": {"male": {"run_1": (4.08, 0.69), "work_1": (4.45, 0.5), "run_2": (4.42, 0.5), "work_2": (2.7, 0.9), "run_3": (4.75, 0.6), "work_3": (4.2, 1.1), "run_4": (4.72, 0.6), "work_4": (3.6, 0.8), "run_5": (4.82, 0.6), "work_5": (4.1, 0.5), "run_6": (4.75, 0.6), "work_6": (1.9, 0.4), "run_7": (4.7, 0.6), "work_7": (3.8, 0.7), "run_8": (5.1, 0.7), "work_8": (4.5, 1.0)}, "female": {"run_1": (4.5, 0.7), "work_1": (4.7, 0.6), "run_2": (4.9, 0.6), "work_2": (3.1, 1.0), "run_3": (5.2, 0.7), "work_3": (4.8, 1.2), "run_4": (5.1, 0.7), "work_4": (4.2, 0.9), "run_5": (5.2, 0.7), "work_5": (4.4, 0.6), "run_6": (5.1, 0.7), "work_6": (2.1, 0.5), "run_7": (5.1, 0.7), "work_7": (4.4, 0.8), "run_8": (5.6, 0.8), "work_8": (5.1, 1.1)}},
    "75-85min": {"male": {"run_1": (4.4, 0.8), "work_1": (4.8, 0.6), "run_2": (4.85, 0.6), "work_2": (3.2, 1.1), "run_3": (5.3, 0.7), "work_3": (4.8, 1.3), "run_4": (5.25, 0.7), "work_4": (4.2, 1.0), "run_5": (5.4, 0.7), "work_5": (4.4, 0.6), "run_6": (5.25, 0.7), "work_6": (2.1, 0.5), "run_7": (5.2, 0.7), "work_7": (4.4, 0.9), "run_8": (5.7, 0.8), "work_8": (5.1, 1.3)}, "female": {"run_1": (4.8, 0.8), "work_1": (5.1, 0.7), "run_2": (5.4, 0.7), "work_2": (3.8, 1.2), "run_3": (5.9, 0.8), "work_3": (5.6, 1.5), "run_4": (5.8, 0.8), "work_4": (5.0, 1.2), "run_5": (5.9, 0.8), "work_5": (4.7, 0.7), "run_6": (5.8, 0.8), "work_6": (2.4, 0.6), "run_7": (5.7, 0.8), "work_7": (5.2, 1.0), "run_8": (6.3, 1.0), "work_8": (5.8, 1.5)}}
}

NEON, CYAN, RED, DARK_BG, CARD_BG, GRID_COLOR = "#DFFF00", "#00F0FF", "#FF4B4B", "#0E1117", "#1A1C23", "#2D2D2D"
COLOR_PEAK, COLOR_STRONG, COLOR_DEVELOPING, COLOR_FOCUS = "#1B5E20", "#4CAF50", "#FF9800", "#D32F2F"

STATION_METADATA = [
    ("Run 1", "run_1"), ("Ski Erg", "work_1"), ("Run 2", "run_2"), 
    ("Sled Push", "work_2"), ("Run 3", "run_3"), ("Sled Pull", "work_3"), 
    ("Run 4", "run_4"), ("Burpees", "work_4"), ("Run 5", "run_5"), 
    ("Rowing", "work_5"), ("Run 6", "run_6"), ("Farmers Carry", "work_6"), 
    ("Run 7", "run_7"), ("Sandbag Lunges", "work_7"), ("Run 8", "run_8"), 
    ("Wall Balls", "work_8")
]

st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; }}
    div[data-testid="stMetric"] {{ background-color: {CARD_BG}; border-left: 6px solid {NEON}; padding: 20px; border-radius: 4px; }}
    h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 2px; font-weight: 900 !important; color: white !important; }}
    .stButton>button {{ width: 100%; background-color: {NEON} !important; color: black !important; font-weight: 900; text-transform: uppercase; border-radius: 0px; height: 3.5em; border: none; }}
    .performance-card {{ background: rgba(26, 28, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); padding: 16px; margin-bottom: 12px; border-radius: 8px; }}
    .strategy-table {{ width: 100%; border-collapse: collapse; color: white; background: {CARD_BG}; border-radius: 8px; overflow: hidden; }}
    .strategy-table th {{ background: {GRID_COLOR}; padding: 12px; text-align: left; font-weight: 900; color: {NEON}; }}
    .strategy-table td {{ padding: 12px; border-bottom: 1px solid {GRID_COLOR}; }}
    .status-badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; font-weight: bold; text-transform: uppercase; color: white; }}
    .badge-peak {{ background: {COLOR_PEAK}; }}
    .badge-strong {{ background: {COLOR_STRONG}; }}
    .badge-developing {{ background: {COLOR_DEVELOPING}; }}
    .badge-focus {{ background: {COLOR_FOCUS}; }}
    .lead-box {{ background: {CARD_BG}; border: 2px solid {NEON}; padding: 30px; border-radius: 10px; text-align: center; margin: 20px 0; }}
    </style>
    """, unsafe_allow_html=True)

# --- UTILS ---
def t_to_d(t_str):
    try:
        if ":" not in str(t_str): return float(t_str)
        parts = str(t_str).strip().split(":")
        if len(parts) == 3: return int(parts[0])*60 + int(parts[1]) + (int(parts[2])/60)
        elif len(parts) == 2: return int(parts[0]) + (int(parts[1]) / 60)
        return float(t_str)
    except: return 5.0

def d_to_t(dec):
    if dec <= 0: return "00:00"
    m = int(dec); s = int(round((dec - m) * 60))
    if s >= 60: m += 1; s = 0
    return f"{m:02d}:{s:02d}"

def calculate_age_group(dob):
    today = date.today()
    age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return f"{age}"

def get_performance_tier_info(gap_sec):
    if gap_sec <= -20: return "PEAK", "badge-peak", COLOR_PEAK
    elif gap_sec <= 0: return "STRONG", "badge-strong", COLOR_STRONG
    elif gap_sec <= 20: return "DEVELOPING", "badge-developing", COLOR_DEVELOPING
    else: return "FOCUS", "badge-focus", COLOR_FOCUS

def get_local_analysis(inputs, gender, bucket):
    gender_key = gender.lower()
    stats_group = HYROX_STATS.get(bucket, HYROX_STATS["75-85min"])[gender_key]
    results = {"targets": {}, "stds": {}}
    for label, key in STATION_METADATA:
        mean_val, std_val = stats_group.get(key, (5.0, 0.5))
        results["targets"][label] = mean_val
        results["stds"][label] = std_val
    return results

def draw_radar_chart(inputs, targets):
    labels = ["Ski", "Sled P", "Sled L", "Burpees", "Row", "Farmers", "Lunges", "Wall B"]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist() + [0]
    key_map = {"work_1": "Ski Erg", "work_2": "Sled Push", "work_3": "Sled Pull", "work_4": "Burpees", "work_5": "Rowing", "work_6": "Farmers Carry", "work_7": "Sandbag Lunges", "work_8": "Wall Balls"}
    def get_norm(val, t_val): return np.clip(t_val / val if val > 0 else 0.5, 0.2, 1.2)
    w_scores = [get_norm(inputs.get(f"work_{i}", 5.0), targets.get(key_map[f"work_{i}"], 5.0)) for i in range(1, 9)] + [get_norm(inputs.get("work_1", 5.0), targets.get("Ski Erg", 5.0))]
    r_scores = [get_norm(inputs.get(f"run_{i}", 5.0), targets.get(f"Run {i}", 5.0)) for i in range(1, 9)] + [get_norm(inputs.get("run_1", 5.0), targets.get("Run 1", 5.0))]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG); ax.spines['polar'].set_color(GRID_COLOR)
    ax.plot(angles, r_scores, color=RED, linewidth=4, label="RUN ENGINE", marker='o')
    ax.plot(angles, w_scores, color=CYAN, linewidth=4, label="STATION POWER", marker='s')
    ax.plot(angles, [1.0]*len(angles), color=NEON, linestyle='--', linewidth=2, label="BENCHMARK")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, color='white', size=12, fontweight='bold')
    ax.set_yticklabels([]); ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=CARD_BG, labelcolor='white')
    st.pyplot(fig); plt.close(fig)

def draw_distribution(station_name, user_val, target_val, sigma, chart_key):
    x_bins = np.linspace(target_val - 3.5*sigma, target_val + 3.5*sigma, 45)
    y_bins = norm.pdf(x_bins, target_val, sigma)
    user_bin_idx = np.abs(x_bins - user_val).argmin()
    percentile = (1 - norm.cdf(user_val, target_val, sigma)) * 100
    base_color = RED if "Run" in station_name else CYAN
    colors = [f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}" for _ in range(45)]
    colors[user_bin_idx] = NEON
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_bins, y=y_bins, marker=dict(color=colors, line=dict(width=0)), width=(x_bins[1] - x_bins[0]) * 0.9, customdata=[d_to_t(val) for val in x_bins], hovertemplate="Split: %{customdata}"))
    fig.update_layout(title=f"{station_name} | TOP {max(0.1, 100-percentile):.1f}%", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=300, font=dict(color="white"))
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def render_heatmap(df_subset, title):
    st.markdown(f"#### {title}")
    for _, row in df_subset.iterrows():
        gap = row['Gap_Sec']
        tier_label, badge_class, border_color = get_performance_tier_info(gap)
        st.markdown(f"""
        <div class="performance-card" style="border-left: 4px solid {border_color};">
            <div style="display:flex; justify-content:space-between;">
                <div><b>{row['Station'].upper()}</b><br><small>Target: {d_to_t(row['Target'])}</small></div>
                <div style="text-align:right;"><b style="color:{NEON};">{d_to_t(row['Actual'])}</b><br><span class="status-badge {badge_class}">{tier_label}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_lead_form(unique_key):
    st.markdown(f'<div class="lead-box"><h2>🔓 UNLOCK FULL AUDIT</h2><p>Submit to see <b>Deep Comparison</b> and <b>PDF Strategy</b>.</p></div>', unsafe_allow_html=True)
    with st.form(f"magnet_{unique_key}"):
        name = st.text_input("Name")
        whatsapp = st.text_input("WhatsApp")
        if st.form_submit_button("GET REPORT"):
            if name and whatsapp:
                st.session_state.lead_submitted = True
                st.session_state.lead_name = name
                st.rerun()

def render_ui_block(mode):
    res, inputs = st.session_state[f'{mode}_results'], st.session_state[f'{mode}_inputs']
    if res and inputs:
        st.divider()
        gaps_df = pd.DataFrame([{"Station": l, "Actual": inputs.get(k, 5.0), "Target": res['targets'].get(l, 5.0), "Gap_Sec": (inputs.get(k, 5.0)-res['targets'].get(l, 5.0))*60} for l, k in STATION_METADATA])
        
        finish_val = st.session_state.get(f'{mode}_actual_finish')
        if isinstance(finish_val, tuple):
             st.metric("PREDICTED FINISH RANGE", f"{d_to_t(finish_val[0])} - {d_to_t(finish_val[1])}")
        else:
             st.metric("FINISH TIME", d_to_t(finish_val))

        rc1, rc2, rc3 = st.columns([1, 4, 1])
        with rc2: draw_radar_chart(inputs, res.get('targets', {}))
        
        c1, c2 = st.columns(2)
        with c1: render_heatmap(gaps_df[gaps_df['Station'].str.contains("Run")], "RUN ENGINE")
        with c2: render_heatmap(gaps_df[~gaps_df['Station'].str.contains("Run")], "STATION POWER")

        if not st.session_state.get('lead_submitted', False):
            render_lead_form(mode)
        else:
            st.success(f"Verified Athlete: {st.session_state.lead_name}")
            for _, row in gaps_df.iterrows(): 
                draw_distribution(row['Station'], row['Actual'], row['Target'], res['stds'].get(row['Station'], 0.5), f"d_{mode}_{row['Station']}")

# --- APP ---
if "profile_saved" not in st.session_state: st.session_state.profile_saved = False
if "lead_submitted" not in st.session_state: st.session_state.lead_submitted = False

st.sidebar.markdown("## 👤 ATHLETE PROFILE")
if not st.session_state.profile_saved:
    u_gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
    u_dob = st.sidebar.date_input("DOB", date(1995, 1, 1))
    if st.sidebar.button("SAVE PROFILE"):
        st.session_state.u_gender = u_gender
        st.session_state.profile_saved = True; st.rerun()

target_window = st.sidebar.selectbox("Benchmark Universe", list(HYROX_STATS.keys()), index=1)

st.title("🏃‍♂️ GRITYARD x HYROX AI")
for m in ['analysis', 'prediction', 'goal']:
    if f'{m}_results' not in st.session_state: st.session_state[f'{m}_results'] = None
    if f'{m}_inputs' not in st.session_state: st.session_state[f'{m}_inputs'] = None

if st.session_state.profile_saved:
    tab1, tab2, tab3 = st.tabs(["📊 ANALYSER", "🔮 PREDICTOR", "🎯 GOALS"])

    with tab1:
        st.subheader("POST-RACE DATA")
        race_t = st.text_input("CHIP TIME (HH:MM:SS)", "01:15:00")
        c1, c2, c3 = st.columns(3)
        manual = {k: [c1, c2, c3][i % 3].text_input(l, "05:00", key=f"a_{k}") for i, (l, k) in enumerate(STATION_METADATA)}
        if st.button("ANALYSE MY RACE"):
            clean = {k: t_to_d(v) for k, v in manual.items()}
            st.session_state.analysis_results = get_local_analysis(clean, st.session_state.u_gender, target_window)
            st.session_state.analysis_inputs, st.session_state.analysis_actual_finish = clean, t_to_d(race_t)
        render_ui_block('analysis')

    with tab2:
        st.subheader("BENCHMARK PREDICTOR")
        pc1, pc2 = st.columns(2)
        b_run = pc1.text_input("2.4KM RUN (MM:SS)", "09:30")
        b_trap = pc2.number_input("7RM TRAPBAR (KG)", 120)
        b_wb = pc2.number_input("WALL BALLS (Total in 4min)", 80)
        rox_in = st.text_input("EST. ROX TIME (MM:SS)", "05:00")
        
        if st.button("PREDICT PERFORMANCE"):
            fresh_1km = (t_to_d(b_run) / 2.4)
            sim = {f"run_{i+1}": fresh_1km * (1.15 + (i * 0.02)) for i in range(8)}
            strength_f = 1.0 - (min(b_trap, 200) / 400)
            for i in range(1, 8): sim[f"work_{i}"] = 4.0 * strength_f + (i * 0.1)
            
            # 4-MINUTE WALL BALL CALC (100 Reps predicted)
            wb_pace = 240.0 / max(b_wb, 1)
            sim["work_8"] = (wb_pace * 100 / 60.0) * 1.15
            
            st.session_state.prediction_results = get_local_analysis(sim, st.session_state.u_gender, target_window)
            st.session_state.prediction_inputs = sim
            mean_finish = sum(sim.values()) + t_to_d(rox_in)
            st.session_state.prediction_actual_finish = (mean_finish * 0.98, mean_finish * 1.05)
        render_ui_block('prediction')

    with tab3:
        st.subheader("🎯 RACE PACING BLUEPRINT")
        t_finish = st.text_input("TARGET FINISH TIME (HH:MM:SS)", "01:10:00")
        if st.button("CALCULATE GOAL"):
            t_m = t_to_d(t_finish)
            ref = HYROX_STATS[target_window][st.session_state.u_gender.lower()]
            goal_sim = {k: (ref[k][0] / sum(v[0] for v in ref.values())) * (t_m * 0.94) for _, k in STATION_METADATA}
            st.session_state.goal_inputs = goal_sim
            st.session_state.goal_results = get_local_analysis(goal_sim, st.session_state.u_gender, target_window)
            st.session_state.goal_actual_finish = t_m
        render_ui_block('goal')