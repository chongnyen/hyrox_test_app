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

# Force Matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# --- SETTINGS ---
st.set_page_config(page_title="GRITYARD x HYROX AI", layout="wide")

# --- DATASET: TIME-BASED COMPARISON STATS ---
HYROX_STATS = {
    "Sub 65min": {"male": {"run_1": (3.72, 0.81), "work_1": (4.12, 0.45), "run_2": (4.01, 0.4), "work_2": (2.1, 0.7), "run_3": (4.25, 0.45), "work_3": (3.5, 0.9), "run_4": (4.2, 0.45), "work_4": (2.8, 0.6), "run_5": (4.3, 0.45), "work_5": (3.8, 0.4), "run_6": (4.2, 0.45), "work_6": (1.6, 0.3), "run_7": (4.2, 0.45), "work_7": (3.2, 0.6), "run_8": (4.5, 0.5), "work_8": (3.9, 0.8)}, "female": {"run_1": (4.1, 0.8), "work_1": (4.3, 0.5), "run_2": (4.4, 0.5), "work_2": (2.4, 0.8), "run_3": (4.7, 0.5), "work_3": (3.9, 1.0), "run_4": (4.6, 0.5), "work_4": (3.2, 0.7), "run_5": (4.7, 0.5), "work_5": (4.1, 0.5), "run_6": (4.6, 0.5), "work_6": (1.8, 0.4), "run_7": (4.6, 0.5), "work_7": (3.6, 0.7), "run_8": (4.9, 0.6), "work_8": (4.2, 0.9)}},
    "65-75min": {"male": {"run_1": (4.08, 0.69), "work_1": (4.45, 0.5), "run_2": (4.42, 0.5), "work_2": (2.7, 0.9), "run_3": (4.75, 0.6), "work_3": (4.2, 1.1), "run_4": (4.72, 0.6), "work_4": (3.6, 0.8), "run_5": (4.82, 0.6), "work_5": (4.1, 0.5), "run_6": (4.75, 0.6), "work_6": (1.9, 0.4), "run_7": (4.7, 0.6), "work_7": (3.8, 0.7), "run_8": (5.1, 0.7), "work_8": (4.5, 1.0)}, "female": {"run_1": (4.5, 0.7), "work_1": (4.7, 0.6), "run_2": (4.9, 0.6), "work_2": (3.1, 1.0), "run_3": (5.2, 0.7), "work_3": (4.8, 1.2), "run_4": (5.1, 0.7), "work_4": (4.2, 0.9), "run_5": (5.2, 0.7), "work_5": (4.4, 0.6), "run_6": (5.1, 0.7), "work_6": (2.1, 0.5), "run_7": (5.1, 0.7), "work_7": (4.4, 0.8), "run_8": (5.6, 0.8), "work_8": (5.1, 1.1)}},
    "75-85min": {"male": {"run_1": (4.4, 0.8), "work_1": (4.8, 0.6), "run_2": (4.85, 0.6), "work_2": (3.2, 1.1), "run_3": (5.3, 0.7), "work_3": (4.8, 1.3), "run_4": (5.25, 0.7), "work_4": (4.2, 1.0), "run_5": (5.4, 0.7), "work_5": (4.4, 0.6), "run_6": (5.25, 0.7), "work_6": (2.1, 0.5), "run_7": (5.2, 0.7), "work_7": (4.4, 0.9), "run_8": (5.7, 0.8), "work_8": (5.1, 1.3)}, "female": {"run_1": (4.8, 0.8), "work_1": (5.1, 0.7), "run_2": (5.4, 0.7), "work_2": (3.8, 1.2), "run_3": (5.9, 0.8), "work_3": (5.6, 1.5), "run_4": (5.8, 0.8), "work_4": (5.0, 1.2), "run_5": (5.9, 0.8), "work_5": (4.7, 0.7), "run_6": (5.8, 0.8), "work_6": (2.4, 0.6), "run_7": (5.7, 0.8), "work_7": (5.2, 1.0), "run_8": (6.3, 1.0), "work_8": (5.8, 1.5)}},
    "85-95min": {"male": {"run_1": (4.68, 0.85), "work_1": (5.2, 0.7), "run_2": (5.3, 0.7), "work_2": (3.7, 1.3), "run_3": (5.8, 0.8), "work_3": (5.5, 1.6), "run_4": (5.75, 0.8), "work_4": (5.0, 1.3), "run_5": (5.9, 0.8), "work_5": (4.7, 0.7), "run_6": (5.75, 0.8), "work_6": (2.4, 0.6), "run_7": (5.7, 0.8), "work_7": (5.1, 1.1), "run_8": (6.3, 0.9), "work_8": (5.8, 1.6)}, "female": {"run_1": (5.1, 0.9), "work_1": (5.5, 0.8), "run_2": (5.8, 0.8), "work_2": (4.4, 1.5), "run_3": (6.4, 1.0), "work_3": (6.4, 1.8), "run_4": (6.3, 1.0), "work_4": (5.8, 1.5), "run_5": (6.5, 1.0), "work_5": (5.1, 0.8), "run_6": (6.3, 1.0), "work_6": (2.7, 0.7), "run_7": (6.3, 1.0), "work_7": (6.0, 1.3), "run_8": (7.1, 1.2), "work_8": (6.6, 1.9)}},
    "95-110min": {"male": {"run_1": (5.0, 0.98), "work_1": (5.6, 0.9), "run_2": (5.8, 0.9), "work_2": (4.5, 1.6), "run_3": (6.4, 1.1), "work_3": (6.5, 2.0), "run_4": (6.3, 1.1), "work_4": (6.2, 1.7), "run_5": (6.6, 1.1), "work_5": (5.2, 0.9), "run_6": (6.4, 1.1), "work_6": (2.8, 0.8), "run_7": (6.3, 1.1), "work_7": (6.2, 1.5), "run_8": (7.1, 1.3), "work_8": (6.9, 2.2)}, "female": {"run_1": (5.5, 1.1), "work_1": (6.0, 1.0), "run_2": (6.5, 1.1), "work_2": (5.2, 1.9), "run_3": (7.2, 1.3), "work_3": (7.8, 2.3), "run_4": (7.1, 1.3), "work_4": (7.2, 2.0), "run_5": (7.4, 1.4), "work_5": (5.6, 1.0), "run_6": (7.1, 1.4), "work_6": (3.2, 0.9), "run_7": (7.1, 1.4), "work_7": (7.5, 1.8), "run_8": (8.3, 1.6), "work_8": (8.2, 2.6)}}
}

# --- GLOBAL THEME ---
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

# --- CSS INJECTION ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {DARK_BG}; }}
    div[data-testid="stMetric"] {{ background-color: {CARD_BG}; border-left: 6px solid {NEON}; padding: 20px; border-radius: 4px; }}
    h1, h2, h3, h4 {{ font-family: 'Inter', sans-serif; text-transform: uppercase; letter-spacing: 2px; font-weight: 900 !important; color: white !important; }}
    .stButton>button {{ width: 100%; background-color: {NEON} !important; color: black !important; font-weight: 900; text-transform: uppercase; border-radius: 0px; height: 3.5em; border: none; }}
    .performance-card {{ background: rgba(26, 28, 35, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); padding: 16px; margin-bottom: 12px; border-radius: 8px; }}
    .status-badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 4px; font-weight: bold; text-transform: uppercase; color: white; }}
    .badge-peak {{ background: {COLOR_PEAK}; }}
    .badge-strong {{ background: {COLOR_STRONG}; }}
    .badge-developing {{ background: {COLOR_DEVELOPING}; }}
    .badge-focus {{ background: {COLOR_FOCUS}; }}
    .strategy-table {{ width: 100%; border-collapse: collapse; color: white; background: {CARD_BG}; border-radius: 8px; overflow: hidden; }}
    .strategy-table th {{ background: {GRID_COLOR}; padding: 12px; text-align: left; font-weight: 900; color: {NEON}; }}
    .strategy-table td {{ padding: 12px; border-bottom: 1px solid {GRID_COLOR}; }}
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
    if age < 25: return "16-24"
    elif age < 30: return "25-29"
    elif age < 35: return "30-34"
    elif age < 40: return "35-39"
    elif age < 45: return "40-44"
    elif age < 50: return "45-49"
    return "50-54"

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
        mean_val, std_val = stats_group[key]
        results["targets"][label] = mean_val
        results["stds"][label] = std_val
    return results

def generate_report_pdf(mode, results, inputs, finish_time):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"GRITYARD x HYROX AI - {mode.upper()} REPORT", ln=True, align='C')
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"ATHLETE: {st.session_state.get('u_email', 'Guest')}", ln=True)
    if isinstance(finish_time, tuple):
        pdf.cell(0, 10, f"PREDICTED RANGE: {d_to_t(finish_time[0])} - {d_to_t(finish_time[1])}", ln=True)
    else:
        pdf.cell(0, 10, f"FINISH TIME: {d_to_t(finish_time)}", ln=True)
    pdf.ln(5)
    
    # Table Header
    pdf.set_fill_color(223, 255, 0) # Neon Yellow
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(50, 10, "STATION", 1, 0, 'C', True)
    pdf.cell(40, 10, "ACTUAL", 1, 0, 'C', True)
    pdf.cell(40, 10, "TARGET", 1, 0, 'C', True)
    pdf.cell(40, 10, "GAP (SEC)", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 10)
    for label, key in STATION_METADATA:
        act = inputs.get(key, 0)
        trg = results['targets'].get(label, 0)
        gap = int((act - trg) * 60)
        pdf.cell(50, 10, label, 1)
        pdf.cell(40, 10, d_to_t(act), 1, 0, 'C')
        pdf.cell(40, 10, d_to_t(trg), 1, 0, 'C')
        pdf.cell(40, 10, f"{gap:+d}s", 1, 1, 'C')
        
    return pdf.output(dest='S').encode('latin-1')

# --- VISUALS ---
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
    num_bins = 45
    x_bins = np.linspace(target_val - 3.5*sigma, target_val + 3.5*sigma, num_bins)
    y_bins = norm.pdf(x_bins, target_val, sigma)
    user_bin_idx = np.abs(x_bins - user_val).argmin()
    percentile = (1 - norm.cdf(user_val, target_val, sigma)) * 100
    base_color = RED if "Run" in station_name else CYAN
    colors = [f"rgba{tuple(int(base_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}" for _ in range(num_bins)]
    colors[user_bin_idx] = NEON
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_bins, y=y_bins, marker=dict(color=colors, line=dict(width=0)), width=(x_bins[1] - x_bins[0]) * 0.9, hovertemplate="<b>Split:</b> %{customdata}<extra></extra>", customdata=[d_to_t(val) for val in x_bins]))
    fig.add_vline(x=target_val, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1)
    fig.add_vline(x=user_val, line_color=NEON, line_width=2)
    fig.update_layout(title=dict(text=f"{station_name.upper()} | TOP {max(0.1, 100-percentile):.1f}%", font=dict(color="white", size=16, weight=900), x=0, y=0.95), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, height=350, xaxis=dict(title=dict(text="TIME (MM:SS)", font=dict(color='white')), showgrid=False, ticktext=[d_to_t(val) for val in x_bins[::9]], tickvals=x_bins[::9]), yaxis=dict(showgrid=False, showticklabels=False))
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

def render_heatmap(df_subset, title):
    st.markdown(f"#### {title}")
    for _, row in df_subset.iterrows():
        gap = row['Gap_Sec']
        tier_label, badge_class, border_color = get_performance_tier_info(gap)
        st.markdown(f"""
        <div class="performance-card" style="border-left: 4px solid {border_color};">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <div><div style="font-size: 0.9em; font-weight: 800; color: white;">{row['Station'].upper()}</div>
                <div style="font-size: 0.75em; color: rgba(255,255,255,0.4);">Target: {d_to_t(row['Target'])}</div></div>
                <div style="text-align: right;"><div style="font-size: 1.1em; font-weight: 900; color: {NEON};">{d_to_t(row['Actual'])}</div>
                <span class="status-badge {badge_class}">{tier_label}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_ui_block(mode):
    res, inputs = st.session_state[f'{mode}_results'], st.session_state[f'{mode}_inputs']
    if res and inputs:
        st.divider()
        gaps_df = pd.DataFrame([{"Station": l, "Actual": inputs.get(k, 5.0), "Target": res['targets'].get(l, 5.0), "Gap_Sec": (inputs.get(k, 5.0)-res['targets'].get(l, 5.0))*60} for l, k in STATION_METADATA])
        
        finish_val = st.session_state.get(f'{mode}_actual_finish')
        
        c1, c2, c3 = st.columns([2, 1, 1])
        if isinstance(finish_val, tuple):
             c1.metric("PREDICTED FINISH RANGE", f"{d_to_t(finish_val[0])} - {d_to_t(finish_val[1])}")
        else:
             c1.metric("FINISH TIME", d_to_t(finish_val))
             
        # Generate PDF Bytes
        pdf_bytes = generate_report_pdf(mode, res, inputs, finish_val)
        c3.download_button("📩 DOWNLOAD REPORT (PDF)", data=pdf_bytes, file_name=f"hyrox_{mode}.pdf", mime="application/pdf")

        st.markdown("### 🎯 PERFORMANCE BALANCE")
        rc1, rc2, rc3 = st.columns([1, 4, 1])
        with rc2: draw_radar_chart(inputs, res.get('targets', {}))
        
        st.divider()
        st.markdown("### ⚡ STATION BREAKDOWN")
        card_col1, card_col2 = st.columns(2)
        with card_col1: render_heatmap(gaps_df[gaps_df['Station'].str.contains("Run")], "RUN ENGINE")
        with card_col2: render_heatmap(gaps_df[~gaps_df['Station'].str.contains("Run")], "STATION POWER")

        st.divider()
        st.markdown("### 📊 FIELD COMPARISON")
        for _, row in gaps_df.iterrows(): 
            sigma = res['stds'].get(row['Station'], 0.5)
            draw_distribution(row['Station'], row['Actual'], row['Target'], sigma, f"dist_{mode}_{row['Station'].replace(' ', '_')}")

# --- APP LOGIC ---
if "profile_saved" not in st.session_state: st.session_state.profile_saved = False
if "workout_history" not in st.session_state: st.session_state.workout_history = []
if "is_premium" not in st.session_state: st.session_state.is_premium = False

st.sidebar.markdown("## 👤 ATHLETE PROFILE")
if not st.session_state.profile_saved:
    u_email = st.sidebar.text_input("Email", "athlete@grityard.com")
    u_gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
    u_dob = st.sidebar.date_input("DOB", date(1995, 1, 1))
    u_weight = st.sidebar.number_input("Weight (kg)", 40, 150, 75)
    if st.sidebar.button("SAVE PROFILE"):
        st.session_state.u_email, st.session_state.u_gender = u_email, u_gender
        st.session_state.u_age_grp, st.session_state.u_weight = calculate_age_group(u_dob), u_weight
        st.session_state.profile_saved = True; st.rerun()
else:
    st.sidebar.info(f"{st.session_state.u_email}\n{st.session_state.u_gender} | {st.session_state.u_age_grp}")

target_window = st.sidebar.selectbox("Benchmark Universe", list(HYROX_STATS.keys()), index=1)

# --- UPDATED PAYWALL ---
st.sidebar.divider()
st.sidebar.markdown("### 💎 PREMIUM ACCESS")
access_code = st.sidebar.text_input("Enter Member Code", type="password")
if access_code == "GRITYARD2026": # Example code
    st.session_state.is_premium = True
    st.sidebar.success("PREMIUM ACTIVE")
else:
    st.session_state.is_premium = False
    if access_code: st.sidebar.error("INVALID CODE")

st.title("🏃‍♂️ GRITYARD x HYROX AI")
for m in ['analysis', 'prediction', 'goal']:
    if f'{m}_results' not in st.session_state: st.session_state[f'{m}_results'] = None
    if f'{m}_inputs' not in st.session_state: st.session_state[f'{m}_inputs'] = None

if st.session_state.profile_saved:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 ANALYSER", "🔮 PREDICTOR", "🎯 GOALS", "📅 LOG", "🤖 COACH"])

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
        b_run = pc1.text_input("2.4KM RUN TIME (MM:SS)", "09:30")
        b_ski = pc1.number_input("4-MIN SKI MAX DISTANCE (M)", 850)
        b_row = pc1.number_input("4-MIN ROW MAX DISTANCE (M)", 1000)
        b_trap = pc2.number_input("7RM TRAPBAR DEADLIFT (KG)", 120)
        b_burp = pc2.number_input("4-MIN BURPEE BROAD JUMP (REPS)", 55)
        b_wb = pc2.number_input("4-MIN WALL BALLS (REPS)", 80)
        b_vjump = pc2.number_input("VERTICAL JUMP HEIGHT (CM)", 20, 100, 50)
        rox_in = st.text_input("EST. ROX TIME (MM:SS)", "05:00")
        
        if st.button("PREDICT PERFORMANCE"):
            # REFINED AGGRESSIVE FATIGUE DECAY (50-100 Percentile Model)
            fresh_1km = (t_to_d(b_run) / 2.4)
            interference_coeffs = [1.02, 1.07, 1.15, 1.25, 1.35, 1.38, 1.48, 1.30]
            
            sim = {}
            for i in range(8):
                sim[f"run_{i+1}"] = fresh_1km * interference_coeffs[i]
            
            v_f = b_vjump / 50.0

            sim.update({
                "work_1": (1000 / (b_ski / 4)) * 1.12,
                "work_2": (8.5 - (b_trap / 20)) * (1.25 - 0.1 * v_f),
                "work_4": (85 / (b_burp / 4)) * 1.15,
                "work_5": (1000 / (b_row / 4)) * 1.10,
                "work_8": (110 / (b_wb / 4)) * 1.20,
                "work_3": 7.5, "work_6": 2.8, "work_7": 6.5
            })
            st.session_state.prediction_results = get_local_analysis(sim, st.session_state.u_gender, target_window)
            st.session_state.prediction_inputs = sim
            
            # --- UPDATED TO RANGE ---
            mean_finish = sum(sim.values()) + t_to_d(rox_in)
            st.session_state.prediction_actual_finish = (mean_finish - 2.0, mean_finish + 2.0)
            
        render_ui_block('prediction')

    with tab3:
        st.subheader("🎯 RACE PACING BLUEPRINT")
        t_finish = st.text_input("TARGET FINISH TIME (HH:MM:SS)", "01:10:00")
        rox_buf = st.slider("ROX Buffer %", 3, 10, 6)
        if st.button("CALCULATE GOAL"):
            t_m = t_to_d(t_finish)
            ref_stats = HYROX_STATS[target_window][st.session_state.u_gender.lower()]
            bench_total = sum(v[0] for v in ref_stats.values())
            avail = t_m * (1 - rox_buf/100)
            goal_sim = {k: (ref_stats[k][0] / bench_total) * avail for _, k in STATION_METADATA}
            st.session_state.goal_inputs, st.session_state.goal_results = goal_sim, get_local_analysis(goal_sim, st.session_state.u_gender, target_window)
            st.session_state.goal_actual_finish = t_m

        if st.session_state.goal_inputs:
            g_data = st.session_state.goal_inputs
            
            # PDF Download for Goal
            goal_pdf_bytes = generate_report_pdf("Goal", st.session_state.goal_results, g_data, st.session_state.goal_actual_finish)
            st.download_button("📩 DOWNLOAD PACING PLAN (PDF)", data=goal_pdf_bytes, file_name="hyrox_pacing.pdf", mime="application/pdf")
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("#### 🏃‍♂️ RUNS")
                html = '<table class="strategy-table"><tr><th>STATION</th><th>SPLIT</th></tr>'
                for l, k in STATION_METADATA:
                    if 'run' in k: html += f'<tr><td>{l}</td><td style="color:{NEON}; font-weight:900;">{d_to_t(g_data[k])}</td></tr>'
                st.markdown(html+'</table>', unsafe_allow_html=True)
            with col_r:
                st.markdown("#### ⚙️ STATIONS")
                html = '<table class="strategy-table"><tr><th>STATION</th><th>SPLIT</th></tr>'
                for l, k in STATION_METADATA:
                    if 'work' in k: html += f'<tr><td>{l}</td><td style="color:{NEON}; font-weight:900;">{d_to_t(g_data[k])}</td></tr>'
                st.markdown(html+'</table>', unsafe_allow_html=True)

    with tab4:
        st.subheader("📅 LOG")
        if st.button("Add Session"): st.session_state.workout_history.append({"Date": date.today(), "Type": "Hyrox Sim"})
        st.dataframe(pd.DataFrame(st.session_state.workout_history))

    with tab5:
        st.subheader("🤖 COACH")
        if st.session_state.is_premium: 
            st.success("PREMIUM AI COACH ACTIVE")
            st.info("Based on your data, focus on R5-R7 pacing to minimize the interference spike seen in your prediction.")
        else: 
            st.warning("PREMIUM COACHING LOCKED")
            st.markdown("Please enter a valid Member Code in the sidebar to access AI programming.")

else:
    st.warning("Please complete your profile.")