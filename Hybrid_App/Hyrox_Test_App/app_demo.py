import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from datetime import date, datetime
import matplotlib.pyplot as plt
import matplotlib
import io
import os
import gspread
from google.oauth2.service_account import Credentials

# Force Matplotlib to use a non-interactive backend
matplotlib.use('Agg')

# --- SETTINGS & THEME ---
st.set_page_config(page_title="GRITYARD x HYROX AI", layout="wide")
WORKSHEET_NAME = "Sheet1"
NEON, CYAN, RED, DARK_BG, CARD_BG, GRID_COLOR = "#DFFF00", "#00F0FF", "#FF4B4B", "#0E1117", "#1A1C23", "#2D2D2D"

# Stats dictionary for benchmarking
HYROX_STATS = {
    "Sub 65min": {"male": {"run_1": (3.72, 0.81), "work_1": (4.12, 0.45), "run_2": (4.01, 0.4), "work_2": (2.1, 0.7), "run_3": (4.25, 0.45), "work_3": (3.5, 0.9), "run_4": (4.2, 0.45), "work_4": (2.8, 0.6), "run_5": (4.3, 0.45), "work_5": (3.8, 0.4), "run_6": (4.2, 0.45), "work_6": (1.6, 0.3), "run_7": (4.2, 0.45), "work_7": (3.2, 0.6), "run_8": (4.5, 0.5), "work_8": (3.9, 0.8)}, "female": {"run_1": (4.1, 0.8), "work_1": (4.3, 0.5), "run_2": (4.4, 0.5), "work_2": (2.4, 0.8), "run_3": (4.7, 0.5), "work_3": (3.9, 1.0), "run_4": (4.6, 0.5), "work_4": (3.2, 0.7), "run_5": (4.7, 0.5), "work_5": (4.1, 0.5), "run_6": (4.6, 0.5), "work_6": (1.8, 0.4), "run_7": (4.6, 0.5), "work_7": (3.6, 0.7), "run_8": (4.9, 0.6), "work_8": (4.2, 0.9)}},
    "65-75min": {"male": {"run_1": (4.08, 0.69), "work_1": (4.45, 0.5), "run_2": (4.42, 0.5), "work_2": (2.7, 0.9), "run_3": (4.75, 0.6), "work_3": (4.2, 1.1), "run_4": (4.72, 0.6), "work_4": (3.6, 0.8), "run_5": (4.82, 0.6), "work_5": (4.1, 0.5), "run_6": (4.75, 0.6), "work_6": (1.9, 0.4), "run_7": (4.7, 0.6), "work_7": (3.8, 0.7), "run_8": (5.1, 0.7), "work_8": (4.5, 1.0)}, "female": {"run_1": (4.5, 0.7), "work_1": (4.7, 0.6), "run_2": (4.9, 0.6), "work_2": (3.1, 1.0), "run_3": (5.2, 0.7), "work_3": (4.8, 1.2), "run_4": (5.1, 0.7), "work_4": (4.2, 0.9), "run_5": (5.2, 0.7), "work_5": (4.4, 0.6), "run_6": (5.1, 0.7), "run_7": (5.1, 0.7), "work_7": (4.4, 0.8), "run_8": (5.6, 0.8), "work_8": (5.1, 1.1)}},
    "75-85min": {"male": {"run_1": (4.4, 0.8), "work_1": (4.8, 0.6), "run_2": (4.85, 0.6), "work_2": (3.2, 1.1), "run_3": (5.3, 0.7), "work_3": (4.8, 1.3), "run_4": (5.25, 0.7), "work_4": (4.2, 1.0), "run_5": (5.4, 0.7), "work_5": (4.4, 0.6), "run_6": (5.25, 0.7), "work_6": (2.1, 0.5), "run_7": (5.2, 0.7), "work_7": (4.4, 0.9), "run_8": (5.7, 0.8), "work_8": (5.1, 1.3)}, "female": {"run_1": (4.8, 0.8), "work_1": (5.1, 0.7), "run_2": (5.4, 0.7), "work_2": (3.8, 1.2), "run_3": (5.9, 0.8), "work_3": (5.6, 1.5), "run_4": (5.8, 0.8), "work_4": (5.0, 1.2), "run_5": (5.9, 0.8), "work_5": (4.7, 0.7), "run_6": (5.8, 0.8), "run_7": (5.7, 0.8), "work_7": (5.2, 1.0), "run_8": (6.3, 1.0), "work_8": (5.8, 1.5)}},
    "85-95min": {"male": {"run_1": (4.68, 0.85), "work_1": (5.2, 0.7), "run_2": (5.3, 0.7), "work_2": (3.7, 1.3), "run_3": (5.8, 0.8), "work_3": (5.5, 1.6), "run_4": (5.75, 0.8), "work_4": (5.0, 1.3), "run_5": (5.9, 0.8), "work_5": (4.7, 0.7), "run_6": (5.75, 0.8), "work_6": (2.4, 0.6), "run_7": (5.7, 0.8), "work_7": (5.1, 1.1), "run_8": (6.3, 0.9), "work_8": (5.8, 1.6)}, "female": {"run_1": (5.1, 0.9), "work_1": (5.5, 0.8), "run_2": (5.8, 0.8), "work_2": (4.4, 1.5), "run_3": (6.4, 1.0), "work_3": (6.4, 1.8), "run_4": (6.3, 1.0), "work_4": (5.8, 1.5), "run_5": (6.5, 1.0), "work_5": (5.1, 0.8), "run_6": (6.3, 1.0), "run_7": (6.3, 1.0), "work_7": (6.0, 1.3), "run_8": (7.1, 1.2), "work_8": (6.6, 1.9)}},
    "95-110min": {"male": {"run_1": (5.0, 0.98), "work_1": (5.6, 0.9), "run_2": (5.8, 0.9), "work_2": (4.5, 1.6), "run_3": (6.4, 1.1), "work_3": (6.5, 2.0), "run_4": (6.3, 1.1), "work_4": (6.2, 1.7), "run_5": (6.6, 1.1), "work_5": (5.2, 0.9), "run_6": (6.4, 1.1), "work_6": (2.8, 0.8), "run_7": (6.3, 1.1), "work_7": (6.2, 1.5), "run_8": (7.1, 1.3), "work_8": (6.9, 2.2)}, "female": {"run_1": (5.5, 1.1), "work_1": (6.0, 1.0), "run_2": (6.5, 1.1), "work_2": (5.2, 1.9), "run_3": (7.2, 1.3), "work_3": (7.8, 2.3), "run_4": (7.1, 1.3), "work_4": (7.2, 2.0), "run_5": (7.4, 1.4), "work_5": (5.6, 1.0), "run_6": (7.1, 1.4), "run_7": (7.1, 1.4), "work_7": (7.5, 1.8), "run_8": (8.3, 1.6), "work_8": (8.2, 2.6)}}
}

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
    .stApp {{ background-color: {DARK_BG} !important; color: white !important; }}
    label, p, span, .stMarkdown {{ color: white !important; }}
    div.stButton > button, div.stFormSubmitButton > button {{
        background-color: {NEON} !important; color: black !important; border: none !important;
        border-radius: 4px !important; width: 100% !important; height: 3.5rem !important;
        font-weight: 900 !important; text-transform: uppercase !important;
    }}
    .performance-card {{
        background: {CARD_BG}; border: 1px solid {GRID_COLOR};
        padding: 20px; margin-bottom: 15px; border-radius: 8px;
    }}
    .sales-box {{ background: rgba(223, 255, 0, 0.1); border: 2px solid {NEON}; padding: 20px; border-radius: 8px; text-align: center; }}
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

def get_local_analysis(inputs, gender, bucket):
    gender_key = gender.lower()
    stats_group = HYROX_STATS.get(bucket, HYROX_STATS["75-85min"])[gender_key]
    results = {"targets": {}, "stds": {}}
    for label, key in STATION_METADATA:
        mean_val, std_val = stats_group[key]
        results["targets"][label] = mean_val
        results["stds"][label] = std_val
    return results

def analyze_archetype(inputs, targets):
    run_total = sum([inputs.get(f"run_{i}", 5.0) for i in range(1, 9)])
    work_total = sum([inputs.get(f"work_{i}", 5.0) for i in range(1, 9)])
    trg_run = sum([targets.get(f"Run {i}", 5.0) for i in range(1, 9)])
    trg_work = sum([v for k, v in targets.items() if "Run" not in k])
    run_diff = (run_total - trg_run) / trg_run
    work_diff = (work_total - trg_work) / trg_work
    if run_diff > 0.1 and work_diff < 0.05:
        return "The Powerhouse", "Your station work is elite, but your running engine is leaking time."
    elif work_diff > 0.1 and run_diff < 0.05:
        return "The Gazelle", "Fast on the carpet, but the heavy stations are draining your soul."
    else:
        return "The Hybrid in Training", "You have a balanced profile, but transition efficiency is key."

# --- UI COMPONENTS ---
def render_sales_cta(archetype_name, struggle):
    st.markdown(f"""
    <div class="sales-box">
        <h2 style="color:{NEON} !important;">🚀 {archetype_name.upper()} DETECTED</h2>
        <p>Data alone won't fix your bottleneck in <b>{struggle}</b>. You need the <b>GritYard Method.</b></p>
        <a href="https://wa.me/YOUR_PHONE?text=Hi, I'm a {archetype_name} needing help with {struggle}." target="_blank" style="text-decoration:none;">
            <div style="background-color:{NEON}; color:black; padding:18px; border-radius:4px; font-weight:900;">BOOK PERFORMANCE CALL</div>
        </a>
    </div>
    """, unsafe_allow_html=True)

def render_lead_form(unique_key):
    with st.form(f"magnet_form_{unique_key}"):
        col_n, col_w = st.columns(2)
        name = col_n.text_input("Name")
        whatsapp = col_w.text_input("WhatsApp")
        struggle = st.selectbox("Your Biggest Bottleneck?", ["Running Speed", "Sled/Heavy Power", "Wall Ball Burnout", "Transition Fatigue"])
        if st.form_submit_button("UNCOVER MY WEAKNESSES"):
            if name and whatsapp:
                st.session_state.lead_submitted = True
                st.session_state.lead_name = name
                st.session_state.main_struggle = struggle
                st.rerun()

def render_ui_block(mode):
    res, inputs = st.session_state[f'{mode}_results'], st.session_state[f'{mode}_inputs']
    if res and inputs:
        st.divider()
        arch_name, arch_desc = analyze_archetype(inputs, res['targets'])
        if not st.session_state.get('lead_submitted', False):
            render_lead_form(mode)
        else:
            render_sales_cta(arch_name, st.session_state.get('main_struggle', 'General Performance'))
            st.markdown(f"### 🧬 PERFORMANCE ARCHETYPE: {arch_name}")
            st.info(arch_desc)
            
            gaps = []
            for l, k in STATION_METADATA:
                gap = (inputs.get(k, 0) - res['targets'].get(l, 0)) * 60
                gaps.append((l, gap))
            gaps.sort(key=lambda x: x[1], reverse=True)
            
            st.markdown("#### 🚩 Biggest Bleed Stations")
            cols = st.columns(3)
            for i, (station, gap) in enumerate(gaps[:3]):
                cols[i].error(f"**{station}**\n\n+{int(gap)}s vs Target")

# --- MAIN APP ---
if "profile_saved" not in st.session_state: st.session_state.profile_saved = False
if "lead_submitted" not in st.session_state: st.session_state.lead_submitted = False

st.sidebar.markdown("## 👤 ATHLETE PROFILE")
if not st.session_state.profile_saved:
    u_gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])
    if st.sidebar.button("SAVE PROFILE"):
        st.session_state.u_gender = u_gender
        st.session_state.profile_saved = True; st.rerun()

target_window = st.sidebar.selectbox("Target Benchmark", list(HYROX_STATS.keys()), index=1)

st.title("🏃‍♂️ GRITYARD x HYROX AI")

if st.session_state.profile_saved:
    tab1, tab2 = st.tabs(["📊 ANALYSER", "🔮 PREDICTOR"])

    with tab1:
        st.subheader("RACE AUDIT")
        c1, c2, c3 = st.columns(3)
        manual = {k: [c1, c2, c3][i % 3].text_input(l, "05:00", key=f"a_{k}") for i, (l, k) in enumerate(STATION_METADATA)}
        if st.button("ANALYSE MY RACE"):
            clean = {k: t_to_d(v) for k, v in manual.items()}
            st.session_state.analysis_results = get_local_analysis(clean, st.session_state.u_gender, target_window)
            st.session_state.analysis_inputs = clean
        if st.session_state.get('analysis_results'): render_ui_block('analysis')

    with tab2:
        st.subheader("🔮 PREDICTOR")
        st.markdown("Predict your finish time based on your current physical markers.")
        pc1, pc2, pc3 = st.columns(3)
        b_run = pc1.text_input("2.4KM RUN (MM:SS)", "09:30")
        b_trap = pc2.number_input("7RM TRAPBAR (KG)", 120)
        b_wb = pc3.number_input("WALL BALLS (Total in 4min)", 80)
        
        if st.button("PREDICT PERFORMANCE"):
            # Simulation logic
            fresh_1km = (t_to_d(b_run) / 2.4)
            sim_inputs = {f"run_{i+1}": fresh_1km * (1.15 + (i * 0.02)) for i in range(8)}
            strength_factor = 1.0 - (min(b_trap, 200) / 400) 
            for i in range(1, 8):
                sim_inputs[f"work_{i}"] = 4.0 * strength_factor + (i * 0.1)
            
            # WALL BALL SIMULATION (4min test to 100 reps race)
            wb_pace_per_rep = 240.0 / max(b_wb, 1)
            race_wb_time_seconds = wb_pace_per_rep * 100
            sim_inputs["work_8"] = (race_wb_time_seconds / 60.0) * 1.15
            
            st.session_state.prediction_results = get_local_analysis(sim_inputs, st.session_state.u_gender, target_window)
            st.session_state.prediction_inputs = sim_inputs
            st.session_state.total_predicted_time = sum(sim_inputs.values())

        if st.session_state.get('prediction_results'):
            t_min = st.session_state.total_predicted_time * 0.98
            t_max = st.session_state.total_predicted_time * 1.06
            
            st.markdown(f"""
                <div class="performance-card" style="text-align:center; border: 2px solid {NEON};">
                    <h2 style="margin:0; opacity:0.8;">PREDICTED FINISH RANGE</h2>
                    <h1 style="color:{NEON}; font-size:3.5rem; margin:10px 0;">{d_to_t(t_min)} - {d_to_t(t_max)}</h1>
                    <p>Calculated engine: 2.4km test, {b_trap}kg strength, and 4-min Wall Ball capacity.</p>
                </div>
            """, unsafe_allow_html=True)
            render_ui_block('prediction')