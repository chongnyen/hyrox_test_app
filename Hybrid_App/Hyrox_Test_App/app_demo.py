import streamlit as st
import pandas as pd
import numpy as np

# Force No Plotly to avoid ModuleNotFoundError
st.set_page_config(page_title="GRITYARD x HYROX AI", layout="wide")
NEON, DARK_BG, CARD_BG = "#DFFF00", "#0E1117", "#1A1C23"

# --- UTILS ---
def t_to_d(t_str):
    try:
        if ":" not in str(t_str): return float(t_str)
        parts = str(t_str).strip().split(":")
        return int(parts[0]) + (int(parts[1]) / 60)
    except: return 5.0

def d_to_t(dec):
    m = int(dec); s = int(round((dec - m) * 60))
    if s >= 60: m += 1; s = 0
    return f"{m:02d}:{s:02d}"

# --- APP ---
st.title("🏃‍♂️ GRITYARD x HYROX AI")
u_gender = st.sidebar.selectbox("Gender", ["MALE", "FEMALE"])

tab1, tab2 = st.tabs(["📊 ANALYSER", "🔮 PREDICTOR"])

with tab2:
    st.subheader("🔮 PERFORMANCE PREDICTOR")
    st.markdown("Predict your finish time based on current markers.")
    pc1, pc2, pc3 = st.columns(3)
    b_run = pc1.text_input("2.4KM RUN (MM:SS)", "09:30")
    b_trap = pc2.number_input("7RM TRAPBAR (KG)", 120)
    b_wb = pc3.number_input("WALL BALLS (Total in 4min)", 80)
    
    if st.button("PREDICT PERFORMANCE"):
        fresh_1km = (t_to_d(b_run) / 2.4)
        sim_inputs = {f"run_{i+1}": fresh_1km * (1.15 + (i * 0.02)) for i in range(8)}
        strength_factor = 1.0 - (min(b_trap, 200) / 400) 
        for i in range(1, 8): sim_inputs[f"work_{i}"] = 4.0 * strength_factor + (i * 0.1)
        
        # 4-MINUTE WALL BALL CALCULATION
        wb_pace_per_rep = 240.0 / max(b_wb, 1)
        sim_inputs["work_8"] = (wb_pace_per_rep * 100 / 60.0) * 1.15
        
        total_time = sum(sim_inputs.values())
        st.markdown(f'''
            <div style="background:{CARD_BG}; padding:25px; border-radius:12px; text-align:center; border: 2px solid {NEON}; margin-top:20px;">
                <h3 style="margin:0; opacity:0.8; color:white;">PREDICTED FINISH RANGE</h3>
                <h1 style="color:{NEON}; font-size:3.5rem; margin:15px 0;">{d_to_t(total_time * 0.98)} - {d_to_t(total_time * 1.05)}</h1>
                <p style="color:white; opacity:0.7;">Based on 2.4km test and {b_wb} Wall Ball reps.</p>
            </div>
        ''', unsafe_allow_html=True)