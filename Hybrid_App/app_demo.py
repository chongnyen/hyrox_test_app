
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Hyrox AI Coach", layout="wide")
API_BASE_URL = "http://127.0.0.1:8000"

def t_to_d(t_str):
    try:
        if ":" in t_str:
            m, s = map(int, t_str.split(":"))
            return m + (s / 60)
        return float(t_str)
    except: return 5.0

def d_to_t(dec):
    m, s = int(dec), int(round((dec - int(dec)) * 60))
    if s >= 60: m += 1; s = 0
    return f"{m:02d}:{s:02d}"

st.title("🏃‍♂️ Hyrox AI Coach")
t1, t2 = st.tabs(["Analyzer", "Predictor"])

with t1:
    st.sidebar.header("Settings")
    gen = st.sidebar.selectbox("Gender", ["male", "female"])
    age = st.sidebar.selectbox("Age Group", ["16-24", "25-29", "30-34", "35-39", "40-44", "45-49"])
    stations = [("Run 1", "run_1"), ("Ski", "work_1"), ("Run 2", "run_2"), ("Sled P", "work_2"),
                ("Run 3", "run_3"), ("Sled L", "work_3"), ("Run 4", "run_4"), ("Burp", "work_4"),
                ("Run 5", "run_5"), ("Row", "work_5"), ("Run 6", "run_6"), ("Farm", "work_6"),
                ("Run 7", "run_7"), ("Lung", "work_7"), ("Run 8", "run_8"), ("Wall", "work_8")]
    inputs = {}
    cols = st.columns(4)
    for i, (l, k) in enumerate(stations): inputs[k] = cols[i%4].text_input(l, "05:00", key=k)

    if st.button("Analyze"):
        payload = {"gender": gen, "age_group": age, **{k: t_to_d(v) for k, v in inputs.items()}}
        res = requests.post(f"{API_BASE_URL}/analyze_custom", json=payload).json()
        st.subheader(f"Focus: {res['coaching']['title']}")
        radar = requests.post(f"{API_BASE_URL}/radar_custom", json=payload)
        st.image(radar.content)
        data = [{"Station": l, "You": inputs[k], "Target": d_to_t(res['targets'][l])} for l, k in stations]
        st.table(pd.DataFrame(data))
