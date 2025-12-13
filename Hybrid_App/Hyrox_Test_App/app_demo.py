import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import traceback 
from datetime import timedelta

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide")

# --- Configuration & Constants ---
CDN_BASE = "https://d2wl4b7sx66tfb.cloudfront.net"
MANIFEST_URL = f"{CDN_BASE}/manifest/latest.csv"

# Time column mapping (used for tables)
STATION_NAMES_MAP = {
    'work_1': 'Work 1 - 1000m SkiErg',
    'work_2': 'Work 2 - 50m Sled Push',
    'work_3': 'Work 3 - 50m Sled Pull',
    'work_4': 'Work 4 - 80m Burpee Broad Jump',
    'work_5': 'Work 5 - 1000m Row',
    'work_6': 'Work 6 - 200m Farmers Carry',
    'work_7': 'Work 7 - 100m Sandbag Lunges',
    'work_8': 'Work 8 - Wall Balls',
    'roxzone_time': 'Roxzone',
    'total_time': 'Total Time',
    'run_time': 'Total Run Time',
    'work_time': 'Total Work Time',
    'run_1': 'Run 1 Split', 'run_2': 'Run 2 Split', 'run_3': 'Run 3 Split', 'run_4': 'Run 4 Split',
    'run_5': 'Run 5 Split', 'run_6': 'Run 6 Split', 'run_7': 'Run 7 Split', 'run_8': 'Run 8 Split',
}


# --- Helper Functions (Time Conversion) ---

def time_to_minutes(time_str):
    """Converts a time value (string, float, int, or Timedelta) into total minutes."""
    if pd.isna(time_str): return None
    try:
        if isinstance(time_str, pd.Timedelta):
            return time_str.total_seconds() / 60.0
        if isinstance(time_str, (int, float)):
             return time_str 
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) == 2: return int(parts[0]) + float(parts[1]) / 60
            elif len(parts) == 3: return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 60
    except Exception:
        return None
    return None

def minutes_to_mmss_string(minutes):
    """Converts total minutes (float) back to a readable MM:SS.ss string."""
    if isinstance(minutes, pd.Series): return minutes.apply(minutes_to_mmss_string)
    if pd.isna(minutes) or minutes is None or not isinstance(minutes, (int, float)): return ""
    
    total_seconds = minutes * 60
    m = int(total_seconds // 60)
    s = total_seconds % 60
    return f"{m:02d}:{s:05.2f}"

# --- Data Fetching Functions (Unchanged) ---

@st.cache_data(ttl=3600, show_spinner="Fetching race list...")
def fetch_race_manifest():
    """Loads the manifest CSV from the CDN to get a list of all races."""
    try:
        df = pd.read_csv(MANIFEST_URL)
        df['race_id'] = 'S' + df['season'].astype(str) + ' - ' + df['location']
        return df[['race_id', 'path']].drop_duplicates().sort_values('race_id')
    except Exception as e:
        st.error(f"Error loading race manifest from CDN: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner="Fetching race data...")
def fetch_race_data(race_path, gender_filter=None, division_filter=None, total_time_range=None):
    """Loads and processes data from the Parquet file on the CDN."""
    s3_key = race_path.split("/", 4)[4] if race_path.startswith("s3://") else race_path
    data_url = f"{CDN_BASE}/{s3_key}"
    try:
        df = pd.read_parquet(data_url)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        time_cols_to_process = ['total_time', 'roxzone_time', 'run_time', 'work_time'] + \
                               [f'run_{i}' for i in range(1, 9)] + [f'work_{i}' for i in range(1, 9)]
        
        all_time_cols_converted = True
        for col in time_cols_to_process:
            if col in df.columns:
                target_col = f'{col}_min'
                converted_series = df[col].apply(time_to_minutes)
                if col == 'total_time' and converted_series.isna().all() and len(converted_series) > 0:
                    all_time_cols_converted = False
                    break
                df[target_col] = converted_series
                if df[col].dtype == object and target_col != col:
                    df = df.drop(columns=[col])

        if not all_time_cols_converted or 'total_time_min' not in df.columns:
            raise KeyError("Conversion to 'total_time_min' failed. Data may be malformed or in an unknown time format.")
        
        df = df.dropna(subset=['total_time_min'])
        if gender_filter and gender_filter.lower() != 'all' and 'gender' in df.columns:
            df = df[df['gender'].str.casefold() == gender_filter.lower()]
        if division_filter and division_filter.lower() != 'all' and 'division' in df.columns:
            df = df[df['division'].str.casefold() == division_filter.lower()]
        if total_time_range and len(total_time_range) == 2:
            lower, upper = total_time_range
            df = df[(df['total_time_min'] >= lower) & (df['total_time_min'] <= upper)]
        
        return df.reset_index(drop=True)

    except Exception as e:
        st.error(f"An unexpected error occurred during data fetching or processing.")
        return pd.DataFrame()


# --- Assessment Logic (Updated with SSAC Weighting) ---

def calculate_athlete_profile(km_time_min, body_weight_kg, trapbar_7rm_kg, four_min_work_reps, vert_jump_cm):
    """
    Classifies the athlete based on benchmark tests and returns a profile.
    
    SCORING STANDARDS ARE UNCHANGED.
    PROFILE CLASSIFICATION LOGIC IS UPDATED TO REFLECT RUNNING DOMINANCE (SSAC REPORT).
    """
    
    # 1. Aerobic Capacity Score (5km Time)
    aerobic_score = np.interp(km_time_min, [17, 28, 35], [100, 50, 0])
    aerobic_score = np.clip(aerobic_score, 0, 100)

    # 2. Strength Score (TrapBar 7RM)
    relative_strength = trapbar_7rm_kg / body_weight_kg
    strength_score = np.interp(relative_strength, [1.0, 1.5, 2.2], [0, 50, 100])
    strength_score = np.clip(strength_score, 0, 100)

    # 3. Power Endurance Score (4-Minute Max Work)
    power_endurance_score = np.interp(four_min_work_reps, [50, 100, 150], [0, 50, 100])
    power_endurance_score = np.clip(power_endurance_score, 0, 100)

    # 4. Explosive Power Score (Vertical Jump)
    explosive_score = np.interp(vert_jump_cm, [20, 45, 70], [0, 50, 100])
    explosive_score = np.clip(explosive_score, 0, 100)

    # Determine Profile (ADJUSTED LOGIC)
    score_diff = strength_score - aerobic_score

    # SSAC Integration: Need a >40 point strength advantage to overcome a running deficit.
    if score_diff > 40: # Strength is significantly higher
        profile = "Powerhouse 🏋️"
    elif score_diff < -30: # Aerobic is significantly higher
        profile = "Runner 🏃"
    else: # Scores are balanced (-30 to +40)
        profile = "Hybrid ✨"

    return {
        'profile': profile,
        'aerobic_score': aerobic_score,
        'strength_score': strength_score,
        'power_endurance_score': power_endurance_score,
        'explosive_score': explosive_score,
    }


def generate_pacing_and_feedback(profile_data):
    """Generates the report card text based on the calculated profile."""
    profile = profile_data['profile']
    aerobic = profile_data['aerobic_score']
    strength = profile_data['strength_score']
    power_endurance = profile_data['power_endurance_score']
    explosive = profile_data['explosive_score']
    
    report = f"### Athlete Profile: {profile}\n"
    report += "This profile determines your overall strategy for the race, highlighting your natural strengths and identifying critical weaknesses.\n\n"
    report += "---\n"
    
    # --- Profile-Specific Feedback ---
    if profile == "Runner 🏃":
        report += "#### 🎯 Run Pacing Chart & Strategy:\n"
        report += f"Your **Aerobic Capacity (Score: {aerobic:.0f})** is your greatest weapon. **(SSAC Validation: Running is the most dominant factor in total time.)** You can maintain a faster pace on the run splits and recover better in the Roxzone.\n\n"
        report += "**Run Pacing Advice:** Aim for a consistent, controlled pace (RPE 7/10) on runs 1-4. Do **NOT** redline the final runs. Use runs 5-8 as strategic recovery after your weakness stations.\n\n"
        report += "#### ⚠️ Predicted Weaknesses (Training Focus):\n"
        report += f"- **Maximal Strength (Score: {strength:.0f}):** Focus on the **Sled Push** and **Sled Pull**. These stations will be your biggest time-sinks. Training should prioritize high-weight, low-rep sets, and heavy compound lifts.\n"
        report += f"- **Explosive Power (Score: {explosive:.0f}):** The **Burpee Broad Jumps** will demand extra energy. Incorporate plyometrics and barbell complexes into your routine.\n"

    elif profile == "Powerhouse 🏋️":
        report += "#### 🎯 Run Pacing Chart & Strategy:\n"
        report += f"Your **Maximal Strength (Score: {strength:.0f})** is elite, allowing you to crush the sleds and other strength stations. This means you will spend less time working on the floor.\n\n"
        report += "**Run Pacing Advice:** Run splits are your biggest challenge. **(SSAC Validation: Low Aerobic Capacity is the biggest limiter for strength-focused athletes.)** Aim for a **conservative, recovery pace** (RPE 5/10) on all 8 runs. Use the first run to settle in and do **NOT** try to bank time on the run. Preserve your legs for the next station.\n\n"
        report += "#### ⚠️ Predicted Weaknesses (Training Focus):\n"
        report += f"- **Aerobic Capacity (Score: {aerobic:.0f}):** Your total run time is your major limiter. Dedicate specific sessions to Zone 2 heart rate training (long, slow distance) to increase running economy.\n"
        report += f"- **Roxzone/Transition:** Strength athletes often spend too much time recovering. Practice quick transitions to minimize Roxzone time, which is crucial for saving total minutes.\n"

    else: # Hybrid
        report += "#### 🎯 Run Pacing Chart & Strategy:\n"
        report += "You have the most versatile foundation. You are strong enough to avoid getting crushed by the sleds and fast enough to maintain a competitive run pace.\n\n"
        report += "**Run Pacing Advice:** Maintain an **even split pace** (RPE 6/10) on runs 1-4, then assess your energy. If you feel good, push runs 5-8 slightly (RPE 7/10) to make up for inevitable slowing in the later strength stations.\n\n"
        report += "#### ✅ Training Recommendations:\n"
        report += f"- **Focus on Consistency:** Given your balanced profile (Aerobic: {aerobic:.0f}, Strength: {strength:.0f}, Power Endurance: {power_endurance:.0f}), your training priority is **sport-specific volume** (i.e., running straight into a workout with minimal rest).\n"
        report += f"- **Identify Micro-Weaknesses:** Your highest-scoring category is **Power Endurance (Score: {power_endurance:.0f})**. Your lowest is **Explosive Power (Score: {explosive:.0f})**. Use the Training Plans tab to build specific volume in your weakest domain.\n"
        
    return report

# --- UI Tab Functions (Data Explorer & Training Plans are unchanged) ---

def data_explorer_tab(manifest_df):
    """The original data fetching and display functionality."""
    st.header("🔍 Race Data Explorer")
    
    # --- Sidebar for Filtering (Re-using existing code structure) ---
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎯 Filters")
        GENDERS = ['All', 'Female', 'Male', 'Mixed']
        DIVISIONS = ['All', 'Pro', 'Open', 'Doubles', 'Relay', 'Staff'] 
        st.session_state.selected_gender = st.selectbox("Gender Filter:", GENDERS, key='de_gender')
        st.session_state.selected_division = st.selectbox("Division Filter:", DIVISIONS, key='de_division')
        st.markdown("---")
        st.subheader("⏱ Total Time Range (Minutes)")
        st.session_state.time_min = st.number_input("Minimum Time (Minutes):", value=50.0, min_value=0.0, key='de_min')
        st.session_state.time_max = st.number_input("Maximum Time (Minutes):", value=100.0, min_value=0.0, key='de_max')
    
    selected_race_id = st.session_state.selected_race_id
    
    st.caption(f"Showing results filtered by: Gender: **{st.session_state.selected_gender}**, Division: **{st.session_state.selected_division}**.")
    st.divider()

    time_range = (st.session_state.time_min, st.session_state.time_max)
    
    if st.button("🚀 Fetch / Refresh Race Data", type="primary"):
        with st.spinner(f"Loading results for {selected_race_id}..."):
            selected_race_path = manifest_df[manifest_df['race_id'] == selected_race_id]['path'].iloc[0]
            results_df = fetch_race_data(
                race_path=selected_race_path, 
                gender_filter=st.session_state.selected_gender,
                division_filter=st.session_state.selected_division,
                total_time_range=time_range
            )
            
            if not results_df.empty:
                st.session_state.results_df = results_df
                st.success(f"Successfully loaded **{len(results_df)}** entries.")
            else:
                st.error("Could not load or process data. Check the filters and manifest.")
                st.session_state.results_df = pd.DataFrame() 

    # --- Display Data Tables ---
    if not st.session_state.results_df.empty:
        df = st.session_state.results_df
        
        # Summary Metrics
        col1, col2, col3 = st.columns(3)
        fastest_time = df['total_time_min'].min()
        average_time = df['total_time_min'].mean()
        
        with col1:
            st.metric(label="Total Athletes Analyzed", value=len(df))
        with col2:
            st.metric(label="🏆 Fastest Time", value=minutes_to_mmss_string(fastest_time))
        with col3:
            st.metric(label="⏱ Average Time", value=minutes_to_mmss_string(average_time))

        st.markdown("---")
            
        # Format the MAIN results table for display
        DISPLAY_TIME_MIN_COLS = [col for col in df.columns if col.endswith('_min')]
        display_df = df.copy()
            
        for col_min in DISPLAY_TIME_MIN_COLS:
            original_col = col_min.replace('_min', '')
            base_name = STATION_NAMES_MAP.get(original_col, original_col.replace('_time', '').replace('_', ' ').title())
            display_col_name = f"{base_name} (MM:SS.ss)"
            display_df[display_col_name] = minutes_to_mmss_string(display_df[col_min]) 
            display_df = display_df.drop(columns=[col_min])
            
        st.subheader("📊 Filtered Race Results")
        final_display_cols = [c for c in ['rank', 'name'] + [col for col in display_df.columns if '(MM:SS.ss)' in col] if c in display_df.columns]
        st.dataframe(display_df[final_display_cols], use_container_width=True)
            
        # Statistics Table
        st.subheader("📈 Station Time Statistics (MM:SS.ss)")
        STAT_MIN_COLS = [col for col in df.columns if col.endswith('_min') and col not in ['work_time_min']]
            
        if STAT_MIN_COLS:
            time_stats = df[df.columns.intersection(STAT_MIN_COLS)].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            stats_index_map = {f'{k}_min': v for k, v in STATION_NAMES_MAP.items() if f'{k}_min' in time_stats.index}
            time_stats = time_stats.rename(index=stats_index_map)
            
            for stat in ['mean', '50%', '75%']:
                time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
                
            st.dataframe(time_stats, use_container_width=True)


def assessment_tab():
    """Client test input and Report Card generation."""
    st.header("🧠 Client Test Assessment")
    st.markdown("Input your client's latest benchmark test scores to generate an Athlete Profile and Report Card, aligning with the core demands of the HYROX race.")

    # --- Athlete Inputs ---
    with st.container(border=True):
        st.subheader("Client Physical Benchmarks")
        
        col_run, col_strength = st.columns(2)
        
        with col_run:
            st.markdown("#### 🏃 Aerobic/Explosive Capacity")
            st.session_state.km_time_str = st.text_input("5km Run Time (MM:SS):", value="25:00", help="Time conversion assumes a constant race pace for HYROX run splits.", key='km_time_str')
            st.session_state.vert_jump_cm = st.number_input("Vertical Jump (cm):", value=45.0, min_value=0.0, help="Explosive power is crucial for Burpee Broad Jumps.", key='vert_jump_cm')

        with col_strength:
            st.markdown("#### 💪 Strength/Endurance Metrics")
            st.session_state.body_weight_kg = st.number_input("Body Weight (kg):", value=80.0, min_value=1.0, help="Used to calculate relative strength.", key='body_weight_kg')
            st.session_state.trapbar_7rm_kg = st.number_input("TrapBar Squat 7RM (kg):", value=130.0, min_value=1.0, help="Maximal strength model for Sleds.", key='trapbar_7rm_kg')
            st.session_state.four_min_work_reps = st.number_input("4-Minute Max Work (Total Reps):", value=100, min_value=0, help="Max work capacity over 4 minutes (e.g., AMRAP of a low-rep complex).", key='four_min_work_reps')

        # --- Report Button ---
        if st.button("Generate Athlete Report Card", type="primary"):
            try:
                # Convert 5km time string to minutes
                km_time_parts = st.session_state.km_time_str.split(':')
                if len(km_time_parts) == 2:
                    km_time_min = int(km_time_parts[0]) + float(km_time_parts[1]) / 60
                else:
                    st.error("5km time must be in MM:SS format.")
                    return
                
                # Calculate Profile
                profile_data = calculate_athlete_profile(
                    km_time_min,
                    st.session_state.body_weight_kg,
                    st.session_state.trapbar_7rm_kg,
                    st.session_state.four_min_work_reps,
                    st.session_state.vert_jump_cm
                )
                
                # Store for use in Training Plans tab
                st.session_state.profile_data = profile_data
                
                # Generate Report Card
                report_markdown = generate_pacing_and_feedback(profile_data)
                
                st.success("Report Card Generated!")
                st.markdown("---")
                
                # --- Display Report Card ---
                st.markdown("## 📋 Athlete Report Card")
                st.markdown(report_markdown)

            except Exception as e:
                st.error(f"Error generating report: Ensure all fields are correctly filled. Detail: {e}")

def training_plans_tab():
    """Tailored training plan recommendations based on profile."""
    st.header("🗓️ Tailored Training Plans")
    st.markdown("Your recommended training focus is based on the Athlete Profile determined in the **Test Assessment** tab.")
    st.divider()

    if 'profile_data' in st.session_state:
        profile = st.session_state.profile_data['profile']
        
        st.subheader(f"Recommended Plan for: {profile}")
        
        if profile == "Runner 🏃":
            st.markdown("""
            ### Phase 1 Focus: Max Strength & Power Endurance (8 Weeks)
            Your priority is converting aerobic fitness into strength-endurance.
            
            * **Strength Work:** 3x per week. Focus on **Sled Pulls/Pushes** (heavy, short sets) and **TrapBar Deadlifts** (heavy 5x5).
            * **Workout Structure:** Integrate **Wall Balls, SkiErg, and Row** into structured workouts that follow a strength stimulus (e.g., heavy Sled Pulls immediately followed by 100 Wall Balls).
            * **Running:** Maintain 3x per week, primarily Zone 2, with one tempo run to maintain speed.
            """)
        elif profile == "Powerhouse 🏋️":
            st.markdown("""
            ### Phase 1 Focus: Aerobic Capacity & Metabolic Conditioning (8 Weeks)
            Your priority is increasing running economy and reducing recovery time.
            
            * **Running Work:** 4x per week. Focus on **Zone 2 L.S.D.** (Long, Slow Distance) and **Interval Training** (e.g., 400m repeats) to boost V02 max and running efficiency.
            * **Workout Structure:** Decrease your time spent lifting heavy. Replace one heavy session with a **Hyrox Simulation** (Run + Station) or a high-rep, light-load metabolic circuit.
            * **Roxzone:** Drill transitions relentlessly. Time yourself moving from one station to the next to save critical seconds.
            """)
        else: # Hybrid
            st.markdown("""
            ### Phase 1 Focus: Sport Specificity & High-Volume Sets (8 Weeks)
            Your balanced profile allows you to train specifically for the demands of the race structure.
            
            * **Combined Workouts:** 3x per week. Practice **Run -> Station -> Run** sequences using different combinations of the 8 stations. Your best time gains will come from learning how to run *after* a demanding station.
            * **High-Volume Sets:** Perform long sets on your individual weaknesses (e.g., if Sandbag Lunges is your weakest score, perform 3 sets of 200m+ lunges with minimal rest).
            * **Running:** 3x per week, with a focus on **pacing runs** where you practice maintaining your target race pace (e.g., 5:00/km) over distances up to 8km.
            """)
    else:
        st.info("Please generate an Athlete Report Card in the **Test Assessment** tab first to receive a tailored plan.")


# --- Main App Execution ---

# Load available races once
manifest_df = fetch_race_manifest()
if manifest_df.empty:
    st.error("Cannot proceed. Could not load the live race manifest.")
    st.stop()
    
# Initialize session state for data storage and selected race
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame()
if 'selected_race_id' not in st.session_state:
    st.session_state.selected_race_id = manifest_df['race_id'].iloc[0] if not manifest_df.empty else None

# --- Global Race Selector in Sidebar ---
st.sidebar.title("Hyrox Race Selection")
st.session_state.selected_race_id = st.sidebar.selectbox("Select Race:", manifest_df['race_id'].tolist(), key='global_race_select')


st.title("🔥 Hyrox Race Analytics: Performance Deep Dive")

# --- Tab Structure ---
tab_explorer, tab_assessment, tab_plans = st.tabs(["📊 Data Explorer", "🧠 Test Assessment", "🗓️ Training Plans"])

with tab_explorer:
    data_explorer_tab(manifest_df)

with tab_assessment:
    assessment_tab()

with tab_plans:
    training_plans_tab()