import streamlit as st
import pandas as pd
import io

# --- Configuration & Time Conversion (Copied from working code) ---

# Helper to convert MM:SS or HH:MM:SS to total minutes
def time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    try:
        parts = str(time_str).split(':')
        if len(parts) == 2:
            # Assuming MM:SS format for stations
            return int(parts[0]) + int(parts[1]) / 60
        elif len(parts) == 3:
            # Assuming HH:MM:SS format for total time (HH:MM:SS)
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    except:
        return None
    return None

# Helper to convert minutes back to readable MM:SS.ss string for display
def minutes_to_mmss_string(minutes):
    if pd.isna(minutes) or minutes is None:
        return ""
    if isinstance(minutes, pd.Series):
        # Apply conversion to each element in a series
        return minutes.apply(minutes_to_mmss_string)
    
    total_seconds = minutes * 60
    m = int(total_seconds // 60)
    s = total_seconds % 60
    return f"{m:02d}:{s:05.2f}"

# --- Function to Fetch Data (Mock Version - GUARANTEED TO WORK) ---

def fetch_mock_results(total_time_range=None):
    """
    Loads mock race results from a hardcoded string and processes time columns.
    """
    
    # Using snake_case for consistency with common DataFrame practice
    MOCK_DATA_STRING = """
rank,name,total_time,run_time,roxzone_time,ski_erg_time,sled_push_time,sled_pull_time,burpee_broad_jump_time,row_time,farmers_carry_time,sandbag_lunge_time,wall_balls_time
1,Athlete A,58:34,31:12,02:10,02:45,03:30,03:20,03:15,02:40,02:55,03:10,03:37
2,Athlete B,61:05,33:05,02:20,02:50,03:45,03:35,03:20,02:50,03:05,03:20,03:45
3,Athlete C,63:15,34:10,02:30,02:55,04:00,03:40,03:25,03:00,03:10,03:30,03:55
4,Athlete D,66:40,35:50,02:40,03:00,04:10,03:50,03:30,03:10,03:20,03:40,04:10
5,Athlete E,70:25,37:30,02:50,03:05,04:25,04:00,03:35,03:20,03:30,03:50,04:15
6,Athlete F,75:00,40:00,03:00,03:10,04:40,04:10,03:40,03:30,03:40,04:00,04:40
7,Athlete G,80:15,42:30,03:10,03:15,04:55,04:20,03:45,03:40,03:50,04:10,05:00
8,Athlete H,85:00,45:00,03:20,03:20,05:10,04:30,03:50,03:50,04:00,04:20,05:20
9,Athlete I,90:00,47:30,03:30,03:25,05:25,04:40,03:55,04:00,04:10,04:30,05:40
10,Athlete J,95:30,50:00,03:40,03:30,05:40,04:50,04:00,04:10,04:20,04:40,06:00
"""
    
    df = pd.read_csv(io.StringIO(MOCK_DATA_STRING.strip()))
    df.columns = df.columns.str.strip() 

    # 1. Convert 'total_time' to 'total_time_min' immediately.
    df['total_time_min'] = df['total_time'].apply(time_to_minutes)
    
    # 2. Apply total time filtering
    if total_time_range and len(total_time_range) == 2:
        lower, upper = total_time_range
        df = df[
            (df['total_time_min'] >= lower) & 
            (df['total_time_min'] <= upper)
        ]

    # 3. Convert remaining time columns 
    time_cols_to_convert = ['run_time', 'roxzone_time', 'ski_erg_time', 'sled_push_time', 'sled_pull_time', 
                      'burpee_broad_jump_time', 'row_time', 'farmers_carry_time', 'sandbag_lunge_time', 'wall_balls_time']
    
    for col in time_cols_to_convert:
        target_col = col.replace('_time', '_min') 
        df[target_col] = df[col].apply(time_to_minutes)
        
    return df

# --- Streamlit UI ---
st.title("🏋️ Hyrox Data Explorer (Stable Mock Demo)")
st.caption("Reverted to stable, self-contained code after pyrox dependency failure.")

# --- Sidebar for Filtering ---
st.sidebar.header("Data Selection & Filters")

# MOCK RACES (For dynamic selection appearance)
MOCK_RACES = ["S7 - London", "S7 - Maastricht", "S6 - Madrid", "S6 - New York"]
selected_key = st.sidebar.selectbox("Select Race:", MOCK_RACES)

# 2. Gender and Division Filters
GENDERS = ['All', 'Female', 'Male']
DIVISIONS = ['All', 'Pro', 'Open', 'Doubles'] 

selected_gender = st.sidebar.selectbox("Gender Filter:", GENDERS)
selected_division = st.sidebar.selectbox("Division Filter:", DIVISIONS)

# 3. Total Time Filter
st.sidebar.subheader("Total Time Range (Minutes)")
time_min = st.sidebar.number_input("Minimum Time (Minutes):", value=50.0, min_value=0.0)
time_max = st.sidebar.number_input("Maximum Time (Minutes):", value=100.0, min_value=0.0)

time_range = None
if time_min is not None and time_max is not None:
    if time_min > time_max:
        st.sidebar.warning("Minimum time must be less than maximum time.")
    else:
        time_range = (time_min, time_max)
elif time_min is not None or time_max is not None:
    st.sidebar.info("Enter both min and max for filtering.")

# --- Main Content ---
st.header(f"Results: {selected_key}")
st.markdown(f"**Filters:** Gender: `{selected_gender}`, Division: `{selected_division}`, Time: `{time_range or 'None'}`")
st.divider()


# Fetch and Display Results (Now fully safe and stable)
if st.button("Fetch / Refresh Race Data"):
    with st.spinner(f"Loading results for {selected_key}..."):
        try:
            results_df = fetch_mock_results(total_time_range=time_range)
        except Exception as e:
            st.error(f"A critical data processing error occurred: {e}")
            st.stop()
        
        if results_df.empty:
            st.warning("No results found for the selected time range.")
        else:
            st.success(f"Successfully loaded **{len(results_df)}** entries (Mock Data).")
            
            # Define columns for display
            DISPLAY_TIME_COLS = ['total_time', 'run_time', 'roxzone_time', 
                                 'ski_erg_time', 'sled_push_time', 'sled_pull_time', 'burpee_broad_jump_time']
            DISPLAY_COLS = ['rank', 'name'] + DISPLAY_TIME_COLS
            
            # --- Format the MAIN results table for display ---
            display_df = results_df.copy()
            
            for col in DISPLAY_TIME_COLS:
                # Convert the decimal minutes (e.g., total_time_min) to readable MM:SS.ss string
                min_col = col.replace('_time', '_min')
                if min_col in display_df.columns:
                    display_df[col] = minutes_to_mmss_string(display_df[min_col]) 
                
            st.subheader("Filtered Race Results (Time in MM:SS.ss)")
            st.dataframe(display_df[display_df.columns.intersection(DISPLAY_COLS)])
            
            # --- Calculate and format the STATISTICS table ---
            st.subheader("Station Time Statistics (MM:SS.ss)")
            
            # Select relevant time columns for statistics (using the _min columns)
            STAT_MIN_COLS = [col.replace('_time', '_min') for col in DISPLAY_TIME_COLS]
            
            time_stats = results_df[
                results_df.columns.intersection(STAT_MIN_COLS)
            ].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            
            STAT_COLUMNS_TO_FORMAT = ['mean', '50%', '75%']
            
            # Format the statistics columns from minutes (decimal) to MM:SS.ss string
            for stat in STAT_COLUMNS_TO_FORMAT:
                time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
            
            st.dataframe(time_stats)