import streamlit as st
import pandas as pd
import time 
import io # Necessary to read the data from a string

# --- Configuration for Mock Data ---
RACE_DATA_FILE = "mock_race_results.csv" # Kept for visual reference only

# Mock mapping for displaying statistics
WORK_STATION_RENAMES = {
    'ski_erg': 'Ski_Erg_Time',
    'sled_push': 'Sled_Push_Time',
    'sled_pull': 'Sled_Pull_Time',
    'burpee_broad_jump': 'Burpee_Broad_Jump_Time',
    'row': 'Row_Time',
    'farmers_carry': 'Farmers_Carry_Time',
    'sandbag_lunge': 'Sandbag_Lunge_Time',
    'wall_balls': 'Wall_Balls_Time'
}

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
            # Assuming HH:MM:SS format for total time
            return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
    except:
        return None
    return None

# --- Function to Fetch Data (Mock Version - HARDCODED) ---
# NOTE: The @st.cache_data decorator has been REMOVED to bypass environment corruption.
# @st.cache_data(ttl=7200, show_spinner=False)
def fetch_mock_results(total_time_range=None):
    """
    Loads mock race results from a hardcoded string.
    """
    
    # The clean, correct mock data is now hardcoded as a multi-line string.
    MOCK_DATA_STRING = """
Rank,Name,Total_Time,Run_Time,Roxzone_Time,Ski_Erg_Time,Sled_Push_Time,Sled_Pull_Time,Burpee_Broad_Jump_Time,Row_Time,Farmers_Carry_Time,Sandbag_Lunge_Time,Wall_Balls_Time
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
    
    time.sleep(1) # Simulate network delay
    
    # Read the data directly from the string using io.StringIO
    df = pd.read_csv(io.StringIO(MOCK_DATA_STRING.strip()))
    
    # Clean column headers
    df.columns = df.columns.str.strip() 

    # Convert all time columns to total minutes for numerical operations
    time_cols = list(WORK_STATION_RENAMES.values()) + ['Total_Time', 'Run_Time', 'Roxzone_Time']
    
    # Ensure all columns exist before attempting to access or create the '_Min' column
    for col in time_cols:
        if col not in df.columns:
            # This should never happen with hardcoded data, but for robustness:
            raise KeyError(f"Critical error: Required column '{col}' not found in hardcoded data.")
        
        # This is where 'Total_Time_Min' and others are created, preventing the KeyError
        df[col.replace('_Time', '_Min')] = df[col].apply(time_to_minutes)
        
    # Apply total time filtering
    if total_time_range and len(total_time_range) == 2:
        lower, upper = total_time_range
        df = df[
            (df['Total_Time_Min'] >= lower) & 
            (df['Total_Time_Min'] <= upper)
        ]
        
    return df

# --- Streamlit UI ---
st.title("🏋️ Hybrid Race Data Analysis (Mock Data)")
st.caption("Running in a clean cloud environment to bypass local installation issues.")

# --- Sidebar for Filtering ---
st.sidebar.header("Data Selection & Filters")

# We keep the UI elements for demonstration
st.sidebar.selectbox("Select Season", options=['2024-2025 (Mock)'], index=0, key='selected_season')
st.sidebar.selectbox("Select Location", options=['Mock Event Name'], index=0, key='selected_location')
st.sidebar.selectbox("Select Gender", options=['All', 'Men', 'Women'], index=0, key='selected_gender')
st.sidebar.selectbox("Select Division", options=['All', 'Pro', 'Open'], index=0, key='selected_division')

# Total Time Filter
time_range_placeholder = st.sidebar.slider(
    "Total Time Range (Minutes)",
    min_value=50, max_value=120, value=(50, 120), step=5
)

# Convert the selected range to a tuple for the function
time_range = (time_range_placeholder[0], time_range_placeholder[1])

# --- Main Content ---
st.header("Filtered Race Results")

if st.button("Fetch / Refresh Race Data"):
    with st.spinner(f"Fetching mock data with filters..."):
        
        results_df = fetch_mock_results(total_time_range=time_range)
        
        st.session_state['current_results'] = results_df
        
        if not results_df.empty:
            st.success(f"Successfully loaded **{len(results_df)}** entries (mock data).")
            
            # Display first 10 rows using the original time-string format columns
            st.dataframe(results_df[[
                'Rank', 'Name', 'Total_Time', 'Run_Time', 'Roxzone_Time', 
                'Ski_Erg_Time', 'Sled_Push_Time', 'Sled_Pull_Time', 'Burpee_Broad_Jump_Time', 
                'Row_Time', 'Farmers_Carry_Time', 'Sandbag_Lunge_Time', 'Wall_Balls_Time'
            ]].head(10)) 
            
            st.subheader("Station Time Statistics (Minutes)")
            
            # Select relevant time columns for analysis (using the new _Min columns)
            time_cols_min = [col.replace('_Time', '_Min') for col in WORK_STATION_RENAMES.values()] 
            
            # Ensure we only select columns that actually exist (for safety)
            cols_for_stats = results_df.columns.intersection(time_cols_min)
            
            time_stats = results_df[cols_for_stats].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            
            st.dataframe(time_stats)
            
        else:
            st.warning("No mock results found for the selected time range.")
