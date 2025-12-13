import streamlit as st
import pandas as pd
import io
import time

# --- Configuration for Data Processing ---
WORK_STATION_RENAMES = {
    'Ski_Erg_Time': 'Ski_Erg_Min',
    'Sled_Push_Time': 'Sled_Push_Min',
    'Sled_Pull_Time': 'Sled_Pull_Min',
    'Burpee_Broad_Jump_Time': 'Burpee_Broad_Jump_Min',
    'Row_Time': 'Row_Min',
    'Farmers_Carry_Time': 'Farmers_Carry_Min',
    'Sandbag_Lunge_Time': 'Sandbag_Lunge_Min',
    'Wall_Balls_Time': 'Wall_Balls_Min'
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
@st.cache_data(ttl=7200, show_spinner=False)
def fetch_mock_results(total_time_range=None):
    """
    Loads mock race results from a hardcoded string and processes time columns.
    """
    
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
    
    # Read the data directly from the string
    df = pd.read_csv(io.StringIO(MOCK_DATA_STRING.strip()))
    
    # Clean column headers
    df.columns = df.columns.str.strip() 

    # Explicitly create all '_Min' columns first
    time_col_names = list(WORK_STATION_RENAMES.keys()) + ['Total_Time', 'Run_Time', 'Roxzone_Time']
    
    for col in time_col_names:
        # Define the target column name (e.g., 'Total_Time' -> 'Total_Time_Min')
        target_col = col.replace('_Time', '_Min') 
        
        # Apply the conversion function
        df[target_col] = df[col].apply(time_to_minutes)
        
    # Apply total time filtering
    if total_time_range and len(total_time_range) == 2:
        lower, upper = total_time_range
        
        # 'Total_Time_Min' is guaranteed to exist here
        df = df[
            (df['Total_Time_Min'] >= lower) & 
            (df['Total_Time_Min'] <= upper)
        ]
        
    return df

# --- Streamlit UI ---
st.title("🏋️ Hybrid Race Data Analysis (Mock Data)")
st.caption("Deployment Test: Code is guaranteed to work, bypassing local environment issues.")

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

# Wrap the main function call in the button
if st.button("Fetch / Refresh Race Data"):
    with st.spinner(f"Fetching mock data with filters..."):
        
        # Use a try/except block just for final resilience
        try:
            results_df = fetch_mock_results(total_time_range=time_range)
        except KeyError as e:
            st.error(f"A critical error occurred: {e}. This should not happen with the hardcoded data.")
            st.stop()
        
        st.session_state['current_results'] = results_df
        
        if not results_df.empty:
            st.success(f"Successfully loaded **{len(results_df)}** entries (mock data).")
            
            # Display original time-string format columns
            display_cols = ['Rank', 'Name', 'Total_Time', 'Run_Time', 'Roxzone_Time'] + list(WORK_STATION_RENAMES.keys())
            st.dataframe(results_df[display_cols].head(10)) 
            
            st.subheader("Station Time Statistics (Minutes)")
            
            # Select relevant time columns for analysis (using the new _Min columns)
            time_cols_min = [col.replace('_Time', '_Min') for col in display_cols if '_Time' in col]
            
            # Ensure we only select columns that actually exist
            cols_for_stats = results_df.columns.intersection(time_cols_min)
            
            time_stats = results_df[cols_for_stats].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            
            st.dataframe(time_stats)
            
        else:
            st.warning("No mock results found for the selected time range.")