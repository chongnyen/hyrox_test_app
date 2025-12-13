import streamlit as st
import pandas as pd
import requests
import io
import traceback 

# --- Configuration ---
CDN_BASE = "https://d2wl4b7sx66tfb.cloudfront.net"
MANIFEST_URL = f"{CDN_BASE}/manifest/latest.csv"

# --- Station Name Mapping (New addition for better display) ---
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
    # Map for the individual run splits if you want them in the table
    'run_1': 'Run 1 Split',
    'run_2': 'Run 2 Split',
    'run_3': 'Run 3 Split',
    'run_4': 'Run 4 Split',
    'run_5': 'Run 5 Split',
    'run_6': 'Run 6 Split',
    'run_7': 'Run 7 Split',
    'run_8': 'Run 8 Split',
}


# --- Helper Functions ---

def time_to_minutes(time_str):
    """Converts a time value (string, float, int, or Timedelta) into total minutes."""
    if pd.isna(time_str):
        return None
    try:
        # CRITICAL FIX: Handle Pandas Timedelta objects, which Parquet often uses for time data.
        if isinstance(time_str, pd.Timedelta):
            return time_str.total_seconds() / 60.0
            
        # If the value is already a number (float/int), assume it's already in minutes.
        if isinstance(time_str, (int, float)):
             return time_str 

        # If it's a string (H:MM:SS or MM:SS)
        if isinstance(time_str, str):
            parts = time_str.split(':')
            if len(parts) == 2:
                # MM:SS format
                return int(parts[0]) + float(parts[1]) / 60
            elif len(parts) == 3:
                # HH:MM:SS format
                return int(parts[0]) * 60 + int(parts[1]) + float(parts[2]) / 60
            
    except Exception:
        return None
    return None

def minutes_to_mmss_string(minutes):
    """Converts total minutes (float) back to a readable MM:SS.ss string."""
    if isinstance(minutes, pd.Series):
        return minutes.apply(minutes_to_mmss_string)

    if pd.isna(minutes) or minutes is None:
        return ""
    
    if not isinstance(minutes, (int, float)):
        return ""

    total_seconds = minutes * 60
    m = int(total_seconds // 60)
    s = total_seconds % 60
    
    return f"{m:02d}:{s:05.2f}"

# --- Data Fetching Functions ---

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
    """
    Loads data from the Parquet file on the CDN and processes it.
    """
    
    s3_key = race_path.split("/", 4)[4] if race_path.startswith("s3://") else race_path
    data_url = f"{CDN_BASE}/{s3_key}"

    try:
        df = pd.read_parquet(data_url)
        
        # 1. Standardize column names (lowercase, snake_case)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # 2. Define the exact time columns to convert based on the confirmed schema
        time_cols_to_process = [
            'total_time', 'roxzone_time', 'run_time', 'work_time', 
        ] + [f'run_{i}' for i in range(1, 9)] + [f'work_{i}' for i in range(1, 9)]

        # 3. Conversion Logic
        all_time_cols_converted = True
        
        for col in time_cols_to_process:
            if col in df.columns:
                target_col = f'{col}_min'
                
                # Apply conversion
                converted_series = df[col].apply(time_to_minutes)
                
                # CRITICAL CHECK for total_time
                if col == 'total_time':
                    # Check if all values are NaN after conversion
                    if converted_series.isna().all() and len(converted_series) > 0:
                        all_time_cols_converted = False
                        break
                    
                df[target_col] = converted_series
                
                # Drop the original column if its type was not numeric to prevent mix-ups
                if df[col].dtype == object and target_col != col:
                    df = df.drop(columns=[col])

        if not all_time_cols_converted or 'total_time_min' not in df.columns:
            raise KeyError("Conversion to 'total_time_min' failed. Data may be malformed or in an unknown time format.")
        

        # 4. Apply filters
        df = df.dropna(subset=['total_time_min'])

        if gender_filter and gender_filter.lower() != 'all' and 'gender' in df.columns:
            df = df[df['gender'].str.casefold() == gender_filter.lower()]
            
        if division_filter and division_filter.lower() != 'all' and 'division' in df.columns:
            df = df[df['division'].str.casefold() == division_filter.lower()]
            
        if total_time_range and len(total_time_range) == 2:
            lower, upper = total_time_range
            df = df[
                (df['total_time_min'] >= lower) & 
                (df['total_time_min'] <= upper)
            ]
        
        return df.reset_index(drop=True)

    except ImportError:
        st.error("Error: Reading Parquet files requires 'pyarrow'. Please ensure it's installed.")
        st.stop()
    except Exception as e:
        st.exception(e)
        st.error(f"An unexpected error occurred during data fetching or processing. See traceback above.")
        return pd.DataFrame()


# --- Streamlit UI ---
st.title("🏋️ Hyrox Data Explorer (Live CDN)")
st.caption("Data is fetched live from the official Pyrox CDN source via Parquet files.")


# Load available races once
manifest_df = fetch_race_manifest()
if manifest_df.empty:
    st.error("Cannot proceed. Could not load the live race manifest.")
    st.stop()
    
race_options = manifest_df['race_id'].tolist()


# --- Sidebar for Filtering ---
st.sidebar.header("Data Selection & Filters")

# 1. Race Selection
selected_race_id = st.sidebar.selectbox("Select Race:", race_options)
selected_race_path = manifest_df[manifest_df['race_id'] == selected_race_id]['path'].iloc[0]


# 2. Gender and Division Filters
GENDERS = ['All', 'Female', 'Male', 'Mixed']
DIVISIONS = ['All', 'Pro', 'Open', 'Doubles', 'Relay', 'Staff'] 

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


# --- Main Content ---
st.header(f"Results: {selected_race_id}")
st.markdown(f"**Filters:** Gender: `{selected_gender}`, Division: `{selected_division}`, Time Range: `{time_range or 'None'}`")
st.divider()


if st.button("Fetch / Refresh Race Data"):
    with st.spinner(f"Loading results for {selected_race_id}..."):
        
        # Calls the function that fetches live data from the CDN
        results_df = fetch_race_data(
            race_path=selected_race_path, 
            gender_filter=selected_gender,
            division_filter=selected_division,
            total_time_range=time_range
        )

        if results_df.empty:
            st.error("Could not load or process data. Check the filters and manifest.")
            st.stop()
        
        st.success(f"Successfully loaded **{len(results_df)}** entries.")
            
        # Define columns for display (all the new *_min columns)
        DISPLAY_TIME_MIN_COLS = [col for col in results_df.columns if col.endswith('_min')]
        
        # --- Format the MAIN results table for display ---
        display_df = results_df.copy()
            
        for col_min in DISPLAY_TIME_MIN_COLS:
            # Get the original column name (e.g., 'work_1')
            original_col = col_min.replace('_min', '')
            
            # Look up the custom name, fall back to a cleaned version if not found
            if original_col in STATION_NAMES_MAP:
                base_name = STATION_NAMES_MAP[original_col]
            else:
                base_name = original_col.replace('_time', '').replace('_', ' ').title()
                
            display_col_name = f"{base_name} (MM:SS.ss)"
            display_df[display_col_name] = minutes_to_mmss_string(display_df[col_min]) 
            
            # Drop the raw _min column after formatting
            display_df = display_df.drop(columns=[col_min])
            
        st.subheader("Filtered Race Results")
        # Ensure correct column order: rank, name, then all time columns
        final_display_cols = [c for c in ['rank', 'name'] + [col for col in display_df.columns if '(MM:SS.ss)' in col] if c in display_df.columns]
        st.dataframe(display_df[final_display_cols])
            
        # --- Calculate and format the STATISTICS table ---
        st.subheader("Station Time Statistics (MM:SS.ss)")
            
        STAT_MIN_COLS = [col for col in results_df.columns if col.endswith('_min')]
        
        # Exclude only work_time_min, as total_time_min is useful here
        STAT_MIN_COLS = [col for col in STAT_MIN_COLS if col not in ['work_time_min']]
            
        if not STAT_MIN_COLS:
            st.warning("No time columns found to calculate statistics.")
        else:
            time_stats = results_df[
                results_df.columns.intersection(STAT_MIN_COLS)
            ].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            
            # Rename the index of the statistics table using the map
            stats_index_map = {f'{k}_min': v for k, v in STATION_NAMES_MAP.items() if f'{k}_min' in time_stats.index}
            time_stats = time_stats.rename(index=stats_index_map)
                
            STAT_COLUMNS_TO_FORMAT = ['mean', '50%', '75%']
                
            for stat in STAT_COLUMNS_TO_FORMAT:
                time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
                
            st.dataframe(time_stats)