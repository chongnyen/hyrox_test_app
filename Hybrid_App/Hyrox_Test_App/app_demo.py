import streamlit as st
import pandas as pd
import requests
import io

# --- Configuration ---
CDN_BASE = "https://d2wl4b7sx66tfb.cloudfront.net"
MANIFEST_URL = f"{CDN_BASE}/manifest/latest.csv"

# --- Helper Functions ---

# Helper to convert MM:SS or HH:MM:SS to total minutes
def time_to_minutes(time_str):
    if pd.isna(time_str):
        return None
    try:
        # If the value is already a number (float/int), assume it's already in minutes or seconds.
        if isinstance(time_str, (int, float)):
             # Assuming if a large number, it might be seconds. If small, it's minutes.
             # We rely on the caller to ensure time_str is reasonable.
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

# Helper to convert minutes back to readable MM:SS.ss string for display
def minutes_to_mmss_string(minutes):
    if isinstance(minutes, pd.Series):
        return minutes.apply(minutes_to_mmss_string)

    if pd.isna(minutes) or minutes is None:
        return ""
    
    if not isinstance(minutes, (int, float)):
        return ""

    total_seconds = minutes * 60
    # Use floor to get minutes, and mod 60 for seconds
    m = int(total_seconds // 60)
    s = total_seconds % 60
    
    # Simple formatting
    return f"{m:02d}:{s:05.2f}"

# --- Data Fetching Functions (Using CDN Strategy) ---

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
    
    # 1. Construct the direct CDN URL for the Parquet file
    s3_key = race_path.split("/", 4)[4] if race_path.startswith("s3://") else race_path
    data_url = f"{CDN_BASE}/{s3_key}"

    try:
        # Requires pyarrow for reading parquet directly from URL
        df = pd.read_parquet(data_url)
        
        # 2. Standardize column names (lowercase, replace spaces/hyphens with underscores)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
        
        # 3. Define the critical renaming map for all known columns
        rename_map = {
            'run_time_split': 'run_time',
            'roxzone': 'roxzone_time',
            'ski_erg': 'ski_erg_time',
            'sled_push': 'sled_push_time',
            'sled_pull': 'sled_pull_time',
            'burpee_broad_jump': 'burpee_broad_jump_time',
            'row': 'row_time',
            'farmers_carry': 'farmers_carry_time',
            'sandbag_lunge': 'sandbag_lunge_time',
            'wall_balls': 'wall_balls_time',
            'time': 'total_time' 
        }
        df = df.rename(columns=rename_map, errors='ignore')

        # 3b. Defensive Check for 'total_time' (The Fix)
        if 'total_time' not in df.columns:
            candidate_total_time_cols = [
                'finish_time', 'race_time', 'result_time', 'final_time', 
                'totaltime', 'finishtime', 'racetime'
            ]
            
            found_total_time = False
            for cand_col in candidate_total_time_cols:
                if cand_col in df.columns:
                    df = df.rename(columns={cand_col: 'total_time'})
                    found_total_time = True
                    break
            
            if not found_total_time and 'total_time' not in df.columns:
                st.error(f"Debug Info: Columns found in data: {list(df.columns)}")
                raise KeyError("The race data is missing the required 'total_time' column after all known renaming attempts.")
        
        # 4. Conversion Logic
        
        # List of all time columns we need to process (including the confirmed 'total_time')
        # We ensure 'work_time' is included if it exists in the data.
        time_cols_to_process = list(set(rename_map.values())) + ['work_time']
        
        for col in time_cols_to_process:
            if col in df.columns:
                # Determine the target column name (e.g., 'total_time' -> 'total_time_min')
                target_col = col.replace('_time', '_min') if col.endswith('_time') else f'{col}_min'
                
                # Apply conversion
                df[target_col] = df[col].apply(time_to_minutes)
                
                # If the original column was a string (object dtype), drop it to clean up the frame.
                if df[col].dtype == object and target_col != col:
                    df = df.drop(columns=[col])
        
        # Final check for the essential column after conversion
        if 'total_time_min' not in df.columns:
            raise KeyError("Conversion to 'total_time_min' failed. Data may be malformed.")


        # 5. Apply filters
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
        st.error(f"Error reading data from live CDN. Check the filters, manifest, or that 'pyarrow' is in your requirements.txt.")
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
            
        # Define columns for display
        DISPLAY_TIME_MIN_COLS = [col for col in results_df.columns if col.endswith('_min')]
        
        # --- Format the MAIN results table for display ---
        display_df = results_df.copy()
            
        for col_min in DISPLAY_TIME_MIN_COLS:
            # Clean up the name for display (e.g., 'ski_erg_time_min' -> 'Ski Erg (MM:SS.ss)')
            base_name = col_min.replace('_min', '').replace('_time', '').replace('_', ' ').title()
            display_col_name = f"{base_name} (MM:SS.ss)"
            display_df[display_col_name] = minutes_to_mmss_string(display_df[col_min]) 
            
            # Drop the raw _min column after formatting
            display_df = display_df.drop(columns=[col_min])
            
        st.subheader("Filtered Race Results")
        final_display_cols = [c for c in ['rank', 'name'] + [col for col in display_df.columns if '(MM:SS.ss)' in col] if c in display_df.columns]
        st.dataframe(display_df[final_display_cols])
            
        # --- Calculate and format the STATISTICS table ---
        st.subheader("Station Time Statistics (MM:SS.ss)")
            
        STAT_MIN_COLS = [col for col in results_df.columns if col.endswith('_min') and col not in ['work_time_min']]
            
        if not STAT_MIN_COLS:
            st.warning("No time columns found to calculate statistics.")
            # st.stop() # Do not stop, let the rest of the app run
            
        else:
            time_stats = results_df[
                results_df.columns.intersection(STAT_MIN_COLS)
            ].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
                
            STAT_COLUMNS_TO_FORMAT = ['mean', '50%', '75%']
                
            for stat in STAT_COLUMNS_TO_FORMAT:
                time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
                
            st.dataframe(time_stats)