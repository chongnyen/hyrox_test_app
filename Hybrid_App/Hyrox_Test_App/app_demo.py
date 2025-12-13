import streamlit as st
import pandas as pd
# Import the new helper function from the library
from pyrox import PyroxClient, RaceNotFound, AthleteNotFound, PyroxError, minutes_to_mmss_string 

# --- Setup ---
# Initialize the Pyrox Client (Runs only once and is cached)
@st.cache_resource
def get_pyrox_client():
    """Create and cache the Pyrox client instance."""
    return PyroxClient()

client = get_pyrox_client()

# --- Functions to Fetch Data ---
# Note: We keep these functions, but they will ONLY be called after the button click.
@st.cache_data(ttl=7200) 
def get_race_list():
    """Fetch the list of available races."""
    return client.list_races()

@st.cache_data(ttl=7200, show_spinner=False)
def fetch_race_results(season, location, gender, division, total_time_range=None):
    """
    Fetch and cache specific race results using PyroxClient.get_race().
    """
    # ... (Your existing fetch_race_results code remains the same) ...
    if total_time_range and len(total_time_range) == 2:
        lower, upper = total_time_range
    else:
        lower, upper = None, None

    return client.get_race(
        season=season,
        location=location,
        gender=gender if gender != 'All' else None,
        division=division if division != 'All' else None,
        total_time=(lower, upper),
        use_cache=True
    )
    
# --- Streamlit UI ---
st.title("🏋️ Pyrox Client Demo: Hyrox Data Explorer (Safe Startup)")
st.caption("Click 'Fetch Race List' first to load available races from the Hyrox server.")

# --- Sidebar for Filters ---
st.sidebar.header("Data Filters")

# NEW SAFE STARTUP BLOCK: Use a button to trigger the slow manifest fetch
if 'race_manifest' not in st.session_state:
    st.session_state['race_manifest'] = None
    st.sidebar.warning("Manifest not loaded. Click the button below.")

# 1. Button to fetch the manifest
if st.sidebar.button("Fetch Race List"):
    with st.spinner("Connecting to Hyrox server to get list of available races..."):
        try:
            st.session_state['race_manifest'] = get_race_list()
            st.sidebar.success("Race list loaded!")
        except PyroxError as e:
            st.sidebar.error(f"Could not load race manifest. Error: {e}")

# 2. Race Selection (Only appears AFTER the manifest is loaded)
selected_season = None
selected_location = None

if st.session_state['race_manifest'] is not None:
    race_manifest = st.session_state['race_manifest']
    race_manifest['key'] = race_manifest.apply(
        lambda row: f"S{row['season']} - {row['location']}", axis=1
    )
    
    selected_key = st.sidebar.selectbox(
        "Select Race:",
        race_manifest['key'].unique(),
        key='race_key_selector'
    )
    
    # Parse selected key back into season and location
    selected_row = race_manifest[race_manifest['key'] == selected_key].iloc[0]
    selected_season = int(selected_row['season'])
    selected_location = selected_row['location']

    st.sidebar.write(f"**Selected:** Season {selected_season}, {selected_location}")
    
# If the manifest is not loaded, we use mock values to prevent crashes
else:
    st.sidebar.selectbox("Select Race:", ["(Click 'Fetch Race List' first)"])
    selected_season = 6 # Mock value
    selected_location = 'Maastricht' # Mock value

# 3. Gender and Division Filters
GENDERS = ['All', 'Female', 'Male']
DIVISIONS = ['All', 'Pro', 'Open', 'Doubles']

selected_gender = st.sidebar.selectbox("Gender Filter (Pre-fetch):", GENDERS, key='gender_select')
selected_division = st.sidebar.selectbox("Division Filter (Pre-fetch):", DIVISIONS, key='division_select')


# 4. Total Time Filter
st.sidebar.subheader("Total Time Range (Minutes)")
time_min = st.sidebar.number_input("Minimum Time (Minutes):", value=None, min_value=0.0, key='time_min')
time_max = st.sidebar.number_input("Maximum Time (Minutes):", value=None, min_value=0.0, key='time_max')

time_range = None
if time_min is not None or time_max is not None:
    # Ensure min < max if both are provided
    if time_min is not None and time_max is not None and time_min > time_max:
        st.sidebar.warning("Minimum time must be less than maximum time.")
    else:
        time_range = (time_min, time_max)

# --- Define the list of columns to be formatted ---
# Assuming client.constants.WORK_STATION_RENAMES is available
try:
    TIME_COLUMNS = list(client.constants.WORK_STATION_RENAMES.values()) + [
        "total_time", "work_time", "roxzone_time", "run_time",
    ]
except AttributeError:
    # Fallback if constants aren't loaded yet
    TIME_COLUMNS = ["total_time", "roxzone_time"]


# --- Main Content ---
st.header(f"Results: {selected_location} (S{selected_season})")
st.markdown(f"**Filters:** Gender: `{selected_gender}`, Division: `{selected_division}`, Time: `{time_range or 'None'}`")
st.divider()

# Fetch and Display Results (This is the slow part, it MUST be in the button block)
if st.button("Fetch / Refresh Race Data"):
    if st.session_state['race_manifest'] is None:
        st.error("Please click 'Fetch Race List' in the sidebar first.")
    else:
        with st.spinner(f"Fetching {selected_location} results... (This may take a moment)"):
            try:
                # Fetching the race results for the selected filters
                results_df = fetch_race_results(
                    season=selected_season,
                    location=selected_location,
                    gender=selected_gender,
                    division=selected_division,
                    total_time_range=time_range
                )
                
                # ... (The rest of your display and formatting code goes here) ...
                
                # Save the original decimal results to the session state for further analysis 
                st.session_state['current_results'] = results_df.copy() 
                
                st.success(f"Successfully loaded **{len(results_df)}** entries.")
                
                # --- Format the MAIN results table for display ---
                display_df = results_df.copy()
                
                for col in TIME_COLUMNS:
                    if col in display_df.columns:
                        display_df[col] = minutes_to_mmss_string(display_df[col]) 
                
                st.subheader("Full Race Results (Time in MM:SS.ss)")
                st.dataframe(display_df) 
                
                # --- Calculate and format the STATISTICS table ---
                st.subheader("Station Time Statistics (MM:SS.ss)")
                
                time_cols_for_stats = list(client.constants.WORK_STATION_RENAMES.values()) + ['total_time', 'roxzone_time']
                time_stats = results_df[
                    results_df.columns.intersection(time_cols_for_stats)
                ].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
                
                STAT_COLUMNS_TO_FORMAT = ['mean', '50%', '75%']
                
                for stat in STAT_COLUMNS_TO_FORMAT:
                    time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
                
                st.dataframe(time_stats)
                
            except RaceNotFound:
                st.warning(f"No results found for {selected_location} with the selected filters.")
            except Exception as e:
                st.error(f"An unexpected error occurred during data retrieval: {e}")