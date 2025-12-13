import streamlit as st
import pandas as pd
# --- CHANGE 1: Import the new helper function from the library ---
from pyrox import PyroxClient, RaceNotFound, AthleteNotFound, PyroxError, minutes_to_mmss_string 

# --- Setup ---
# Initialize the Pyrox Client (it manages its own cache in ~/.cache/pyrox)
@st.cache_resource
def get_pyrox_client():
    """Create and cache the Pyrox client instance."""
    return PyroxClient()

client = get_pyrox_client()

# --- Functions to Fetch Data ---
@st.cache_data(ttl=7200) # Cache the list for 2 hours (Pyrox manifest TTL)
def get_race_list():
    """Fetch the list of available races."""
    return client.list_races()

@st.cache_data(ttl=7200, show_spinner=False) # Use a custom spinner message below
def fetch_race_results(season, location, gender, division, total_time_range=None):
    """
    Fetch and cache specific race results using PyroxClient.get_race().
    """
    if total_time_range and len(total_time_range) == 2:
        lower, upper = total_time_range
    else:
        lower, upper = None, None

    # Call the Pyrox core method
    return client.get_race(
        season=season,
        location=location,
        gender=gender if gender != 'All' else None,
        division=division if division != 'All' else None,
        total_time=(lower, upper),
        use_cache=True
    )

# --- Streamlit UI ---
st.title("🏋️ Pyrox Client Demo: Hyrox Data Explorer")

# --- Sidebar for Filters ---
st.sidebar.header("Data Filters")

# 1. Race Selection
try:
    race_manifest = get_race_list()
    # Create combined (Season, Location) key for selection
    race_manifest['key'] = race_manifest.apply(
        lambda row: f"S{row['season']} - {row['location']}", axis=1
    )
    
    selected_key = st.sidebar.selectbox(
        "Select Race:",
        race_manifest['key'].unique()
    )
    
    # Parse selected key back into season and location
    selected_row = race_manifest[race_manifest['key'] == selected_key].iloc[0]
    selected_season = int(selected_row['season'])
    selected_location = selected_row['location']

    st.sidebar.write(f"**Fetching:** Season {selected_season}, {selected_location}")
    
except PyroxError as e:
    st.error(f"Could not load race manifest. Check CDN/network connection. Error: {e}")
    st.stop()


# 2. Gender and Division Filters (Optional)
GENDERS = ['All', 'Female', 'Male']
DIVISIONS = ['All', 'Pro', 'Open', 'Doubles'] # Simplified list

selected_gender = st.sidebar.selectbox("Gender Filter (Pre-fetch):", GENDERS)
selected_division = st.sidebar.selectbox("Division Filter (Pre-fetch):", DIVISIONS)


# 3. Total Time Filter (Post-fetch)
st.sidebar.subheader("Total Time Range (Minutes)")
time_min = st.sidebar.number_input("Minimum Time (Minutes):", value=None, min_value=0.0)
time_max = st.sidebar.number_input("Maximum Time (Minutes):", value=None, min_value=0.0)

time_range = None
if time_min is not None or time_max is not None:
    # Ensure min < max if both are provided
    if time_min is not None and time_max is not None and time_min > time_max:
        st.sidebar.warning("Minimum time must be less than maximum time.")
    else:
        time_range = (time_min, time_max)


# --- CHANGE 2: Define the list of columns to be formatted ---
TIME_COLUMNS = list(client.constants.WORK_STATION_RENAMES.values()) + [
    "total_time",
    "work_time",
    "roxzone_time",
    "run_time",
]

# --- Main Content ---
st.header(f"Results: {selected_key}")
st.markdown(f"**Filters:** Gender: `{selected_gender}`, Division: `{selected_division}`")
st.markdown(f"**Time Range:** {time_range or 'None'}")
st.divider()

# Fetch and Display Results
if st.button("Fetch / Refresh Race Data"):
    with st.spinner(f"Fetching {selected_key} with filters... (May take a moment if not cached)"):
        try:
            results_df = fetch_race_results(
                season=selected_season,
                location=selected_location,
                gender=selected_gender,
                division=selected_division,
                total_time_range=time_range
            )
            
            # Save the original decimal results to the session state for further analysis 
            st.session_state['current_results'] = results_df.copy() 
            
            st.success(f"Successfully loaded **{len(results_df)}** entries.")
            
            # --- CHANGE 3A: Format the MAIN results table for display (Fixes limited display and decimal time) ---
            display_df = results_df.copy() # Create a display-only copy
            
            for col in TIME_COLUMNS:
                if col in display_df.columns:
                    # Convert the decimal minutes to a readable MM:SS.ss string
                    display_df[col] = minutes_to_mmss_string(display_df[col]) 
            
            st.subheader("Full Race Results (Time in MM:SS.ss)")
            # FIX: st.dataframe(display_df) shows the full results (no .head(10))
            st.dataframe(display_df) 
            
            # --- CHANGE 3B: Calculate and format the STATISTICS table (Fixes decimal time in statistics) ---
            st.subheader("Station Time Statistics (MM:SS.ss)")
            
            # Select relevant time columns for analysis (using constants mapping)
            time_cols_for_stats = list(client.constants.WORK_STATION_RENAMES.values()) + ['total_time', 'roxzone_time']
            time_stats = results_df[
                results_df.columns.intersection(time_cols_for_stats)
            ].describe(percentiles=[.50, .75]).loc[['mean', '50%', '75%']].transpose()
            
            # CRITICAL FIX: Format the calculated statistics columns
            STAT_COLUMNS_TO_FORMAT = ['mean', '50%', '75%']
            
            for stat in STAT_COLUMNS_TO_FORMAT:
                # FIX: Use column selection (time_stats[stat]) instead of row selection (.loc[stat])
                time_stats[stat] = minutes_to_mmss_string(time_stats[stat])
            
            st.dataframe(time_stats)
            
        except RaceNotFound:
            st.warning(f"No results found for {selected_key} with the selected filters.")
        except Exception as e:
            # Re-raise the error to provide better debugging info if the 'mean' issue persists or another error occurs
            st.error(f"An unexpected error occurred during data retrieval: {e}")
            raise # Raise the exception to the Streamlit console for full traceback