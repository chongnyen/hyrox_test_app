import streamlit as st
import requests
import json
import numpy as np

# --- 1. CONFIGURATION ---
# CRITICAL CHANGE: Use the internal host/port for Codespaces communication
API_URL = "http://localhost:8000/predict_feedback" 

# --- 2. FASTAPI COMMUNICATION FUNCTION ---

def process_race_data_via_api(request_data: dict) -> dict:
    """
    Function for calling the ML pipeline's FastAPI endpoint.
    """
    try:
        response = requests.post(API_URL, json=request_data)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = f"API Request Failed: Check if FastAPI server is running on port 8000.\nError: {e}"
        st.error(error_message)
        st.code(f"Request URL: {API_URL}\nData: {request_data}", language="json")
        return {"error": error_message}
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return {"error": str(e)}

# --- 3. STREAMLIT UI ---

st.set_page_config(
    page_title="HYROX Performance Feedback Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🐂 HYROX Performance Report Card")
st.markdown("Enter the athlete's details and station times below to generate a personalized performance report and feedback from the ML pipeline.")

# Define the input fields dynamically for better management
stations = {
    "run_1": "Run 1 (m)", "work_1": "1000m Ski (m)", "roxzone_1": "Roxzone 1 (m)",
    "run_2": "Run 2 (m)", "work_2": "50m Sled Push (m)", "roxzone_2": "Roxzone 2 (m)",
    "run_3": "Run 3 (m)", "work_3": "50m Sled Pull (m)", "roxzone_3": "Roxzone 3 (m)",
    "run_4": "Run 4 (m)", "work_4": "80m Burpee Broad Jump (m)", "roxzone_4": "Roxzone 4 (m)",
    "run_5": "Run 5 (m)", "work_5": "1000m Row (m)", "roxzone_5": "Roxzone 5 (m)",
    "run_6": "Run 6 (m)", "work_6": "200m Farmer Carry (m)", "roxzone_6": "Roxzone 6 (m)",
    "run_7": "Run 7 (m)", "work_7": "100m Sandbag Lunges (m)", "roxzone_7": "Roxzone 7 (m)",
    "run_8": "Run 8 (m)", "work_8": "100 Wall Balls (m)", "total_time": "Total Time (m)"
}

# --- Sidebar Inputs (Context) ---
st.sidebar.header("Athlete Context")
gender_options = ["Male", "Female"]
gender_raw = st.sidebar.radio("Gender", gender_options)
age_input = st.sidebar.slider("Age", min_value=18, max_value=70, value=30, step=1)
gender_code = "M" if gender_raw == "Male" else "F"

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API Endpoint:** `{API_URL}`")

# --- Main Area Inputs (Performance Data) ---
st.subheader("Performance Data (Minutes)")
st.warning("All times must be entered in **minutes** (e.g., 60 seconds = 1.0, 1 minute 30 seconds = 1.5).")

data_cols = list(stations.keys())
num_cols = len(data_cols) - 1 # Exclude total_time for input layout

# Organize inputs into 3 columns for better layout
cols = st.columns(3)
station_inputs = {}

for i, key in enumerate(data_cols):
    placeholder = 5.0 if 'run' in key else 10.0 if 'work' in key else 2.0
    label = stations[key]
    
    # Place inputs into one of the three columns
    col_index = i % 3
    
    station_inputs[key] = cols[col_index].number_input(
        label=label, 
        min_value=0.1, 
        value=placeholder, 
        step=0.1, 
        format="%.2f"
    )

# --- Button and Processing ---
st.markdown("---")
if st.button("Generate ML Report Card", type="primary"):
    # 1. Prepare Request Data
    try:
        request_data = {
            "gender": gender_code,
            "age": age_input
        }
        
        # Add all station data, mapping work_1 to 1000m Ski, etc.
        # The 'work' keys need to be mapped back to the proper station names 
        # based on the HyroxModelRequest model in main.py
        
        # Collect and combine run, work, and roxzone times
        for key in stations.keys():
            if key != 'total_time':
                 request_data[key] = station_inputs[key]
            else:
                 # This 'total_time' is not part of the HyroxModelRequest model,
                 # but we need it for the pipeline to compute residuals.
                 # The 'extract_race_minutes_from_df' function in the backend 
                 # expects this value, even though it's not in the Pydantic model.
                 # We will pass it as a separate key in the JSON body.
                 request_data['total_time_actual'] = station_inputs[key]

    except Exception as e:
        st.error(f"Error preparing data: {e}")
        st.stop()
        
    st.info("Sending request to FastAPI ML Pipeline...")
    
    # 2. Call the API
    with st.spinner('Waiting for ML pipeline to process data and generate feedback...'):
        api_result = process_race_data_via_api(request_data)

    # 3. Display Results
    if "Performance Feedback" in api_result:
        st.success("✅ Report Card Generated Successfully!")
        
        # The feedback is expected to be a structured string
        feedback_text = api_result["Performance Feedback"]
        
        st.markdown("## 📊 Personalized Feedback")
        st.text_area(
            label="HYROX AI Coach Report:",
            value=feedback_text,
            height=300
        )
        
    elif "error" in api_result:
        # Error already displayed inside process_race_data_via_api
        st.error("❌ Failed to generate report. Check the error message above.")