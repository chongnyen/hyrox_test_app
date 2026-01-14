# feedback_logic.py

import sys
import os
# Essential for your project structure to handle local imports like src.models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import joblib
import os
import pandas as pd
import torch

from src.rag.vector_store import build_chroma_vector_store_from_df
from src.models import HyroxModelRequest
from src.data_utils import find_age_range
from src import pipeline
from src.pipeline import extract_race_minutes_from_df
from chromadb import PersistentClient
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig


# global definitions for LLM/RAG system
MODEL_PATH = "Syllerim/hyrox_mistral_lora_model"
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_full_data_instructed_feedback.csv")
MIN_MAX_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_stations.pkl")
AGE_SCALER_PATH = os.path.join(ROOT_DIR, "data", "processed", "hyrox_minmax_scaler_ages.pkl")

CHROMA_FOLDER_PATH = '../data/store'
os.makedirs(CHROMA_FOLDER_PATH, exist_ok=True)

df = None
stations_scaler = None
age_scaler = None

model = None # This is the LLM (Mistral/PEFT)
tokenizer = None
generator = None

# --- Feature list for the MLflow Regression Model (Time Prediction) ---
# This feature list MUST be in the exact order the regression model was trained
RUN_COLUMNS = [f'run_{i}' for i in range(1, 9)] 
WORK_COLUMNS = [f'work_{i}' for i in range(1, 9)] 
ROXZONE_COLUMNS = [f'roxzone_{i}' for i in range(1, 8)] 

REGRESSION_FEATURE_COLUMNS = (
    ['age'] + 
    RUN_COLUMNS + 
    WORK_COLUMNS + 
    ROXZONE_COLUMNS + 
    ['gender_Male'] # Assuming 'Female' is the dropped baseline after OHE
)
# ----------------------------------------------------------------------------
def load_model():
    # This function loads the LLM/RAG components
    global df, stations_scaler, age_scaler, model, tokenizer, generator
    if model is None or tokenizer is None or generator is None or df is None or stations_scaler is None or age_scaler is None:
        df = pd.read_csv(CSV_PATH)
        stations_scaler = joblib.load(MIN_MAX_SCALER_PATH)
        age_scaler = joblib.load(AGE_SCALER_PATH)

        peft_model_id = MODEL_PATH
        peft_config = PeftConfig.from_pretrained(peft_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        merged = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        generator = pipeline(
            "text-generation",
            model=merged,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            max_new_tokens=300,
            do_sample=True
        )
    return generator

# ----------------------------------------------------------------------------
# MODIFIED: Accepts the MLflow model (model_regressor) as an argument
def generate_feedback(data: HyroxModelRequest, model_regressor): 
    
    # Load RAG/LLM components on first call
    load_model()

    # --- NEW: Regression Model Prediction (Quantitative Analysis) ---
    
    # 1. Convert Pydantic model to a single-row DataFrame
    input_dict = data.model_dump(exclude_none=True) 
    input_df = pd.DataFrame([input_dict])

    # 2. Handle Categorical Features (One-Hot Encoding for 'gender')
    # This is required for the ML model input
    input_df = pd.get_dummies(input_df, columns=['gender'], drop_first=True)
    
    # 3. Column Alignment (Crucial Step: Must match training data)
    for col in REGRESSION_FEATURE_COLUMNS:
        if col not in input_df.columns:
            # Fill missing OHE columns (e.g., if gender was 'Female') with 0.0
            input_df[col] = 0.0

    # Select and reorder columns to match the trained regression model's feature space
    input_features = input_df[REGRESSION_FEATURE_COLUMNS]

    # 4. Prediction
    predicted_time = model_regressor.predict(input_features)[0]
    predicted_time_mins = round(float(predicted_time), 2)
    
    # --- END NEW: Regression Model Prediction ---


    # process input data through the full pipeline (used for RAG/LLM features)
    processed_df = pipeline.process_hyrox_request(data) 

    # prepare query to the vector store
    query_text = "I want to find participants who had similar overall performance to me."

    race_minutes = pipeline.extract_race_minutes_from_df(processed_df)

    # perform vector search
    collection_name = "hyrox_participants"
    persist_dir = CHROMA_FOLDER_PATH
    client = PersistentClient(path=persist_dir)
    collection = client.get_collection(name=collection_name)
    results = collection.query(query_texts=[query_text], n_results=1, where={"race_minutes": race_minutes})
    
    if not results["documents"] or not results["documents"][0]:
        return {"error": "No similar participants found."}

    # convert vector search result back into a DataFrame
    similar_participant_data = pd.DataFrame([results["documents"][0]["metadata"]])

    # analyze improvement suggestions
    perf_only_suggestions = analyze_improvement_from_similar_perf_only(processed_df, similar_participant_data)
    perf_context_suggestions = analyze_improvement_from_similar_perf_context(processed_df, similar_participant_data)

    # --- Update the return dictionary to include the quantitative prediction ---
    return {
        "predicted_total_time_minutes": predicted_time_mins, # <-- NEW
        "performance_feedback": processed_df["performance_feedback"].iloc[0],
        "perf_only_improvements": perf_only_suggestions,
        "perf_context_improvements": perf_context_suggestions 
    }


def build_prompt(data):
    # This function is currently commented out/placeholder
    gender_map = {"male": 0, "female": 1}
    gender_value = gender_map.get(data.gender.lower())

    # prompt = f"""You are a HYROX coach. Given the following normalized performance values for a race participant...
    # return prompt
    pass


def analyze_improvement_from_similar_perf_only(user_df: pd.DataFrame, similar_df: pd.DataFrame) -> dict:
    """
    Compares the user's weakest stations (from suggestions_perf_only) with those of a similar participant
    and estimates potential improvement.

    Returns a dict mapping station names to estimated improvement in z-score.
    """
    user_row = user_df.iloc[0]
    similar_row = similar_df.iloc[0]
    weak_stations = user_row["suggestions_perf_only"]

    improvement_dict = {}

    for station in weak_stations:
        station_key = f"{station}_zscore"
        if station_key in similar_df.columns:
            user_score = user_row[station_key]
            similar_score = similar_row[station_key]

            if similar_score < user_score:
                estimated_improvement = user_score - similar_score
                improvement_dict[station] = round(estimated_improvement, 2)

    return improvement_dict


def analyze_improvement_from_similar_perf_context(user_df: pd.DataFrame, similar_df: pd.DataFrame) -> dict:
    """
    Same logic as above but using suggestions_perf_context and zscore_context columns.

    Parameters:
    user_df : pd.DataFrame
        DataFrame containing the processed user data (single row).
    similar_df : pd.DataFrame
        DataFrame containing the most similar participant data from vector store.

    Returns:
    dict
        Dictionary mapping station names to estimated improvement in minutes.
    """
    user_row = user_df.iloc[0]
    similar_row = similar_df.iloc[0]

    weak_stations = user_row["suggestions_perf_context"]

    improvement_dict = {}

    for station in weak_stations:
        station_key = f"{station}_zscore_context"
        if station_key in similar_df.columns:
            user_score = user_row[station_key]
            similar_score = similar_row[station_key]

            if similar_score < user_score:
                estimated_improvement = user_row[station] - similar_row[station]
                improvement_dict[station] = round(estimated_improvement, 2)

    return improvement_dict