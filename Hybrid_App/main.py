from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

CSV_PATH = '/workspaces/hyrox_test_app/Hybrid_App/Hyrox_Test_App/data/processed/hyrox_final_model_data.csv'

try:
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
except:
    df = pd.DataFrame()

SEGMENTS = [
    ("Run 1", "run_1"), ("Ski Erg", "work_1"), ("Run 2", "run_2"), ("Sled Push", "work_2"),
    ("Run 3", "run_3"), ("Sled Pull", "work_3"), ("Run 4", "run_4"), ("Burpees", "work_4"),
    ("Run 5", "run_5"), ("Rowing", "work_5"), ("Run 6", "run_6"), ("Farmers Carry", "work_6"),
    ("Run 7", "run_7"), ("Sandbag Lunges", "work_7"), ("Run 8", "run_8"), ("Wall Balls", "work_8")
]

def get_detailed_plan(weakness):
    plans = {
        "Run": {
            "title": "Aerobic Power & Lactate Threshold",
            "focus": "Improving running economy and sustaining pace under fatigue.",
            "w1": "Session A: 8x400m @ 3k pace (90s rest). Session B: 45min Zone 2 + strides.",
            "w2": "Session A: 5x1km @ Hyrox Goal Pace. Session B: 60min Progressive Run.",
            "w3": "Session A: 3x2km @ 10k pace. Session B: 15km Long Easy Run.",
            "w4": "Deload: 30min Easy Run + Mobility & 4x200m light intervals."
        },
        "Strength": {
            "title": "Mechanical Power & Force Production",
            "focus": "Building raw strength for Sleds and Farmers Carry.",
            "w1": "Session A: Trapbar 5x5 (80% 1RM). Session B: Sled Push 8x25m (Heavy).",
            "w2": "Session A: Front Squat 4x6. Session B: Sled Pull 6x25m + Farmers Carry.",
            "w3": "Session A: Trapbar 3x3 (90% 1RM). Session B: 4 rounds: 50m Sled + 20 Lunges.",
            "w4": "Deload: Technique work (50% weight) + Hip & Ankle Mobility."
        },
        "Metcon": {
            "title": "Movement Economy & Work Capacity",
            "focus": "Optimizing efficiency in Burpees, Wall Balls, and transitions.",
            "w1": "Session A: 10 rounds: 1min Row / 1min Wall Balls. Session B: 30min EMOM: 12 Burpees.",
            "w2": "Session A: 5 rounds: 500m Ski + 30 Lunges + 500m Row. Session B: Wall Ball 10x15.",
            "w3": "Session A: Hyrox Sim: 1km Run + 80 Burpees + 1km Run + 100 Wall Balls.",
            "w4": "Deload: 20min light Metcon + Full Body Flow."
        }
    }
    if "Run" in weakness: return plans["Run"]
    if any(s in weakness for s in ["Sled", "Farmers", "Sandbag"]): return plans["Strength"]
    return plans["Metcon"]

@app.post("/analyze_custom")
async def analyze_custom(data: dict = Body(...)):
    gender, age = data.get("gender", "MALE").upper(), str(data.get("age_group"))
    target_pct = float(data.get("target_percentile", 0.10))
    peer_group = df[(df['gender'] == gender) & (df['age_group'] == age)]
    if peer_group.empty: peer_group = df[df['gender'] == gender]

    targets, stats = {}, {}
    for label, key in SEGMENTS:
        col = peer_group[key].dropna() if not peer_group.empty else pd.Series([5.0])
        targets[label] = float(col.quantile(target_pct))
        val = float(data.get(key, 5.0))
        stats[label] = {"z": (val - col.mean()) / (col.std() or 0.5)}

    worst = max(stats, key=lambda k: stats[k]["z"])
    return {"targets": targets, "stats": stats, "coaching": get_detailed_plan(worst)}