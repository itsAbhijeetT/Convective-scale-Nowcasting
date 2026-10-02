from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json
import os

from models.blend import blend_nowcasts
from hazards.hazards import detect_hazards

app = FastAPI(title="SIH Nowcasting API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if not os.path.exists("static"): os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

DATA_FILE = "data/synthetic_data.npy"
DEMO_FILE = "data/demo_event.npy"
current_event = None
ensemble = None

def load_event(path):
    global current_event, ensemble
    if not os.path.exists(path):
        return False
    data = np.load(path)
    current_event = data[0] if len(data.shape) == 4 else data
    
    # Pre-generate ensemble for 36 steps to serve quickly
    past_frames = current_event[:6]
    ensemble = blend_nowcasts(past_frames, lead_steps=36)
    return True

@app.on_event("startup")
def startup():
    if not load_event(DATA_FILE):
        print("Warning: No synthetic data found. Run python run.py first.")

@app.post("/api/demo")
def switch_to_demo():
    if load_event(DEMO_FILE):
        return {"status": "switched to demo cloudburst"}
    return {"error": "demo file not found"}

@app.get("/api/observed")
def get_observed():
    if current_event is None: return []
    return current_event[:6].tolist()

@app.get("/api/nowcast")
def get_nowcast(lead: int):
    if ensemble is None or lead < 1 or lead > 36: return []
    # return the mean prediction for the lead index (0-35)
    mean_pred = np.mean(ensemble[:, lead - 1], axis=0)
    return {
        "grid": mean_pred.tolist(),
        "bounds": {"lat": [29.8, 30.6], "lon": [77.6, 78.6]}
    }

@app.get("/api/hazards")
def get_api_hazards(lead: int, min_prob: float = 0.3):
    if ensemble is None or lead < 1 or lead > 36: return {"type":"FeatureCollection","features":[]}
    alerts = detect_hazards(ensemble, lead - 1)
    
    features = []
    for a in alerts:
        if a["probability"] >= min_prob:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
                "properties": a
            })
    return {"type": "FeatureCollection", "features": features}

@app.get("/api/alerts")
def get_all_alerts():
    if ensemble is None: return []
    all_alerts = []
    for lead_idx in range(36):
        alerts = detect_hazards(ensemble, lead_idx)
        all_alerts.extend(alerts)
    
    # deduplicate roughly and sort by ETA
    all_alerts.sort(key=lambda x: x["eta_minutes"])
    return all_alerts[:10] # top 10

@app.get("/api/metrics")
def get_metrics():
    if os.path.exists("eval/metrics.json"):
        with open("eval/metrics.json") as f:
            return json.load(f)
    return {}
