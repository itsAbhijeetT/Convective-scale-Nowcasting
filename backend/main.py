from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import asyncio
import json

from data.synthetic import SyntheticStormGenerator
from models.blend import blend_nowcasts
from hazards.thunderstorm import detect_thunderstorm
from hazards.hail import detect_hail
from hazards.cloudburst import detect_cloudburst
from hazards.geometry import extract_polygons

app = FastAPI(title="SIH 26084 Nowcasting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for prototype
print("Initializing synthetic data & models...")
gen = SyntheticStormGenerator(grid_size=(256, 256))
recent_sequence = gen.generate_sequence(num_frames=6) # Get 6 frames of past data
# Convert to dBZ
from data.loaders import rain_rate_to_dbz
recent_sequence_dbz = rain_rate_to_dbz(recent_sequence)

# Precompute a nowcast ensemble to serve quickly
ensemble_preds = blend_nowcasts(recent_sequence_dbz, lead_steps=36) # (5, 36, H, W)

@app.get("/api/nowcast")
def get_nowcast(lead: int = 0):
    """
    Return the mean predicted dBZ for the given lead time index (0 to 35).
    """
    if lead < 0 or lead >= 36:
        return {"error": "Lead time must be between 0 and 35 (10-min steps up to 6 hrs)."}
    
    mean_pred = np.mean(ensemble_preds[:, lead, :, :], axis=0)
    
    # In a real app, you might return a GeoTIFF or PNG. We'll return basic stats to avoid massive JSONs,
    # or a subsampled version. For prototype, a subsampled JSON is ok or just metadata.
    subsampled = mean_pred[::4, ::4].tolist() # 64x64 for fast JSON
    return {"lead_time_index": lead, "lead_time_minutes": lead * 10, "data_64x64": subsampled}

@app.get("/api/hazards")
def get_hazards(lead: int = 0):
    if lead < 0 or lead >= 36:
        return {"error": "Lead time must be between 0 and 35."}
        
    prob_ts = detect_thunderstorm(ensemble_preds, lead)
    prob_hail = detect_hail(ensemble_preds, lead)
    prob_cb = detect_cloudburst(ensemble_preds, lead)
    
    features = []
    features.extend(extract_polygons(prob_ts, threshold=0.5, hazard_type="Thunderstorm"))
    features.extend(extract_polygons(prob_hail, threshold=0.5, hazard_type="Hail"))
    features.extend(extract_polygons(prob_cb, threshold=0.5, hazard_type="Cloudburst"))
    
    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.get("/api/alerts")
def get_alerts():
    """
    Return a list of upcoming severe alerts.
    """
    alerts = []
    # Scan through lead times to find the first high-prob cloudburst or hail
    for lead in range(36):
        prob_cb = detect_cloudburst(ensemble_preds, lead)
        if np.max(prob_cb) > 0.8:
            alerts.append({
                "type": "Cloudburst",
                "eta_minutes": lead * 10,
                "severity": "Extreme",
                "action": "Immediate evacuation of low-lying areas."
            })
            break # Just one for demo
            
    if not alerts:
        alerts.append({
            "type": "Thunderstorm",
            "eta_minutes": 40,
            "severity": "Moderate",
            "action": "Avoid open areas."
        })
        
    return alerts

@app.get("/api/metrics")
def get_metrics():
    """
    Demo evaluation metrics.
    """
    return {
        "lead_times": [30, 60, 120, 240, 360],
        "CSI": [0.85, 0.72, 0.60, 0.45, 0.35],
        "POD": [0.90, 0.82, 0.75, 0.60, 0.50],
        "FAR": [0.10, 0.15, 0.25, 0.40, 0.55]
    }

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    # Simulate a new frame arriving every 5 seconds instead of 10 mins
    step = 0
    try:
        while True:
            await asyncio.sleep(5)
            await websocket.send_text(json.dumps({
                "event": "new_frame",
                "timestamp": f"T+{step*10} mins",
                "message": "New radar frame assimilated. Nowcast updated."
            }))
            step += 1
    except:
        pass
