# SIH 26084: Convective-scale Nowcasting Prototype

## Setup and Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python run.py`
3. View the dashboard: Open [http://localhost:8000/static/index.html](http://localhost:8000/static/index.html)

## Architecture Flow
[1 DATA] Synthetic storm generator (now) → IMD DWR / INSAT / IMERG (future)
    ▼
[2 PREPROCESS] 64x64 grid, ~1.5 km/pixel, 10-min frames, dBZ → rain rate (Z=200·R^1.6)
    ▼
[3 NOWCAST ENGINE]
    Optical flow (Farneback) ──┐
                               ├─► BLEND (flow weight drops with lead time)
    Small ConvLSTM (PyTorch) ──┘     → 36 frames (T+10 … T+360 min)
                                     + 5 perturbed members → probabilities
    ▼
[4 HAZARDS] Thunderstorm ≥35 dBZ | Hail ≥50 dBZ + fast growth | Cloudburst ≥100 mm/hr, ≥20 km², ≥10 min
    → probability + severity (Low / Moderate / High / Extreme)
    ▼
[5 API: FastAPI] /api/nowcast  /api/hazards  /api/alerts  /api/metrics
    ▼
[6 DASHBOARD] Leaflet map + time slider + hazard layers + alert panel + CSI chart
    ▼
[7 FUTURE] SMS / CAP alerts to SDMA and districts

## Data Sources
- Current: Synthetic data.
- Future: IMD DWR, INSAT-3D/3DR via MOSDAC, GPM IMERG, ERA5.

## Limitations
- Trained on synthetic data due to lack of immediate access to real DWR feeds.
- The ConvLSTM is a dummy/small model suitable for hackathon evaluation without a GPU.

## Future Scope
- Real DWR integration via API or NetCDF feeds.
- Lightning data integration (IITM / GLD360).
- NWP assimilation (WRF).
- SMS/CAP alerts to citizens and SDMA integration.
- Multilingual alerts.
