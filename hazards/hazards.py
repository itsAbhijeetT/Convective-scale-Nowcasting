import numpy as np
from scipy import ndimage

def dbz_to_rain_rate(dbz):
    z = 10 ** (dbz / 10.0)
    return (z / 200.0) ** (1 / 1.6)

def get_latlon(r, c):
    lat = 30.6 - r * (30.6 - 29.8) / 64
    lon = 77.6 + c * (78.6 - 77.6) / 64
    return lat, lon

def detect_hazards(ensemble, lead_idx):
    """
    ensemble: (5, 36, 64, 64)
    returns list of alerts for the given lead time index
    """
    mean_pred = np.mean(ensemble[:, lead_idx], axis=0)
    alerts = []
    
    # Cloudburst: rain rate >= 100 mm/hr, >= 9 pixels connected
    rr = dbz_to_rain_rate(mean_pred)
    cb_mask = (rr >= 100.0)
    labeled, num_features = ndimage.label(cb_mask)
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) >= 9:
            r, c = ndimage.center_of_mass(mean_pred * (labeled == i))
            lat, lon = get_latlon(r, c)
            alerts.append({
                "type": "Cloudburst",
                "severity": "Extreme",
                "probability": 0.8,
                "eta_minutes": (lead_idx + 1) * 10,
                "lat": lat, "lon": lon,
                "peak_dbz": np.max(mean_pred[labeled == i]),
                "advisory": "Take immediate shelter. Risk of flash floods."
            })

    # Hail: dBZ >= 50
    hail_mask = (mean_pred >= 50.0)
    labeled_hail, num_hail = ndimage.label(hail_mask)
    for i in range(1, num_hail + 1):
        r, c = ndimage.center_of_mass(mean_pred * (labeled_hail == i))
        lat, lon = get_latlon(r, c)
        alerts.append({
            "type": "Hail",
            "severity": "High",
            "probability": 0.7,
            "eta_minutes": (lead_idx + 1) * 10,
            "lat": lat, "lon": lon,
            "peak_dbz": np.max(mean_pred[labeled_hail == i]),
            "advisory": "Large hail possible. Stay indoors."
        })

    # Thunderstorm: dBZ >= 35
    ts_mask = (mean_pred >= 35.0)
    labeled_ts, num_ts = ndimage.label(ts_mask)
    for i in range(1, num_ts + 1):
        # Prevent duplicate overlapping alerts with higher severity
        if np.max(mean_pred[labeled_ts == i]) >= 50.0: continue
        
        r, c = ndimage.center_of_mass(mean_pred * (labeled_ts == i))
        lat, lon = get_latlon(r, c)
        alerts.append({
            "type": "Thunderstorm",
            "severity": "Moderate",
            "probability": 0.6,
            "eta_minutes": (lead_idx + 1) * 10,
            "lat": lat, "lon": lon,
            "peak_dbz": np.max(mean_pred[labeled_ts == i]),
            "advisory": "Lightning and gusty winds possible."
        })
        
    return alerts
