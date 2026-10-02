import numpy as np

def calculate_metrics(obs, pred, threshold=35.0):
    """
    Calculate POD, FAR, CSI for a given threshold (e.g., 35 dBZ for thunderstorm).
    obs: (H, W) true observation
    pred: (H, W) prediction
    """
    hits = np.sum((obs >= threshold) & (pred >= threshold))
    misses = np.sum((obs >= threshold) & (pred < threshold))
    false_alarms = np.sum((obs < threshold) & (pred >= threshold))
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0
    csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0.0
    
    return pod, far, csi
    
if __name__ == "__main__":
    print("Evaluating models...")
    # Mock evaluation
    for lead in [30, 60, 120, 240, 360]:
        csi = max(0, 1.0 - (lead / 360.0) * 0.7)
        print(f"Lead time {lead} min -> CSI: {csi:.2f}")
