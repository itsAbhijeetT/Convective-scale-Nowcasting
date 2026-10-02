import cv2
import numpy as np

def dbz_to_rain_rate(dbz):
    z = 10.0 ** (dbz / 10.0)
    return (z / 200.0) ** (1 / 1.6)

def detect_cloudburst(ensemble_preds, lead_time_idx, resolution_km=1.0):
    """
    Cloudburst: rain rate >= 100 mm/hr over a cluster >= 20 km2.
    ensemble_preds: (N, T, H, W)
    """
    preds_t = ensemble_preds[:, lead_time_idx, :, :] # (N, H, W)
    mean_dbz = np.mean(preds_t, axis=0) # (H, W)
    rain_rate = dbz_to_rain_rate(mean_dbz)
    
    # Thresholding
    mask = (rain_rate >= 100.0).astype(np.uint8)
    
    # Find connected components (clusters)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    prob_map = np.zeros_like(mean_dbz)
    
    # Check cluster area >= 20 km2
    area_px_needed = 20.0 / (resolution_km ** 2)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_px_needed:
            # Recompute probability for this valid cluster
            cluster_mask = (labels == i)
            # Probability is fraction of members meeting criteria in this cluster
            # For simplicity, we just set the prob_map to a high value based on mean ensemble rain rate > 100
            prob_map[cluster_mask] = np.mean((dbz_to_rain_rate(preds_t) >= 100.0), axis=0)[cluster_mask]
            
    return prob_map
