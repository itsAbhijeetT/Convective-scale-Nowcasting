import cv2
import numpy as np

def detect_thunderstorm(ensemble_preds, lead_time_idx):
    """
    Thunderstorm: dBZ >= 35.
    ensemble_preds: (N, T, H, W)
    Returns probability map (H, W)
    """
    # Slice the ensemble for the given lead time
    preds_t = ensemble_preds[:, lead_time_idx, :, :] # (N, H, W)
    
    # Calculate probability: fraction of members exceeding 35 dBZ
    prob_map = np.mean(preds_t >= 35.0, axis=0)
    return prob_map
