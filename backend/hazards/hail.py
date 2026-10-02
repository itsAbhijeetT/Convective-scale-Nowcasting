import cv2
import numpy as np

def detect_hail(ensemble_preds, lead_time_idx):
    """
    Hail: dBZ >= 50 + rapid growth proxy.
    ensemble_preds: (N, T, H, W)
    """
    preds_t = ensemble_preds[:, lead_time_idx, :, :] # (N, H, W)
    
    # Probability of dBZ >= 50
    prob_map = np.mean(preds_t >= 50.0, axis=0)
    
    # Proxy for vertical growth: check if previous step was much lower
    if lead_time_idx > 0:
        preds_prev = ensemble_preds[:, lead_time_idx - 1, :, :]
        # Growth > 10 dBZ in 10 mins
        growth_prob = np.mean((preds_t - preds_prev) >= 10.0, axis=0)
        prob_map = prob_map * growth_prob
        
    return prob_map
