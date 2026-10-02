import numpy as np
from .optical_flow import nowcast_optical_flow
from .convlstm import nowcast_convlstm

def blend_nowcasts(recent_frames, lead_steps=36):
    """
    Blend Optical Flow (good for short term, e.g., 0-2 hours)
    with ConvLSTM (good for capturing non-linear growth/decay, 1-6 hours).
    
    Returns: ensemble of predictions (N, lead_steps, H, W)
    """
    # 1. Get OF prediction
    of_preds = nowcast_optical_flow(recent_frames, lead_steps=lead_steps)
    
    # 2. Get ML prediction (say ML predicts all 36 steps or is padded)
    ml_preds = nowcast_convlstm(recent_frames, lead_steps=lead_steps)
    
    ensemble = []
    
    # Base member: weighted blend where OF weight decays from 1 to 0 over 36 steps
    weights_of = np.linspace(1.0, 0.0, lead_steps).reshape(-1, 1, 1)
    blend_base = weights_of * of_preds + (1 - weights_of) * ml_preds
    ensemble.append(blend_base)
    
    # Perturbations to create an ensemble (e.g. 5 members)
    for i in range(4):
        noise = np.random.normal(0, 2.0, blend_base.shape) # Add some dBZ noise
        perturbed = np.clip(blend_base + noise, 0, None)
        ensemble.append(perturbed)
        
    return np.stack(ensemble, axis=0)
