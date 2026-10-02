import numpy as np
from models.optical_flow import optical_flow_extrapolation
from models.convlstm import get_convlstm_model, convlstm_predict

def blend_nowcasts(past_frames, lead_steps=36, members=5):
    flow_pred = optical_flow_extrapolation(past_frames, lead_steps)
    
    model = get_convlstm_model()
    lstm_pred = convlstm_predict(model, past_frames, lead_steps)
    
    # Weights decay linearly
    w = np.linspace(1.0, 0.3, lead_steps).reshape(-1, 1, 1)
    
    base_pred = w * flow_pred + (1 - w) * lstm_pred
    
    ensemble = []
    for _ in range(members):
        # Add small noise perturbation
        noise = np.random.normal(0, 2, size=base_pred.shape)
        perturbed = np.clip(base_pred + noise, 0, 70)
        ensemble.append(perturbed)
        
    return np.array(ensemble) # (5, 36, H, W)
