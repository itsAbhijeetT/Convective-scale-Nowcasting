import cv2
import numpy as np

def optical_flow_extrapolation(past_frames, lead_steps=36):
    """
    past_frames: (T, H, W)
    """
    if len(past_frames) < 2:
        return np.repeat(past_frames[-1:], lead_steps, axis=0)

    # Use last two frames for flow
    prev_frame = past_frames[-2]
    curr_frame = past_frames[-1]

    # Normalize for flow calculation
    prev_norm = np.uint8(np.clip(prev_frame / 70.0 * 255, 0, 255))
    curr_norm = np.uint8(np.clip(curr_frame / 70.0 * 255, 0, 255))

    flow = cv2.calcOpticalFlowFarneback(prev_norm, curr_norm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    H, W = curr_frame.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    
    forecasts = []
    current_image = curr_frame.copy()
    
    for i in range(lead_steps):
        map_x = x - flow[..., 0] * (i + 1)
        map_y = y - flow[..., 1] * (i + 1)
        
        extrapolated = cv2.remap(curr_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        forecasts.append(extrapolated)
        
    return np.array(forecasts)

def persistence_baseline(past_frames, lead_steps=36):
    return np.repeat(past_frames[-1:], lead_steps, axis=0)
