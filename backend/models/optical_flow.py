import cv2
import numpy as np

def compute_optical_flow(frame1, frame2):
    """
    Compute Farneback optical flow between two consecutive frames.
    Inputs are typically 2D numpy arrays representing dBZ or rain rate.
    Returns: flow (H, W, 2)
    """
    # Normalize to 0-255 for cv2 optical flow
    f1_min, f1_max = frame1.min(), frame1.max()
    if f1_max > f1_min:
        f1 = ((frame1 - f1_min) / (f1_max - f1_min) * 255).astype(np.uint8)
    else:
        f1 = np.zeros_like(frame1, dtype=np.uint8)
        
    f2_min, f2_max = frame2.min(), frame2.max()
    if f2_max > f2_min:
        f2 = ((frame2 - f2_min) / (f2_max - f2_min) * 255).astype(np.uint8)
    else:
        f2 = np.zeros_like(frame2, dtype=np.uint8)

    flow = cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def extrapolate_semi_lagrangian(frame, flow, num_steps=36):
    """
    Extrapolate the given frame for num_steps using semi-Lagrangian advection.
    flow: (H, W, 2) array containing dx and dy.
    Returns: sequence of shape (num_steps, H, W)
    """
    h, w = frame.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    
    extrapolated = np.zeros((num_steps, h, w), dtype=np.float32)
    current_frame = frame.copy()
    
    for i in range(num_steps):
        # Semi-Lagrangian backward trajectory
        # For forward prediction by dt, we find where the pixel comes from:
        # P(x, y, t+dt) = P(x - dx, y - dy, t)
        dx = flow[..., 0]
        dy = flow[..., 1]
        
        map_x = x - dx
        map_y = y - dy
        
        # Remap
        next_frame = cv2.remap(current_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        extrapolated[i] = next_frame
        current_frame = next_frame # update for next step
        
    return extrapolated

def nowcast_optical_flow(recent_frames, lead_steps=36):
    """
    Main entry point for optical flow nowcast.
    recent_frames: (T, H, W) where T >= 2.
    """
    # Use the last two frames to compute flow
    flow = compute_optical_flow(recent_frames[-2], recent_frames[-1])
    # Extrapolate from the most recent frame
    return extrapolate_semi_lagrangian(recent_frames[-1], flow, num_steps=lead_steps)
