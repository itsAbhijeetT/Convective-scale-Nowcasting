import torch
import torch.nn as nn
import numpy as np

class DummyConvLSTM(nn.Module):
    """
    A simplified ConvLSTM / U-Net equivalent for demonstration.
    In a real scenario, this would be a full Spatiotemporal model.
    """
    def __init__(self, in_channels=6, out_channels=12):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (B, C_in, H, W)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # return shape: (B, C_out, H, W)
        return torch.relu(x) # Rain/dBZ is positive

def nowcast_convlstm(recent_frames, lead_steps=12, model_path=None):
    """
    Predict next lead_steps frames using the ML model.
    recent_frames: (T, H, W) where T >= 6.
    Returns: (lead_steps, H, W)
    """
    T = recent_frames.shape[0]
    if T < 6:
        # Pad if not enough frames
        pad = np.zeros((6 - T, *recent_frames.shape[1:]))
        inputs = np.concatenate([pad, recent_frames], axis=0)
    else:
        inputs = recent_frames[-6:]
        
    # Convert to tensor: (1, 6, H, W)
    x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    
    model = DummyConvLSTM(in_channels=6, out_channels=lead_steps)
    if model_path:
        pass # Load weights here in production
        
    model.eval()
    with torch.no_grad():
        preds = model(x)
        
    preds_np = preds.squeeze(0).numpy() # (lead_steps, H, W)
    
    # If lead_steps requested > model output, pad with last frame
    if lead_steps > preds_np.shape[0]:
        diff = lead_steps - preds_np.shape[0]
        pad = np.repeat(preds_np[-1:], diff, axis=0)
        preds_np = np.concatenate([preds_np, pad], axis=0)
        
    return preds_np
