import os
import numpy as np

# Fallback if torch is missing
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class DummyConvLSTM(nn.Module if HAS_TORCH else object):
    def __init__(self):
        super().__init__()
        if HAS_TORCH:
            self.conv = nn.Conv2d(6, 1, kernel_size=3, padding=1)
            
    def forward(self, x):
        # x is (B, 6, H, W)
        return torch.relu(self.conv(x))

def get_convlstm_model():
    if not HAS_TORCH:
        print("Warning: PyTorch not found. Falling back to optical flow.")
        return None
    model = DummyConvLSTM()
    model_path = os.path.join("models", "convlstm.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model

def convlstm_predict(model, past_frames, lead_steps=36):
    """
    past_frames: (6, H, W)
    """
    if not HAS_TORCH or model is None:
        from models.optical_flow import persistence_baseline
        return persistence_baseline(past_frames, lead_steps)
        
    H, W = past_frames.shape[1], past_frames.shape[2]
    preds = []
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor(past_frames / 70.0, dtype=torch.float32).unsqueeze(0) # (1, 6, H, W)
        for _ in range(lead_steps):
            y = model(x) # (1, 1, H, W)
            pred = y.squeeze().numpy() * 70.0
            preds.append(pred)
            # Roll out: drop oldest, add newest
            x = torch.cat((x[:, 1:], y), dim=1)
            
    return np.array(preds)

def train_convlstm():
    if not HAS_TORCH:
        return
        
    data_path = os.path.join("data", "synthetic_data.npy")
    if not os.path.exists(data_path):
        return
        
    data = np.load(data_path) # (150, 60, 64, 64)
    model = DummyConvLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print("Training ConvLSTM for a few epochs...")
    for epoch in range(2):
        total_loss = 0
        for seq in data[:20]: # train on small subset for speed
            for t in range(0, 10):
                x = torch.tensor(seq[t:t+6] / 70.0, dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(seq[t+6] / 70.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss}")
        
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", "convlstm.pt"))
    print("Saved models/convlstm.pt")

if __name__ == "__main__":
    train_convlstm()
