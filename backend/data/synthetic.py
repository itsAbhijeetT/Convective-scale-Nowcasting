import numpy as np
import os

class SyntheticStormGenerator:
    def __init__(self, grid_size=(256, 256), resolution_km=1.0, dt_min=10):
        self.grid_size = grid_size
        self.resolution_km = resolution_km
        self.dt_min = dt_min
        
    def generate_sequence(self, num_frames=48, num_cells=5, advection_vector=(2.0, 1.0)):
        """
        Generate a synthetic sequence of storms moving and evolving.
        num_frames: 48 frames (8 hours of 10-min steps)
        num_cells: Number of main storm cells
        advection_vector: (dx, dy) pixels per frame
        """
        sequence = np.zeros((num_frames, self.grid_size[0], self.grid_size[1]))
        
        # Initialize cell properties
        # (y, x, max_intensity_mmhr, radius_km, growth_rate)
        cells = []
        for _ in range(num_cells):
            y = np.random.uniform(0, self.grid_size[0])
            x = np.random.uniform(0, self.grid_size[1] // 2) # Start mostly on the left
            intensity = np.random.uniform(10, 50)
            radius = np.random.uniform(5, 20)
            growth = np.random.uniform(-0.5, 2.0)
            cells.append([y, x, intensity, radius, growth])
            
        # Add one extreme cloudburst cell
        cb_y = np.random.uniform(0, self.grid_size[0])
        cb_x = np.random.uniform(0, self.grid_size[1] // 2)
        cells.append([cb_y, cb_x, 150.0, 4.0, 5.0]) # Very high intensity, small radius, fast growth
        
        y_coords, x_coords = np.mgrid[0:self.grid_size[0], 0:self.grid_size[1]]
        
        for t in range(num_frames):
            frame = np.zeros(self.grid_size)
            for i, cell in enumerate(cells):
                cy, cx, intensity, radius, growth = cell
                
                # Render Gaussian cell
                dist_sq = (y_coords - cy)**2 + (x_coords - cx)**2
                cell_rain = intensity * np.exp(-dist_sq / (2 * (radius / self.resolution_km)**2))
                frame = np.maximum(frame, cell_rain)
                
                # Update cell for next frame
                cells[i][0] += advection_vector[0] + np.random.normal(0, 0.5)
                cells[i][1] += advection_vector[1] + np.random.normal(0, 0.5)
                
                # Evolve intensity and size
                cells[i][2] += growth
                cells[i][3] += growth * 0.1
                
                # Add some lifecycle (growth then decay)
                if cells[i][2] > 180: # Cap intensity
                    cells[i][4] = -abs(growth) # Force decay
                if cells[i][2] < 5:
                    cells[i][2] = 0 # Dissipate
                
            # Add some per-pixel noise
            noise = np.random.normal(0, 1.0, self.grid_size)
            frame = frame + noise
            frame = np.clip(frame, 0, None)
            
            sequence[t] = frame
            
        return sequence

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    gen = SyntheticStormGenerator()
    out_dir = os.path.join(os.path.dirname(__file__), "synthetic_dataset")
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate 5 sequences for training/testing
    for seq_idx in range(5):
        seq = gen.generate_sequence()
        # Convert to dBZ for storage/consistency with radar
        from loaders import rain_rate_to_dbz
        seq_dbz = rain_rate_to_dbz(seq)
        np.save(os.path.join(out_dir, f"seq_{seq_idx:03d}.npy"), seq_dbz)
        print(f"Generated sequence {seq_idx}")
    print("Done!")
