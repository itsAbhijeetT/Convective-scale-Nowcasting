import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_sequence(frames=60, size=64):
    seq = np.zeros((frames, size, size), dtype=np.float32)
    
    # 2-5 cells
    num_cells = np.random.randint(2, 6)
    cells = []
    for _ in range(num_cells):
        r, c = np.random.randint(10, 54, size=2)
        dr, dc = np.random.uniform(-1, 1, size=2)
        peak = np.random.uniform(40, 60)
        width = np.random.uniform(2, 5)
        cells.append({'r':r, 'c':c, 'dr':dr, 'dc':dc, 'peak':peak, 'width':width})
        
    for t in range(frames):
        grid = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        for cell in cells:
            cell['r'] += cell['dr']
            cell['c'] += cell['dc']
            
            # Change intensity slightly
            cell['peak'] += np.random.uniform(-1, 1)
            cell['peak'] = np.clip(cell['peak'], 0, 70)
            
            dist_sq = (x - cell['c'])**2 + (y - cell['r'])**2
            grid += cell['peak'] * np.exp(-dist_sq / (2 * cell['width']**2))
            
        grid += np.random.normal(0, 2, size=(size, size))
        seq[t] = np.clip(grid, 0, 70)
        
    return seq

def generate_cloudburst(frames=60, size=64):
    seq = np.zeros((frames, size, size), dtype=np.float32)
    # create intense cell
    cell = {'r':32, 'c':32, 'dr':0.2, 'dc':0.1, 'peak':30, 'width':2}
    
    for t in range(frames):
        grid = np.zeros((size, size), dtype=np.float32)
        y, x = np.ogrid[:size, :size]
        
        cell['r'] += cell['dr']
        cell['c'] += cell['dc']
        if 20 <= t <= 30:
            cell['peak'] += 3  # rapid growth
            cell['width'] += 0.1
        elif t > 40:
            cell['peak'] -= 1 # decay
        
        cell['peak'] = np.clip(cell['peak'], 0, 70)
        
        dist_sq = (x - cell['c'])**2 + (y - cell['r'])**2
        grid += cell['peak'] * np.exp(-dist_sq / (2 * cell['width']**2))
        
        grid += np.random.normal(0, 1, size=(size, size))
        seq[t] = np.clip(grid, 0, 70)
        
    return seq

def create_synthetic_data(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    if os.path.exists(os.path.join(data_dir, "demo_event.npy")):
        print("Data already exists. Skipping generation.")
        return
        
    print("Generating 150 synthetic sequences...")
    sequences = []
    for i in range(150):
        if i % 10 == 0:
            seq = generate_cloudburst()
        else:
            seq = generate_sequence()
        sequences.append(seq)
        
    np.save(os.path.join(data_dir, "synthetic_data.npy"), np.array(sequences))
    
    print("Generating demo event...")
    demo = generate_cloudburst()
    np.save(os.path.join(data_dir, "demo_event.npy"), demo)
    
    # Save GIF
    fig = plt.figure()
    im = plt.imshow(demo[0], vmin=0, vmax=70, cmap='jet')
    def update(i):
        im.set_data(demo[i])
        return im,
    ani = animation.FuncAnimation(fig, update, frames=60, interval=100)
    ani.save(os.path.join(data_dir, "sample.gif"), writer='pillow')
    plt.close()
    print("Done generating data.")

if __name__ == "__main__":
    create_synthetic_data()
