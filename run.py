import os
import subprocess
import sys

def main():
    print("Step 1: Generating data...")
    from data.synthetic import create_synthetic_data
    create_synthetic_data()
    
    print("Step 2: Training model (if needed)...")
    from models.convlstm import train_convlstm
    if not os.path.exists("models/convlstm.pt"):
        train_convlstm()
        
    print("Step 3: Generating metrics...")
    from eval.metrics import run_evaluation
    run_evaluation()

    print("Step 4: Running Pytest...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/"])

    print("Step 5: Starting Server on http://localhost:8000")
    print("Visit http://localhost:8000/static/index.html to view the dashboard")
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])

if __name__ == "__main__":
    main()
