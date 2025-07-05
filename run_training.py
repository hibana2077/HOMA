#!/usr/bin/env python3
"""
Simple training script runner
"""
import subprocess
import sys
import os

def run_training(config_path="src/config.yaml"):
    """Run training with specified config"""
    cmd = [sys.executable, "src/main.py", "--config", config_path]
    
    print(f"Running command: {' '.join(cmd)}")
    print("="*50)
    
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        print("\nTraining completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run training script")
    parser.add_argument("--config", type=str, default="src/config.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    exit_code = run_training(args.config)
    sys.exit(exit_code)
