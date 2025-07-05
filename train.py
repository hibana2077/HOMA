#!/usr/bin/env python3
"""
Simple training script that can be run from project root
Usage: python train.py [--config path/to/config.yaml] [--resume path/to/checkpoint.pth]
"""
import os
import sys
import subprocess

def main():
    """Run training using the main script"""
    # Get the directory containing this script (project root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Path to the main training script
    main_script = os.path.join(project_root, 'src', 'main.py')
    
    # Forward all arguments to the main script
    cmd = [sys.executable, main_script] + sys.argv[1:]
    
    print(f"Project root: {project_root}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        # Change to project root directory
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        # Run the training script
        result = subprocess.run(cmd, check=True)
        print("\nTraining completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with error code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return 1
    finally:
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass

if __name__ == "__main__":
    sys.exit(main())
