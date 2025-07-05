#!/usr/bin/env python3
"""
Example training script showing different configurations
"""
import os
import sys
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_homa_config():
    """Create config for HOMA model training"""
    config = {
        'model': {
            'name': 'homa',
            'pretrained': False,
            'num_classes': 200,
            'input_size': 224,
            'homa': {
                'in_ch': 3,
                'out_dim': 2048,
                'rank': 64,
                'orders': [2, 3, 4]
            }
        },
        'dataset': {
            'name': 'cub200',
            'root': './data',
            'batch_size': 32,
            'num_workers': 4,
            'download': True
        },
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'scheduler': 'cosine',
            'scheduler_params': {
                'cosine': {
                    'T_max': 50,
                    'eta_min': 0.00001
                }
            }
        },
        'loss': {
            'name': 'crossentropy'
        },
        'optimizer': {
            'name': 'adamw',
            'adamw': {
                'betas': [0.9, 0.999],
                'eps': 1e-8
            }
        },
        'output': {
            'save_dir': './results',
            'exp_name': 'homa_cub200',
            'save_checkpoint': False,
            'save_best_only': True,
            'log_interval': 10
        },
        'system': {
            'device': 'auto',
            'seed': 42,
            'mixed_precision': True
        },
        'early_stopping': {
            'enabled': False,
            'patience': 15,
            'min_delta': 0.001
        },
        'validation': {
            'enabled': True,
            'interval': 1
        }
    }
    return config

def create_resnet_config():
    """Create config for ResNet50 training"""
    config = {
        'model': {
            'name': 'resnet50',
            'pretrained': True,
            'num_classes': 200,
            'input_size': 224
        },
        'dataset': {
            'name': 'cub200',
            'root': './data',
            'batch_size': 64,
            'num_workers': 4,
            'download': True
        },
        'training': {
            'epochs': 30,
            'learning_rate': 0.01,
            'weight_decay': 0.0001,
            'scheduler': 'step',
            'scheduler_params': {
                'step': {
                    'step_size': 10,
                    'gamma': 0.1
                }
            }
        },
        'loss': {
            'name': 'crossentropy'
        },
        'optimizer': {
            'name': 'sgd',
            'sgd': {
                'momentum': 0.9,
                'nesterov': True
            }
        },
        'output': {
            'save_dir': './results',
            'exp_name': 'resnet50_cub200',
            'save_checkpoint': False,
            'save_best_only': True,
            'log_interval': 10
        },
        'system': {
            'device': 'auto',
            'seed': 42,
            'mixed_precision': True
        },
        'early_stopping': {
            'enabled': True,
            'patience': 10,
            'min_delta': 0.001
        },
        'validation': {
            'enabled': True,
            'interval': 1
        }
    }
    return config

def save_config(config, filename):
    """Save config to YAML file"""
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {filename}")

def main():
    """Main function to create example configs"""
    print("Creating example configurations...")
    
    # Create configs directory
    os.makedirs('configs', exist_ok=True)
    
    # Create HOMA config
    homa_config = create_homa_config()
    save_config(homa_config, 'configs/homa_example.yaml')
    
    # Create ResNet config
    resnet_config = create_resnet_config()
    save_config(resnet_config, 'configs/resnet50_example.yaml')
    
    print("\nExample configurations created!")
    print("To train HOMA model:")
    print("  python src/main.py --config configs/homa_example.yaml")
    print("To train ResNet50 model:")
    print("  python src/main.py --config configs/resnet50_example.yaml")

if __name__ == "__main__":
    main()
