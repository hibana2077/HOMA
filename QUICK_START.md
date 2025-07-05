# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Option 1: Using the wrapper script (Recommended)
```bash
# From project root directory
python train.py
```

### Option 2: Direct execution
```bash
# From project root directory
python src/main.py --config config.yaml
```

### Option 3: From src directory
```bash
# From src directory
cd src
python main.py --config config.yaml
```

## Configuration

Edit `config.yaml` to customize training:

- **Model**: Change `model.name` to use different models ('homa', 'resnet50', 'efficientnet_b0', etc.)
- **Dataset**: Change `dataset.name` to use different datasets ('cub200', 'soylocal')
- **Training**: Adjust epochs, learning rate, batch size, etc.

## Common Issues

1. **Import errors**: Make sure you're running from the project root directory
2. **CUDA out of memory**: Reduce batch size in config.yaml
3. **Dataset not found**: The datasets will be automatically downloaded on first run

## Examples

```bash
# Train HOMA model on CUB-200
python train.py --config config.yaml

# Train ResNet50 (modify config.yaml first)
python train.py --config config.yaml

# Resume training from checkpoint
python train.py --config config.yaml --resume results/experiment_name/best_model.pth
```

## Output

Results are saved to `results/experiment_name_timestamp/`:
- `best_model.pth`: Best model checkpoint
- `config.yaml`: Configuration used
- `metrics.log`: Training metrics
- `train.log`: Detailed training logs
