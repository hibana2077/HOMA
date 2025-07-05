# HOMA Training Framework

This is a complete training framework for the HOMA (Higher-Order Moment Aggregation) model and other timm models.

## Features

- Support for custom HOMA model and timm models
- Automatic mixed precision training
- Configurable loss functions (CrossEntropy, Focal Loss, Label Smoothing)
- Multiple optimizers (SGD, Adam, AdamW)
- Learning rate schedulers (StepLR, CosineAnnealingLR, ReduceLROnPlateau)
- Early stopping
- Comprehensive logging and metrics tracking
- Checkpoint saving and resuming
- Support for CUB-200 and SoyLocal datasets

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python src/main.py --config src/config.yaml
```

### Resume Training

```bash
python src/main.py --config src/config.yaml --resume results/experiment_name/best_model.pth
```

### Using the Runner Script

```bash
python run_training.py --config src/config.yaml
```

## Configuration

Edit `src/config.yaml` to customize training parameters:

### Model Configuration
- `model.name`: Model name ('homa' for custom model, or any timm model name)
- `model.pretrained`: Whether to use pretrained weights
- `model.num_classes`: Number of classes (auto-detected for supported datasets)

### Dataset Configuration
- `dataset.name`: Dataset name ('cub200' or 'soylocal')
- `dataset.root`: Root directory for dataset
- `dataset.batch_size`: Batch size for training
- `dataset.num_workers`: Number of data loading workers

### Training Configuration
- `training.epochs`: Number of training epochs
- `training.learning_rate`: Initial learning rate
- `training.scheduler`: Learning rate scheduler type

### Loss and Optimizer
- `loss.name`: Loss function name
- `optimizer.name`: Optimizer name

## Supported Models

### Custom Models
- `homa`: The HOMA model from src/model/homa.py

### Timm Models
Any model from the timm library, including:
- `resnet50`
- `efficientnet_b0`
- `vit_base_patch16_224`
- `swin_base_patch4_window7_224`
- And many more...

## Supported Datasets

- **CUB-200**: Caltech-UCSD Birds-200-2011 dataset
- **SoyLocal**: Custom soybean dataset

## Output

Training results are saved to:
- `results/experiment_name_timestamp/`
  - `config.yaml`: Configuration used
  - `best_model.pth`: Best model checkpoint
  - `metrics.log`: Training metrics
  - `train.log`: Detailed training logs

## Model Architecture

The HOMA model uses higher-order moment aggregation:
- 2nd order: Covariance features
- 3rd order: Third-order approximation with random tensors
- 4th order: Fourth-order cumulant features

## Examples

### Train HOMA on CUB-200
```yaml
# config.yaml
model:
  name: 'homa'
  pretrained: false
  
dataset:
  name: 'cub200'
  root: './data'
```

### Train ResNet50 on SoyLocal
```yaml
# config.yaml
model:
  name: 'resnet50'
  pretrained: true
  
dataset:
  name: 'soylocal'
  root: './data'
```

## Monitoring

The framework provides:
- Real-time progress bars with tqdm
- Training and validation loss/accuracy tracking
- Learning rate monitoring
- Best model saving based on validation accuracy
- Optional early stopping

## Advanced Features

### Mixed Precision Training
Enable for faster training on modern GPUs:
```yaml
system:
  mixed_precision: true
```

### Early Stopping
Prevent overfitting:
```yaml
early_stopping:
  enabled: true
  patience: 15
  min_delta: 0.001
```

### Checkpoint Saving
Save intermediate checkpoints:
```yaml
output:
  save_checkpoint: true
  save_best_only: false
```
