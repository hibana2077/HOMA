import os
import sys
import torch
import random
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader
import timm
import json
from datetime import datetime

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def create_output_dir(base_dir: str, exp_name: str) -> str:
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{exp_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def save_checkpoint(model, optimizer, epoch: int, best_acc: float, save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(model, optimizer, checkpoint_path: str) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    return epoch, best_acc

def get_transforms(model_name: str, pretrained: bool = True, input_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for training and testing
    
    Args:
        model_name (str): Name of the model
        pretrained (bool): Whether to use pretrained transforms
        input_size (int): Input image size
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: Train and test transforms
    """
    if pretrained and model_name in timm.list_models():
        try:
            # Try to get timm model's transform
            model_config = timm.get_model_config(model_name)
            data_config = timm.data.resolve_model_data_config(model_config)
            
            train_transform = timm.data.create_transform(
                **data_config,
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic'
            )
            
            test_transform = timm.data.create_transform(
                **data_config,
                is_training=False,
                interpolation='bicubic'
            )
            
            return train_transform, test_transform
        except:
            pass
    
    # Default transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    return train_transform, test_transform

def calculate_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate accuracy from outputs and targets"""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100 * correct / total

def get_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...]) -> str:
    """Get model summary information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
- Total parameters: {total_params:,}
- Trainable parameters: {trainable_params:,}
- Non-trainable parameters: {total_params - trainable_params:,}
- Input size: {input_size}
"""
    return summary

def create_data_loaders(dataset_name: str, data_root: str, batch_size: int, 
                       num_workers: int, train_transform: transforms.Compose,
                       test_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders based on dataset name
    
    Args:
        dataset_name (str): Name of the dataset
        data_root (str): Root directory of the dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers
        train_transform (transforms.Compose): Training transforms
        test_transform (transforms.Compose): Testing transforms
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and test data loaders
    """
    dataset_name = dataset_name.lower()
    
    # Setup import paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    if dataset_name == 'cub200':
        from dataset.CUB200 import CUB200Dataset
        train_dataset = CUB200Dataset(root=data_root, train=True, transform=train_transform, download=True)
        test_dataset = CUB200Dataset(root=data_root, train=False, transform=test_transform, download=True)
    elif dataset_name == 'soylocal':
        from dataset.SoyLocal import SoyLocalDataset
        train_dataset = SoyLocalDataset(root=data_root, split='train', transform=train_transform, download=True)
        test_dataset = SoyLocalDataset(root=data_root, split='test', transform=test_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader

def log_metrics(metrics: Dict[str, float], log_file: str):
    """Log metrics to file"""
    with open(log_file, 'a') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        f.write(log_entry + "\n")

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: torch.nn.Module):
        """Save model weights"""
        self.best_weights = model.state_dict().copy()
