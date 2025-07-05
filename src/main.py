#!/usr/bin/env python3
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import argparse
import timm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from utils.misc import (
    set_seed, load_config, create_output_dir, save_checkpoint, 
    get_transforms, calculate_accuracy, get_model_summary,
    create_data_loaders, log_metrics, AverageMeter, EarlyStopping
)
from utils.loss import get_loss_function
from model.homa import HOMA

def get_model(config):
    """Get model based on configuration"""
    model_name = config['model']['name'].lower()
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    
    if model_name == 'homa':
        # Use custom HOMA model
        homa_config = config['model']['homa']
        model = HOMA(
            in_ch=homa_config['in_ch'],
            out_dim=homa_config['out_dim'],
            rank=homa_config['rank'],
            orders=tuple(homa_config['orders'])
        )
        # Add final classification layer
        model.classifier = nn.Linear(homa_config['out_dim'], num_classes)
        
        # Modify forward method to include classifier
        original_forward = model.forward
        def new_forward(x):
            features = original_forward(x)
            return model.classifier(features)
        model.forward = new_forward
        
    else:
        # Use timm model
        if model_name in timm.list_models():
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    return model

def get_optimizer(model, config):
    """Get optimizer based on configuration"""
    opt_name = config['optimizer']['name'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if opt_name == 'sgd':
        params = config['optimizer'].get('sgd', {})
        momentum = params.get('momentum', 0.9)
        nesterov = params.get('nesterov', True)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif opt_name == 'adam':
        params = config['optimizer'].get('adam', {})
        betas = params.get('betas', [0.9, 0.999])
        eps = params.get('eps', 1e-08)
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    elif opt_name == 'adamw':
        params = config['optimizer'].get('adamw', {})
        betas = params.get('betas', [0.9, 0.999])
        eps = params.get('eps', 1e-08)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer

def get_scheduler(optimizer, config):
    """Get learning rate scheduler based on configuration"""
    scheduler_name = config['training']['scheduler'].lower()
    
    if scheduler_name == 'step':
        params = config['training']['scheduler_params']['step']
        scheduler = StepLR(
            optimizer,
            step_size=params['step_size'],
            gamma=params['gamma']
        )
    elif scheduler_name == 'cosine':
        params = config['training']['scheduler_params']['cosine']
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=params['T_max'],
            eta_min=params['eta_min']
        )
    elif scheduler_name == 'plateau':
        params = config['training']['scheduler_params']['plateau']
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=params['mode'],
            factor=params['factor'],
            patience=params['patience'],
            threshold=params['threshold']
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, config):
    """Train for one epoch"""
    model.train()
    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if config['system']['mixed_precision']:
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Calculate accuracy
        acc = calculate_accuracy(output, target)
        
        # Update meters
        train_loss.update(loss.item(), data.size(0))
        train_acc.update(acc, data.size(0))
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{train_loss.avg:.4f}',
            'Acc': f'{train_acc.avg:.2f}%'
        })
        
        # Log interval
        if batch_idx % config['output']['log_interval'] == 0:
            log_metrics({
                'train_loss': train_loss.avg,
                'train_acc': train_acc.avg,
                'lr': optimizer.param_groups[0]['lr']
            }, os.path.join(config['_output_dir'], 'train.log'))
    
    return train_loss.avg, train_acc.avg

def validate_epoch(model, val_loader, criterion, device, config):
    """Validate for one epoch"""
    model.eval()
    
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    
    progress_bar = tqdm(val_loader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            if config['system']['mixed_precision']:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            # Calculate accuracy
            acc = calculate_accuracy(output, target)
            
            # Update meters
            val_loss.update(loss.item(), data.size(0))
            val_acc.update(acc, data.size(0))
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{val_loss.avg:.4f}',
                'Acc': f'{val_acc.avg:.2f}%'
            })
    
    return val_loss.avg, val_acc.avg

def main():
    parser = argparse.ArgumentParser(description='Train model with HOMA or timm models')
    parser.add_argument('--config', type=str, default='src/config.yaml', help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['system']['seed'])
    
    # Create output directory
    output_dir = create_output_dir(config['output']['save_dir'], config['output']['exp_name'])
    config['_output_dir'] = output_dir
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        import yaml
        yaml.dump(config, f, default_flow_style=False)
    
    # Setup device
    if config['system']['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['system']['device'])
    
    print(f"Using device: {device}")
    
    # Get dataset info for num_classes
    dataset_name = config['dataset']['name'].lower()
    if dataset_name == 'cub200':
        config['model']['num_classes'] = 200
    elif dataset_name == 'soylocal':
        config['model']['num_classes'] = 200  # Assuming 200 classes for soylocal
    
    # Get transforms
    train_transform, test_transform = get_transforms(
        config['model']['name'],
        config['model']['pretrained'],
        config['model']['input_size']
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        dataset_name,
        config['dataset']['root'],
        config['dataset']['batch_size'],
        config['dataset']['num_workers'],
        train_transform,
        test_transform
    )
    
    # Get model
    model = get_model(config)
    model = model.to(device)
    
    # Print model summary
    print(get_model_summary(model, (config['model']['input_size'], config['model']['input_size'])))
    
    # Get loss function
    criterion = get_loss_function(
        config['loss']['name'],
        config['model']['num_classes'],
        **config['loss'].get(config['loss']['name'], {})
    )
    
    # Get optimizer
    optimizer = get_optimizer(model, config)
    
    # Get scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Mixed precision scaler
    scaler = GradScaler() if config['system']['mixed_precision'] else None
    
    # Early stopping
    early_stopping = None
    if config['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['early_stopping']['patience'],
            min_delta=config['early_stopping']['min_delta'],
            restore_best_weights=config['early_stopping']['restore_best_weights']
        )
    
    # Training variables
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        print(f"Resumed from epoch {start_epoch}, best accuracy: {best_acc:.2f}%")
    
    # Training loop
    epochs = config['training']['epochs']
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, config
        )
        
        # Validate
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device, config
        )
        
        # Step scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()
        
        # Log metrics
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        log_metrics(metrics, os.path.join(output_dir, 'metrics.log'))
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            }, os.path.join(output_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        # Save checkpoint if enabled
        if config['output']['save_checkpoint'] and not config['output']['save_best_only']:
            save_checkpoint(
                model, optimizer, epoch, best_acc,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
