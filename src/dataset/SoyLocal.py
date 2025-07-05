import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from typing import Tuple, Optional, Callable
import zipfile
import urllib.request

class SoyLocalDataset(Dataset):
    """
    SoyLocal Dataset for PyTorch, similar to MNIST
    
    Args:
        root (str): Root directory where dataset exists or will be downloaded
        split (str): One of 'train', 'val', 'test'
        transform (callable, optional): Optional transform to be applied on a sample
        target_transform (callable, optional): Optional transform to be applied on the target
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    
    url = "https://huggingface.co/datasets/hibana2077/Ultra-FGVC/resolve/main/SoyLocal/SoyLocal.zip?download=true"
    
    def __init__(
        self, 
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
        
        # Load annotations
        self.samples = []
        self.targets = []
        self._load_annotations()
    
    def _check_exists(self) -> bool:
        """Check if the dataset exists"""
        return os.path.exists(os.path.join(self.root, 'soybean200'))
    
    def download(self) -> None:
        """Download the dataset if it doesn't exist"""
        if self._check_exists():
            return
        
        os.makedirs(self.root, exist_ok=True)
        
        # Download zip file
        zip_path = os.path.join(self.root, 'SoyLocal.zip')
        print(f'Downloading {self.url}...')
        urllib.request.urlretrieve(self.url, zip_path)
        
        # Extract zip file
        print('Extracting...')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Remove zip file
        os.remove(zip_path)
        print('Download and extraction completed')
    
    def _load_annotations(self) -> None:
        """Load annotations from txt files"""
        anno_path = os.path.join(self.root, 'soybean200', 'anno', f'{self.split}.txt')
        
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f'Annotation file not found: {anno_path}')
        
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                filename, label = line.split()
                self.samples.append(filename)
                # Convert label from 1-indexed to 0-indexed
                self.targets.append(int(label) - 1)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        filename = self.samples[idx]
        target = self.targets[idx]
        
        # Load image
        img_path = os.path.join(self.root, 'soybean200', 'images', filename)
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target
    
    @property
    def classes(self):
        """Get unique class labels (0-indexed)"""
        return sorted(list(set(self.targets)))
    
    @property
    def num_classes(self):
        """Get number of classes"""
        return len(self.classes)


# Example usage and helper functions
def get_soylocal_transforms(split='train'):
    """Get default transforms for SoyLocal dataset"""
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_soylocal_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    download: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = SoyLocalDataset(
        root=root,
        split='train',
        transform=get_soylocal_transforms('train'),
        download=download
    )
    
    val_dataset = SoyLocalDataset(
        root=root,
        split='val',
        transform=get_soylocal_transforms('val')
    )
    
    test_dataset = SoyLocalDataset(
        root=root,
        split='test',
        transform=get_soylocal_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    
    return train_loader, val_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Example 1: Basic usage
    dataset = SoyLocalDataset(
        root='./data',
        split='train',
        transform=get_soylocal_transforms('train'),
        download=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Classes: {dataset.classes}")
    
    # Get a sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
    
    # Example 2: Create dataloaders
    train_loader, val_loader, test_loader = create_soylocal_dataloaders(
        root='./data',
        batch_size=32,
        download=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Example 3: Iterate through a batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}: images shape {images.shape}, targets shape {targets.shape}")
        if batch_idx == 0:  # Only show first batch
            break