import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms
import tarfile
import urllib.request
from typing import Optional, Callable, Tuple, Any
from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        return self.update(b * bsize - self.n)


class CUB200Dataset(Dataset):
    """
    CUB-200-2011 Birds Dataset
    
    Args:
        root (str): Root directory where dataset will be stored
        train (bool): If True, creates dataset from training set, otherwise from test set
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it
        download (bool): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again
    """
    
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    folder_name = 'CUB_200_2011'
    
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            self.download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
            
        self.data_dir = os.path.join(self.root, self.folder_name)
        self._load_data()
    
    def _check_exists(self) -> bool:
        """Check if the dataset exists"""
        return os.path.exists(os.path.join(self.root, self.folder_name))
    
    def download(self):
        """Download and extract the dataset"""
        if self._check_exists():
            print('Dataset already exists')
            return
            
        os.makedirs(self.root, exist_ok=True)
        
        # Download the dataset with progress bar
        print('Downloading CUB-200-2011 dataset...')
        filepath = os.path.join(self.root, self.filename)
        
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(self.url, filepath, reporthook=t.update_to)
        
        # Extract the dataset with progress bar
        print('Extracting dataset...')
        with tarfile.open(filepath, 'r:gz') as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, path=self.root)
                    pbar.update(1)
        
        # Remove the tar file
        os.remove(filepath)
        print('Dataset downloaded and extracted successfully!')
    
    def _load_data(self):
        """Load the dataset metadata"""
        # Load image paths and labels
        images_file = os.path.join(self.data_dir, 'images.txt')
        labels_file = os.path.join(self.data_dir, 'image_class_labels.txt')
        train_test_split_file = os.path.join(self.data_dir, 'train_test_split.txt')
        classes_file = os.path.join(self.data_dir, 'classes.txt')
        
        # Read images
        with open(images_file, 'r') as f:
            images_data = [line.strip().split() for line in f.readlines()]
        
        # Read labels
        with open(labels_file, 'r') as f:
            labels_data = [line.strip().split() for line in f.readlines()]
        
        # Read train/test split
        with open(train_test_split_file, 'r') as f:
            split_data = [line.strip().split() for line in f.readlines()]
        
        # Read class names
        with open(classes_file, 'r') as f:
            classes_data = [line.strip().split(maxsplit=1) for line in f.readlines()]
        
        # Create class mapping (remove numbering from class names)
        self.classes = [class_name.split('.', 1)[1] if '.' in class_name else class_name 
                       for _, class_name in classes_data]
        self.class_to_idx = {class_name.split('.', 1)[1] if '.' in class_name else class_name: int(class_id) - 1 
                            for class_id, class_name in classes_data}
        
        # Filter data based on train/test split
        self.data = []
        for i, (img_id, img_path) in enumerate(images_data):
            _, label = labels_data[i]
            _, is_train = split_data[i]
            
            if (self.train and is_train == '1') or (not self.train and is_train == '0'):
                full_img_path = os.path.join(self.data_dir, 'images', img_path)
                self.data.append((full_img_path, int(label) - 1))  # Convert to 0-based indexing
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Args:
            idx (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img_path, target = self.data[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform is not None:
            image = self.transform(image)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return image, target
    
    def get_class_name(self, idx: int) -> str:
        """Get class name by index"""
        return self.classes[idx]
    
    def get_class_stats(self):
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.data),
            'num_classes': len(self.classes),
            'samples_per_class': {}
        }
        
        # Count samples per class
        for _, label in self.data:
            class_name = self.classes[label]
            stats['samples_per_class'][class_name] = stats['samples_per_class'].get(class_name, 0) + 1
            
        return stats


# Example usage and data loader creation
def create_cub_dataloaders(root: str, batch_size: int = 32, num_workers: int = 4):
    """
    Create train and test dataloaders for CUB-200-2011 dataset
    
    Args:
        root (str): Root directory for dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CUB200Dataset(
        root=root,
        train=True,
        transform=train_transform,
        download=True
    )
    
    test_dataset = CUB200Dataset(
        root=root,
        train=False,
        transform=test_transform,
        download=False  # Already downloaded
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Create dataset
    dataset_root = "./data"
    
    # Create train and test loaders
    train_loader, test_loader = create_cub_dataloaders(dataset_root, batch_size=32)
    
    print(f"Number of classes: {len(train_loader.dataset.classes)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Show some example class names (cleaned)
    print(f"Example classes: {train_loader.dataset.classes[:5]}")
    
    # Show dataset statistics
    train_stats = train_loader.dataset.get_class_stats()
    print(f"Average samples per class (train): {train_stats['total_samples'] / train_stats['num_classes']:.1f}")
    
    # Get a sample
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample class names: {[train_loader.dataset.get_class_name(label.item()) for label in labels[:5]]}")
        break