import os
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_FILES = {
    "train": os.path.join(BASE_DIR, "data", "pcam", "training_split.h5"),
    "val":   os.path.join(BASE_DIR, "data", "pcam", "validation_split.h5"),
    "test":  os.path.join(BASE_DIR, "data", "pcam", "test_split.h5"),
}

LABEL_FILES = {
    "train": os.path.join(BASE_DIR, "data", "Labels", "Labels", "camelyonpatch_level_2_split_train_y.h5"),
    "val":   os.path.join(BASE_DIR, "data", "Labels", "Labels", "camelyonpatch_level_2_split_valid_y.h5"),
    "test":  os.path.join(BASE_DIR, "data", "Labels", "Labels", "camelyonpatch_level_2_split_test_y.h5"),
}

class PCAMDataset(Dataset):
    def __init__(self, split: str, transform=None):
        self.image_file = IMAGE_FILES[split]
        self.label_file = LABEL_FILES[split]
        self.transform = transform

        with h5py.File(self.label_file, 'r') as f:
            key = list(f.keys())[0]
            self.labels = f[key][:].squeeze().astype(np.int64)
        
        self.length = len(self.labels)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.image_file, 'r') as f:
            key = list(f.keys())[0]
            image = f[key][idx]
        
        image = Image.fromarray(image.astype('uint8'))

        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]
    

def get_transforms(split: str):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    

def get_dataloaders(batch_size: int = 64, subset_fraction: float = 1.0):
    train_dataset = PCAMDataset(split="train", transform=get_transforms("train"))
    val_dataset = PCAMDataset(split="val", transform=get_transforms("val"))

    if subset_fraction < 1.0:
        train_size = int(len(train_dataset) * subset_fraction)
        val_size = int(len(val_dataset) * subset_fraction)
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}  | Val batches: {len(val_loader)}")

    return train_loader, val_loader