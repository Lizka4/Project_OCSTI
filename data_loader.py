import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .dataset import ThermalDataset
from config.data_config import DATA_CONFIG
from config.model_config import MODEL_CONFIG

class ThermalDataLoader:
    def __init__(self, data_path=None):
        self.data_path = data_path or DATA_CONFIG['raw_data_path']
        self.batch_size = MODEL_CONFIG['batch_size']
        self.num_workers = DATA_CONFIG['num_workers']
        self.pin_memory = DATA_CONFIG['pin_memory']
        
    def get_transforms(self, mode='train'):
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size'])),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size'])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    def create_data_loaders(self):
        dataset = ThermalDataset(self.data_path, transform=self.get_transforms('train'))
        
        train_size = int(DATA_CONFIG['train_split'] * len(dataset))
        val_size = int(DATA_CONFIG['val_split'] * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        val_dataset.dataset.transform = self.get_transforms('val')
        test_dataset.dataset.transform = self.get_transforms('test')
        
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory
        )
        
        return train_loader, val_loader, test_loader