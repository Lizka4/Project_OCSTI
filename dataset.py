import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config.data_config import DATA_CONFIG

class ThermalDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = ['human', 'dog']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            if os.path.exists(class_path):
                for filename in os.listdir(class_path):
                    if any(filename.lower().endswith(ext) for ext in DATA_CONFIG['image_extensions']):
                        file_path = os.path.join(class_path, filename)
                        samples.append((file_path, self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self):
        counts = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts