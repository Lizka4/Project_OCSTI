import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class ThermalImageProcessor:
    def __init__(self, target_size=224):
        self.target_size = target_size
        
    def load_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.array(Image.open(image_path).convert('L'))
        return image
    
    def resize_image(self, image, size=None):
        size = size or self.target_size
        return cv2.resize(image, (size, size))
    
    def normalize_image(self, image):
        return image.astype(np.float32) / 255.0
    
    def apply_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
    
    def apply_gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def enhance_contrast(self, image, alpha=1.2, beta=10):
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def preprocess_thermal_image(self, image_path):
        image = self.load_image(image_path)
        image = self.resize_image(image)
        image = self.apply_clahe(image)
        image = self.apply_gaussian_filter(image)
        image = self.normalize_image(image)
        return image
    
    def to_tensor(self, image):
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image).float()