import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

class ThermalPreprocessor:
    def __init__(self, target_size=224):
        self.target_size = target_size
        
    def load_thermal_image(self, image_path):
        try:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                image = np.array(Image.open(image_path).convert('L'))
            return image
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {str(e)}")
    
    def temperature_normalization(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        return (normalized * 255).astype(np.uint8)
    
    def adaptive_histogram_equalization(self, image, clip_limit=3.0, grid_size=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)
    
    def noise_reduction(self, image, kernel_size=5):
        return cv2.medianBlur(image, kernel_size)
    
    def edge_enhancement(self, image, alpha=1.5):
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(image, -1, kernel)
        return cv2.addWeighted(image, 1-alpha, enhanced, alpha, 0)
    
    def thermal_gradient_enhancement(self, image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        return cv2.addWeighted(image, 0.7, gradient_magnitude, 0.3, 0)
    
    def process_thermal_image(self, image_path):
        image = self.load_thermal_image(image_path)
        image = cv2.resize(image, (self.target_size, self.target_size))
        image = self.temperature_normalization(image)
        image = self.adaptive_histogram_equalization(image)
        image = self.noise_reduction(image)
        image = self.thermal_gradient_enhancement(image)
        return image
    
    def create_thermal_transform(self, mode='train'):
        base_transforms = [
            transforms.ToPILImage(),
            transforms.Resize((self.target_size, self.target_size))
        ]
        
        if mode == 'train':
            base_transforms.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ])
        
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        return transforms.Compose(base_transforms)