import cv2
import numpy as np
from torchvision import transforms
import torch

class ThermalAugmentation:
    def __init__(self):
        self.transform_functions = {
            'horizontal_flip': self.horizontal_flip,
            'vertical_flip': self.vertical_flip,
            'rotation': self.rotation,
            'brightness': self.brightness_adjustment,
            'contrast': self.contrast_adjustment,
            'noise': self.add_gaussian_noise,
            'blur': self.gaussian_blur,
            'elastic': self.elastic_transform
        }
    
    def horizontal_flip(self, image, probability=0.5):
        if np.random.random() < probability:
            return cv2.flip(image, 1)
        return image
    
    def vertical_flip(self, image, probability=0.3):
        if np.random.random() < probability:
            return cv2.flip(image, 0)
        return image
    
    def rotation(self, image, max_angle=15):
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def brightness_adjustment(self, image, factor_range=(0.8, 1.2)):
        factor = np.random.uniform(factor_range[0], factor_range[1])
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def contrast_adjustment(self, image, factor_range=(0.8, 1.2)):
        factor = np.random.uniform(factor_range[0], factor_range[1])
        mean = np.mean(image)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
    def add_gaussian_noise(self, image, noise_level=25):
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def gaussian_blur(self, image, kernel_range=(3, 7)):
        kernel_size = np.random.choice(range(kernel_range[0], kernel_range[1] + 1, 2))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    def elastic_transform(self, image, alpha=50, sigma=5):
        h, w = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha, (0, 0), sigma)
        dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1) * alpha, (0, 0), sigma)
        
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    def apply_random_augmentations(self, image, num_augmentations=2):
        augmented = image.copy()
        selected_augs = np.random.choice(
            list(self.transform_functions.keys()), 
            size=min(num_augmentations, len(self.transform_functions)), 
            replace=False
        )
        
        for aug_name in selected_augs:
            augmented = self.transform_functions[aug_name](augmented)
        
        return augmented
    
    def create_augmentation_pipeline(self, augmentations=['horizontal_flip', 'rotation', 'brightness']):
        def pipeline(image):
            augmented = image.copy()
            for aug_name in augmentations:
                if aug_name in self.transform_functions:
                    augmented = self.transform_functions[aug_name](augmented)
            return augmented
        return pipeline