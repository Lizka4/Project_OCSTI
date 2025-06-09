import numpy as np
import torch
import cv2

class ThermalNormalization:
    def __init__(self):
        self.thermal_mean = 0.5
        self.thermal_std = 0.5
        
    def min_max_normalization(self, image):
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 0:
            normalized = (image - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(image)
        return normalized
    
    def z_score_normalization(self, image):
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            normalized = (image - mean) / std
        else:
            normalized = image - mean
        return normalized
    
    def percentile_normalization(self, image, lower_percentile=1, upper_percentile=99):
        lower = np.percentile(image, lower_percentile)
        upper = np.percentile(image, upper_percentile)
        if upper - lower > 0:
            normalized = np.clip((image - lower) / (upper - lower), 0, 1)
        else:
            normalized = np.zeros_like(image)
        return normalized
    
    def adaptive_normalization(self, image, window_size=64):
        h, w = image.shape
        normalized = np.zeros_like(image, dtype=np.float32)
        
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = image[i:min(i+window_size, h), j:min(j+window_size, w)]
                normalized_window = self.min_max_normalization(window)
                normalized[i:min(i+window_size, h), j:min(j+window_size, w)] = normalized_window
        
        return normalized
    
    def thermal_specific_normalization(self, image):
        image_float = image.astype(np.float32) / 255.0
        normalized = (image_float - self.thermal_mean) / self.thermal_std
        return normalized
    
    def histogram_equalization(self, image):
        return cv2.equalizeHist(image.astype(np.uint8))
    
    def robust_normalization(self, image, quantile_range=(25, 75)):
        q1, q3 = np.percentile(image, quantile_range)
        iqr = q3 - q1
        if iqr > 0:
            normalized = (image - q1) / iqr
        else:
            normalized = image - q1
        return np.clip(normalized, 0, 1)
    
    def normalize_batch(self, batch_images, method='min_max'):
        normalized_batch = []
        
        normalization_methods = {
            'min_max': self.min_max_normalization,
            'z_score': self.z_score_normalization,
            'percentile': self.percentile_normalization,
            'adaptive': self.adaptive_normalization,
            'thermal': self.thermal_specific_normalization,
            'histogram': self.histogram_equalization,
            'robust': self.robust_normalization
        }
        
        norm_func = normalization_methods.get(method, self.min_max_normalization)
        
        for image in batch_images:
            normalized = norm_func(image)
            normalized_batch.append(normalized)
        
        return np.array(normalized_batch)
    
    def denormalize(self, normalized_image, original_min=None, original_max=None):
        if original_min is not None and original_max is not None:
            return normalized_image * (original_max - original_min) + original_min
        else:
            return normalized_image * 255.0