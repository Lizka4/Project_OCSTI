import os

DATA_CONFIG = {
    'data_root': 'data',
    'raw_data_path': os.path.join('data', 'raw'),
    'processed_data_path': os.path.join('data', 'processed'),
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'image_extensions': ['.jpg', '.jpeg', '.png', '.tiff'],
    'num_workers': 4,
    'pin_memory': True
}