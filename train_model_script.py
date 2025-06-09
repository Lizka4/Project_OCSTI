import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.data.data_loader import ThermalDataLoader
from src.models.resnet_model import ThermalResNet50
from src.models.model_trainer import ModelTrainer
from src.visualization.results_plotter import ResultsPlotter
from src.utils.logger import setup_logger
from src.utils.file_utils import save_json
from config.model_config import MODEL_CONFIG
from config.data_config import DATA_CONFIG

def main():
    logger = setup_logger('TrainingScript')
    logger.info('Starting thermal object classification training...')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    data_loader = ThermalDataLoader()
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()
    
    logger.info(f'Training samples: {len(train_loader.dataset)}')
    logger.info(f'Validation samples: {len(val_loader.dataset)}')
    logger.info(f'Test samples: {len(test_loader.dataset)}')
    
    model = ThermalResNet50(num_classes=MODEL_CONFIG['num_classes'])
    trainer = ModelTrainer(model, device=device)
    
    logger.info('Starting model training...')
    history = trainer.fit(train_loader, val_loader)
    
    logger.info('Training completed. Generating visualizations...')
    plotter = ResultsPlotter()
    
    training_plot_path = plotter.plot_training_history(history)
    logger.info(f'Training history plot saved: {training_plot_path}')
    
    save_json(history, 'results/metrics/training_history.json')
    logger.info('Training history saved to JSON')
    
    final_model_path = f'models/saved_models/{MODEL_CONFIG["model_name"]}_final.pth'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f'Final model saved: {final_model_path}')
    
    logger.info('Training script completed successfully!')

if __name__ == '__main__':
    main()