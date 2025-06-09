import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models.resnet_model import ThermalResNet50
from src.visualization.activation_maps import ActivationVisualizer
from src.visualization.results_plotter import ResultsPlotter
from src.utils.logger import setup_logger
from src.utils.file_utils import load_json, ensure_dir
from config.model_config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Visualize model results')
    parser.add_argument('--image_path', type=str, help='Path to thermal image for activation visualization')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--history_path', type=str, default='results/metrics/training_history.json', help='Path to training history')
    parser.add_argument('--output_dir', type=str, default='results/visualizations', help='Output directory')
    
    args = parser.parse_args()
    
    logger = setup_logger('VisualizationScript')
    logger.info('Starting result visualization...')
    
    ensure_dir(args.output_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ThermalResNet50(num_classes=MODEL_CONFIG['num_classes'])
    
    model_path = args.model_path or f'models/saved_models/{MODEL_CONFIG["model_name"]}_final.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f'Model loaded from: {model_path}')
    else:
        logger.error(f'Model file not found: {model_path}')
        return
    
    if args.image_path and os.path.exists(args.image_path):
        logger.info(f'Creating activation visualization for: {args.image_path}')
        
        visualizer = ActivationVisualizer(model)
        
        result = visualizer.visualize_prediction(
            args.image_path, 
            save_path=os.path.join(args.output_dir, 'activation_map.jpg')
        )
        
        logger.info(f'Predicted class: {result["predicted_class"]}')
        logger.info(f'Confidence: {result["confidence"]:.4f}')
        logger.info(f'Activation map saved to: {args.output_dir}')
    
    if os.path.exists(args.history_path):
        logger.info('Creating training history visualizations...')
        
        history = load_json(args.history_path)
        plotter = ResultsPlotter(output_dir=args.output_dir)
        
        training_plot = plotter.plot_training_history(
            history, 
            save_path=os.path.join(args.output_dir, 'training_history.png')
        )
        logger.info(f'Training history plot saved: {training_plot}')
    
    if os.path.exists('results/metrics/evaluation_results.json'):
        logger.info('Creating metrics visualization...')
        
        metrics = load_json('results/metrics/evaluation_results.json')
        plotter = ResultsPlotter(output_dir=args.output_dir)
        
        metrics_plot = plotter.plot_metrics_comparison(
            metrics,
            save_path=os.path.join(args.output_dir, 'metrics_comparison.png')
        )
        logger.info(f'Metrics comparison saved: {metrics_plot}')
    
    logger.info('Visualization script completed successfully!')

if __name__ == '__main__':
    main()