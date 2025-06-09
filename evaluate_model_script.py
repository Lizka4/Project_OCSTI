import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from src.data.data_loader import ThermalDataLoader
from src.models.resnet_model import ThermalResNet50
from src.models.model_evaluator import ModelEvaluator
from src.visualization.confusion_matrix import ConfusionMatrixVisualizer
from src.visualization.results_plotter import ResultsPlotter
from src.utils.logger import setup_logger
from src.utils.metrics import tensor_to_numpy
from config.model_config import MODEL_CONFIG

def main():
    logger = setup_logger('EvaluationScript')
    logger.info('Starting model evaluation...')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    
    data_loader = ThermalDataLoader()
    _, _, test_loader = data_loader.create_data_loaders()
    
    model = ThermalResNet50(num_classes=MODEL_CONFIG['num_classes'])
    
    model_path = f'models/saved_models/{MODEL_CONFIG["model_name"]}_final.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f'Model loaded from: {model_path}')
    else:
        logger.error(f'Model file not found: {model_path}')
        return
    
    evaluator = ModelEvaluator(model, device=device)
    
    logger.info('Evaluating model on test set...')
    metrics = evaluator.evaluate_model(test_loader)
    
    logger.info('Generating confusion matrix...')
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(tensor_to_numpy(predicted))
            all_targets.extend(tensor_to_numpy(target))
    
    cm_visualizer = ConfusionMatrixVisualizer()
    
    cm_path = cm_visualizer.plot_confusion_matrix(all_targets, all_predictions)
    logger.info(f'Confusion matrix saved: {cm_path}')
    
    cm_norm_path = cm_visualizer.plot_confusion_matrix(all_targets, all_predictions, normalize=True)
    logger.info(f'Normalized confusion matrix saved: {cm_norm_path}')
    
    report_path = cm_visualizer.plot_classification_report(all_targets, all_predictions)
    logger.info(f'Classification report saved: {report_path}')
    
    detailed_path = cm_visualizer.create_detailed_analysis(all_targets, all_predictions)
    logger.info(f'Detailed analysis saved: {detailed_path}')
    
    plotter = ResultsPlotter()
    metrics_plot_path = plotter.plot_metrics_comparison(metrics)
    logger.info(f'Metrics comparison saved: {metrics_plot_path}')
    
    evaluator.save_evaluation_results(metrics, 'results/metrics/evaluation_results.json')
    
    class_names = ['Human', 'Dog']
    class_metrics = evaluator.get_class_predictions(test_loader, class_names)
    
    logger.info('Per-class performance:')
    for class_name, class_metric in class_metrics.items():
        logger.info(f'{class_name}: Precision={class_metric["precision"]:.3f}, '
                   f'Recall={class_metric["recall"]:.3f}, '
                   f'F1-Score={class_metric["f1_score"]:.3f}')
    
    logger.info('Model evaluation completed successfully!')

if __name__ == '__main__':
    main()