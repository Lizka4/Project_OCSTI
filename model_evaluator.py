import torch
import torch.nn as nn
import numpy as np
from src.utils.metrics import calculate_all_metrics, tensor_to_numpy
from src.utils.logger import setup_logger
from src.utils.file_utils import save_json

class ModelEvaluator:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.logger = setup_logger('ModelEvaluator')
        
    def evaluate_model(self, test_loader):
        self.model.eval()
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                
                all_predictions.extend(tensor_to_numpy(predicted))
                all_targets.extend(tensor_to_numpy(target))
        
        test_loss /= len(test_loader)
        
        metrics = calculate_all_metrics(all_targets, all_predictions)
        metrics['test_loss'] = test_loss
        
        self.logger.info(f'Test Loss: {test_loss:.4f}')
        self.logger.info(f'Test Accuracy: {metrics["accuracy"]:.4f}')
        self.logger.info(f'Test Precision: {metrics["precision"]:.4f}')
        self.logger.info(f'Test Recall: {metrics["recall"]:.4f}')
        self.logger.info(f'Test F1-Score: {metrics["f1_score"]:.4f}')
        
        return metrics
    
    def predict_single_image(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            return {
                'prediction': tensor_to_numpy(predicted)[0],
                'probabilities': tensor_to_numpy(probabilities)[0],
                'confidence': float(torch.max(probabilities))
            }
    
    def predict_batch(self, data_loader):
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                predictions.extend(tensor_to_numpy(predicted))
                probabilities.extend(tensor_to_numpy(probs))
        
        return predictions, probabilities
    
    def get_class_predictions(self, test_loader, class_names):
        metrics = self.evaluate_model(test_loader)
        confusion_matrix = np.array(metrics['confusion_matrix'])
        
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            tn = confusion_matrix.sum() - tp - fp - fn
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': confusion_matrix[i, :].sum()
            }
        
        return class_metrics
    
    def save_evaluation_results(self, metrics, filepath):
        save_json(metrics, filepath)
        self.logger.info(f'Evaluation results saved to {filepath}')