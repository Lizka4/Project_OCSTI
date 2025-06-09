import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from torchvision import transforms
from src.models.resnet_model import ThermalResNet50
from src.models.model_evaluator import ModelEvaluator
from src.preprocessing.thermal_processor import ThermalPreprocessor
from src.utils.logger import setup_logger
from config.model_config import MODEL_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Classify thermal image')
    parser.add_argument('--image_path', type=str, required=True, help='Path to thermal image')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model')
    parser.add_argument('--output_dir', type=str, default='results/predictions', help='Output directory')
    
    args = parser.parse_args()
    
    logger = setup_logger('ClassificationScript')
    logger.info(f'Classifying image: {args.image_path}')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ThermalResNet50(num_classes=MODEL_CONFIG['num_classes'])
    
    model_path = args.model_path or f'models/saved_models/{MODEL_CONFIG["model_name"]}_final.pth'
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        logger.info(f'Model loaded from: {model_path}')
    else:
        logger.error(f'Model file not found: {model_path}')
        return
    
    transform = transforms.Compose([
        transforms.Resize((MODEL_CONFIG['input_size'], MODEL_CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        image = Image.open(args.image_path).convert('L')
        image_tensor = transform(image)
        
        evaluator = ModelEvaluator(model, device=device)
        result = evaluator.predict_single_image(image_tensor)
        
        class_names = ['Human', 'Dog']
        predicted_class = class_names[result['prediction']]
        confidence = result['confidence']
        
        logger.info(f'Prediction: {predicted_class}')
        logger.info(f'Confidence: {confidence:.4f}')
        logger.info(f'Probabilities: {result["probabilities"]}')
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        result_file = os.path.join(args.output_dir, 'prediction_result.txt')
        with open(result_file, 'w') as f:
            f.write(f'Image: {args.image_path}\n')
            f.write(f'Predicted Class: {predicted_class}\n')
            f.write(f'Confidence: {confidence:.4f}\n')
            f.write(f'Probabilities: {result["probabilities"]}\n')
        
        logger.info(f'Results saved to: {result_file}')
        
    except Exception as e:
        logger.error(f'Error processing image: {str(e)}')

if __name__ == '__main__':
    main()