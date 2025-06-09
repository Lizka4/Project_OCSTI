import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        target_layer = dict(self.model.named_modules())[self.target_layer]
        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_backward_hook(backward_hook))
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = torch.mean(gradients, dim=[1, 2])
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        
        return cam.detach().cpu().numpy()
    
    def overlay_heatmap(self, image, heatmap, alpha=0.6):
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

class ActivationVisualizer:
    def __init__(self, model):
        self.model = model
        self.class_names = ['Human', 'Dog']
    
    def create_gradcam(self, input_tensor, target_layer='resnet.layer4'):
        gradcam = GradCAM(self.model, target_layer)
        
        cam = gradcam.generate_cam(input_tensor)
        gradcam.cleanup()
        
        return cam
    
    def visualize_prediction(self, image_path, save_path=None):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image_tensor = self.preprocess_image(image)
        
        prediction = self.model(image_tensor.unsqueeze(0))
        predicted_class = torch.argmax(prediction, dim=1).item()
        confidence = torch.softmax(prediction, dim=1).max().item()
        
        cam = self.create_gradcam(image_tensor.unsqueeze(0))
        overlay = self.overlay_cam_on_image(image, cam)
        
        result = {
            'predicted_class': self.class_names[predicted_class],
            'confidence': confidence,
            'overlay_image': overlay,
            'activation_map': cam
        }
        
        if save_path:
            cv2.imwrite(save_path, overlay)
        
        return result
    
    def preprocess_image(self, image):
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)
        image_tensor = (image_tensor - 0.5) / 0.5
        return image_tensor
    
    def overlay_cam_on_image(self, image, cam, alpha=0.6):
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        overlay = cv2.addWeighted(image_rgb, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    def batch_visualization(self, image_paths, output_dir):
        results = []
        for i, image_path in enumerate(image_paths):
            save_path = f"{output_dir}/activation_map_{i}.jpg"
            result = self.visualize_prediction(image_path, save_path)
            results.append(result)
        return results