import torch
import torch.nn as nn
import torchvision.models as models
from config.model_config import MODEL_CONFIG

class ThermalResNet50(nn.Module):
    def __init__(self, num_classes=None):
        super(ThermalResNet50, self).__init__()
        self.num_classes = num_classes or MODEL_CONFIG['num_classes']
        
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.normal_(self.resnet.fc.weight, 0, 0.01)
        nn.init.constant_(self.resnet.fc.bias, 0)
    
    def forward(self, x):
        return self.resnet(x)
    
    def get_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        features = self.resnet.layer4(x)
        
        return features
    
    def save_model(self, filepath):
        torch.save(self.state_dict(), filepath)
    
    def load_model(self, filepath):
        self.load_state_dict(torch.load(filepath))