import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class ResNet18(nn.Module): 
    def __init__(self, state):
        super(ResNet18, self).__init__()
        self.backbone = resnet18(pretrained=True)  
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
        if state == 'freeze':
            for param in self.base.parameters():
                param.requires_grad = False
        self.out_features = 512
        self.fc = nn.Linear(self.out_features, 2) 
    def forward(self, x):
        x = torch.squeeze(self.base(x))
        return self.fc(x)

class ResNet50(nn.Module): 
    def __init__(self, state):
        super(ResNet50, self).__init__()
        self.backbone = resnet50(pretrained=True)  
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
        if state == 'freeze':
            for param in self.base.parameters():
                param.requires_grad = False
        self.out_features = 2048
        self.fc = nn.Linear(self.out_features, 2) 
    def forward(self, x):
        x = torch.squeeze(self.base(x))
        return self.fc(x)