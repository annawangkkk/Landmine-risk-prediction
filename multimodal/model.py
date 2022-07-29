import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

class ResNet18Fusion(nn.Module): 
    def __init__(self, num_tab_features, state):
        super(ResNet18Fusion, self).__init__()
        self.backbone = resnet18(pretrained=True)  
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
        self.out_features = 512
        if state == 'freeze':
            for param in self.base.parameters():
                param.requires_grad = False 
        self.mlp = nn.Sequential(nn.Linear(self.out_features+num_tab_features,32),
                                nn.BatchNorm1d(32),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(32, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2) 
                                )    
    def forward(self, img, tab):
        img = torch.squeeze(self.base(img))
        common = torch.cat([img,tab],dim=1)
        common = common.float()
        return self.mlp(common)

class ResNet50Fusion(nn.Module): 
    def __init__(self, num_tab_features, state):
        super(ResNet50Fusion, self).__init__()
        self.backbone = resnet50(pretrained=True)  
        self.base = nn.Sequential(*list(self.backbone.children())[:-1])
        self.out_features = 2048
        if state == 'freeze':
            for param in self.base.parameters():
                param.requires_grad = False
        self.mlp = nn.Sequential(nn.Linear(self.out_features+num_tab_features,32),
                                nn.BatchNorm1d(32),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(32, 128),
                                nn.ReLU(),
                                nn.Linear(128, 2) 
                                )
    def forward(self, img, tab):
        img = torch.squeeze(self.base(img))
        common = torch.cat([img,tab],dim=1)
        common = common.float()
        return self.mlp(common)