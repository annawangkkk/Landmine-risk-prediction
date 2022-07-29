import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import numpy as np

from dataset import *
from model import ResNet18

from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undo the normalization and returns the reconstructed images in the input domain.
    """
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
    
def tensor_to_rgb(t, mean, std):
    return np.array(torchvision.transforms.ToPILImage()
                    (NormalizeInverse(mean=mean, std=std)(t)))

def draw_saliency_map():
    seed = 0
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    root = '~/social_good/Landmine-risk-prediction'
    for split in ['caldas']:
        if split == 'random':
            model = nn.DataParallel(ResNet18('unfreeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/satellite_models/random/pure/res18_unfreeze_satellite_seed{seed}.pth'))
            test_data = Landmine(root, split, 'test_labeled', 'satellite', transform=test_transform)
        elif split == 'sonson':
            model = nn.DataParallel(ResNet18('unfreeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/terrain_models/sonson/pure/res18_unfreeze_terrain_seed{seed}.pth'))
            test_data = Landmine(root, split, 'test_labeled', 'terrain', transform=test_transform)
        elif split == 'caldas':
            model = nn.DataParallel(ResNet18('unfreeze').to(device))
            model.load_state_dict(torch.load(f'~/social_good/roadmap_models/caldas/pure/res18_unfreeze_roadmap_seed{seed}.pth'))
            test_data = Landmine(root, split, 'test_labeled', 'roadmap', transform=test_transform)
        dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
        target_layers = [model.module.backbone.layer4[-1]]
        for idx, (data, target) in enumerate(tqdm(dataloader)):
            input_tensor = data
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)]) 
            grayscale_cam = grayscale_cam[0, :]
            rgb_img = np.float32(tensor_to_rgb(input_tensor.squeeze(), mean, std))/255
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            pil_visual = Image.fromarray(visualization, 'RGB')
            pil_visual.save(root + f'/refactor/images/results/{split}_cam/{str(idx).zfill(4)}.jpg')  

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    draw_saliency_map()