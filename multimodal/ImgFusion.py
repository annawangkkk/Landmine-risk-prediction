from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchvision

from dataset import *
from model import ResNet18Fusion, ResNet50Fusion

from tqdm import tqdm
from datetime import datetime

def combined_model(split):
    root = '/home/siqiz/social_good/Landmine-risk-prediction'
    epochs = 100
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor(), 
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    five_seeds_res = np.zeros((5,))
    cols_info = FeatureNames('new')
    numeric_cols = cols_info.numeric_cols
    binary_cols = cols_info.binary_cols
    features = numeric_cols + binary_cols
    num_tab_features = len(features)
    for seed in range(5):
        if split == 'random':
            model = nn.DataParallel(ResNet18Fusion(num_tab_features,'unfreeze').to(device))
            model_dict = model.state_dict()
            pretrained = torch.load(f'/home/siqiz/social_good/satellite_models/random/pure/res18_unfreeze_satellite_seed{seed}.pth')
            pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            train_data = Landmine(root, split, 'train_labeled', 'satellite', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'satellite', transform=test_transform)
            log_dir = f'/home/siqiz/social_good/satellite_models/random/combined/log.txt'
            save_dir = f'/home/siqiz/social_good/satellite_models/random/combined/res18_unfreeze_satellite_seed{seed}.pth'
        elif split == 'sonson':
            model = nn.DataParallel(ResNet50Fusion(num_tab_features,'unfreeze').to(device)) # use unfreeze for all settings here
            model_dict = model.state_dict()
            pretrained = torch.load(f'/home/siqiz/social_good/terrain_models/sonson/pure/res50_freeze_terrain_seed{seed}.pth')
            pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            train_data = Landmine(root, split, 'train_labeled', 'terrain', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'terrain', transform=test_transform)
            log_dir = f'/home/siqiz/social_good/terrain_models/sonson/combined/log.txt'
            save_dir = f'/home/siqiz/social_good/terrain_models/sonson/combined/red50_unfreeze_terrain_seed{seed}.pth'
        elif split == 'caldas':
            model = nn.DataParallel(ResNet50Fusion(num_tab_features,'unfreeze').to(device)) # use unfreeze for all settings here
            model_dict = model.state_dict()
            pretrained = torch.load(f'/home/siqiz/social_good/satellite_models/caldas/pure/res50_unfreeze_satellite_seed{seed}.pth')
            pretrained_dict = {k: v for k, v in pretrained.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            model.load_state_dict(model_dict)
            train_data = Landmine(root, split, 'train_labeled', 'satellite', transform=train_transform)
            test_data = Landmine(root, split, 'test_labeled', 'satellite', transform=test_transform)
            log_dir = f'/home/siqiz/social_good/satellite_models/caldas/combined/log.txt'
            save_dir = f'/home/siqiz/social_good/satellite_models/caldas/combined/res50_unfreeze_satellite_seed{seed}.pth'
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.99)
        seed_everything(seed)
        model.train()
        auc_max = 0
        early_stop = 0
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for epoch in tqdm(range(epochs)):
            if early_stop >= 30:
                print("Valid ROC consistently decreases. Skip training.")
                break
            fin_acc = []
            num_examples_train = 0
            t_start = datetime.now()
            for idx, (img, tab, target) in enumerate(train_loader):
                img, tab, target = img.to(device), tab.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(img, tab).squeeze() 
                loss = nn.CrossEntropyLoss()
                output = loss(out, target)
                output.backward()
                optimizer.step()
                num_examples_train += len(img)
                rocauc = roc_auc_score(target.detach().cpu().numpy(),F.softmax(out,dim=1)[:,1].detach().cpu().numpy())
                fin_acc.append(rocauc)
                if idx % 100 == 1:
                    with open(log_dir,'a') as f:
                        f.write(f"Train Epoch: {epoch}, Trained Examples: {num_examples_train}, Loss: {output.item()}\n")
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            with open(log_dir,'a') as f:
                f.write(f"Train one epoch takes: {t_delta} sec, ROC_AUC: {sum(fin_acc)/len(fin_acc)}\n")
            eval_rocauc = evaluation(model, test_loader, log_dir)
            if eval_rocauc > auc_max:
                early_stop = 0
                auc_max = eval_rocauc
                torch.save(model.state_dict(), save_dir)
            else:
                early_stop += 1
            five_seeds_res[seed] = auc_max
    print(f"Final auc: {np.mean(five_seeds_res)}, split: {split}")
    return five_seeds_res

def evaluation(model, eval_loader, log_dir):
    model.eval()
    fin_prob = []
    fin_targets = []
    with torch.no_grad():
        for img, tab, target in eval_loader:
            img, tab = img.to(device), tab.to(device)
            out = model(img, tab).squeeze()
            fin_prob.extend(list(F.softmax(out,dim=1)[:,1].detach().cpu().numpy()))
            fin_targets.extend(list(target.detach().cpu().numpy()))
    rocauc = roc_auc_score(fin_targets, fin_prob)
    with open(log_dir,'a') as f:
        f.write(f"Valid ROC_AUC: {rocauc}\n")
    return rocauc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for split in ['random','sonson','caldas']:
        combined_model(split)