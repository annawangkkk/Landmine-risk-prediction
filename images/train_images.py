import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
import torchvision

from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from datetime import datetime

from dataset import *
from model import ResNet18, ResNet50
from ..utils import *

def train_test(state, arch, test_name, map_type):
    '''
    state: freeze/unfreeze
    arch: res18/res50
    test_name : random/sonson/caldas
    map_type : satellite/terrain/roadmap
    '''
    epochs = 100
    log_dir = f'~/social_good/{map_type}_models/{test_name}/pure/log.txt'
    train_transform = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.RandomVerticalFlip(),
                                                      torchvision.transforms.ToTensor(), 
                                                      torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    five_seeds_res = np.zeros((5,))
    for seed in range(5):
        if arch == 'res18':
            model = nn.DataParallel(ResNet18(state).to(device))
        else: # 'res50'
            model = nn.DataParallel(ResNet50(state).to(device))
        optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.99)
        seed_everything(seed)
        model.train()
        auc_max = 0
        early_stop = 0
        train_data = Landmine('/home/siqiz/social_good/Landmine-risk-prediction', test_name, 'train_labeled', map_type, transform=train_transform)
        test_data = Landmine('/home/siqiz/social_good/Landmine-risk-prediction', test_name, 'test_labeled', map_type, transform=test_transform)
        train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
        for epoch in tqdm(range(epochs)):
            if early_stop >= 30:
                print("Valid ROC consistently decreases. Skip training.")
                break
            fin_acc = []
            num_examples_train = 0
            t_start = datetime.now()
            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(data).squeeze() 
                loss = nn.CrossEntropyLoss()
                output = loss(out, target)
                output.backward()
                optimizer.step()
                num_examples_train += len(data)
                rocauc = roc_auc_score(target.detach().cpu().numpy(),F.softmax(out,dim=1)[:,1].detach().cpu().numpy())
                fin_acc.append(rocauc)
                if idx % 100 == 1:
                    with open(f'/home/siqiz/social_good/{map_type}_models/{test_name}/pure/log.txt','a') as f:
                        f.write(f"Train Epoch: {epoch}, Trained Examples: {num_examples_train}, Loss: {output.item()}\n")
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            with open(f'/home/siqiz/social_good/{map_type}_models/{test_name}/pure/log.txt','a') as f:
                f.write(f"Train one epoch takes: {t_delta} sec, ROC_AUC: {sum(fin_acc)/len(fin_acc)}\n")
            eval_rocauc = evaluation(model, test_loader, log_dir)
            if eval_rocauc > auc_max:
                early_stop = 0
                auc_max = eval_rocauc
                torch.save(model.state_dict(), f"/home/siqiz/social_good/{map_type}_models/{test_name}/pure/{arch}_{state}_{map_type}_seed{seed}.pth")
            else:
                early_stop += 1
            five_seeds_res[seed] = auc_max
    print(f"Final auc: {np.mean(five_seeds_res)}, state: {state}, arch: {arch}, test_split: {test_name}, map_type: {map_type}")
    return five_seeds_res

def evaluation(model, eval_loader, log_dir):
    model.eval()
    fin_prob = []
    fin_targets = []
    with torch.no_grad():
        for data, target in eval_loader:
            data = data.to(device)
            out = model(data).squeeze()
            fin_prob.extend(list(F.softmax(out,dim=1)[:,1].detach().cpu().numpy()))
            fin_targets.extend(list(target.detach().cpu().numpy()))
    rocauc = roc_auc_score(fin_targets, fin_prob)
    with open(log_dir,'a') as f:
        f.write(f"Valid ROC_AUC: {rocauc}\n")
    return rocauc

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    five_seeds_all = []
    for state in ['freeze','unfreeze']:
        for arch in ['res18','res50']:
            for test_name in ['random','sonson','caldas']:
                for map_type in ['satellite','terrain','roadmap']:
                    five_seeds_all.append(train_test(state, arch, test_name, map_type))