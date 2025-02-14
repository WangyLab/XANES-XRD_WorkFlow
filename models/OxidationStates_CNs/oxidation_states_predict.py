import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

element = 'Cr'  # Change element here
data_json_pth = 'D:\\OneDrive - USTC\\Code\\XAS\\Element_Cls\\States.json'  # This is the path to the json file containing the elements data
df = pd.read_json(data_json_pth)

pattern  = re.compile(f'{element}(\d*\d*[+-])')
X_states = df['possible_species'].dropna().apply(lambda x: pattern.findall(str(x)))  # X state in each entry

SpecList = []
StatesList = []
for i in range(len(df)):
    if element in df['Elements'][i]:
        idx = df['Elements'][i].index(element)
        yspec = df['ySpec'][i][idx]
        
        if len(X_states[i]) == 1 and X_states[i][0] in ['2+', '3+', '4+', '6+']:  # Choose some reasonable valence states of element X
            state = X_states[i][0]
            SpecList.append(yspec)
            StatesList.append(state)
            
encoded, uniques = pd.factorize(StatesList)
print(uniques)
TargetsTensor = torch.tensor(encoded)
SpecTensor = torch.tensor(SpecList)
unique_states = pd.factorize(StatesList)[1]

'''Dataset'''
class StatesDataset(Dataset):
    def __init__(self, spectra, states):
        assert len(spectra) == len(states)
        self.spec = spectra
        self.states = states
        
    def __len__(self):
        return len(self.spec)
    
    def __getitem__(self, idx):
        spec = self.spec[idx]
        state = self.states[idx]
        return spec, state

def dataset_random_split(dataset, train_value=0.7, val_value=0.15):
    train_size = int(len(dataset)*train_value)
    val_size = int(len(dataset)*val_value)
    test_size = len(dataset)-train_size-val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_idx = train_dataset.indices
    val_idx = val_dataset.indices
    test_idx = test_dataset.indices
    np.save('train_idx.npy', np.array(train_idx))
    np.save('val_idx.npy', np.array(val_idx))
    np.save('test_idx.npy', np.array(test_idx))
    
'''Net'''
class Spec2State(nn.Module):
    def __init__(self):
        super(Spec2State, self).__init__()
        
        self.dense1 = nn.Linear(200, 512)
        # self.dense2 = nn.Linear(1024, 512)
        self.dense2 = nn.Linear(512, 2)
        
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

def compute_class_weights(labels):
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(class_weights, dtype=torch.float)

def train(num_epoch):
    model = Spec2State().to(device)
    class_weights = compute_class_weights(TargetsTensor)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    Precision_best = 0.0
    Recall_best = 0.0
    F1_best = 0.0
    Acc_best = 0.0
    
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        
        for spectra, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epoch}'):
            spectra = spectra.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            corrects = torch.sum(preds == targets)
            running_loss += loss.item() * spectra.size(0)
            running_corrects += corrects.item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            for spectra, targets in tqdm(val_loader, desc='Val'):
                spectra = spectra.to(device)
                targets = targets.to(device)
                outputs = model(spectra)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == targets)
                
                val_loss += loss.item() * spectra.size(0)
                val_corrects += corrects.item()
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        
        test_loss = 0.0
        test_corrects = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for spectra, targets in tqdm(test_loader, desc='Test'):
                spectra = spectra.to(device)
                targets = targets.to(device)
                outputs = model(spectra)
                loss = criterion(outputs, targets)
                
                _, preds = torch.max(outputs, 1)
                corrects = torch.sum(preds == targets)
                test_loss += loss.item() * spectra.size(0)
                test_corrects += corrects.item()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_corrects / len(test_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epoch}, Train: Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}')
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        precision = precision_score(all_targets, all_preds, average='macro')
        recall = recall_score(all_targets, all_preds, average='macro')
        f1 = f1_score(all_targets, all_preds, average='macro')
        acc = accuracy_score(all_targets, all_preds)
        
        print(f'Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Acc: {acc:.4f}')
        
        if f1 > F1_best:
            Precision_best = precision
            Recall_best = recall
            F1_best = f1
            Acc_best = acc
            torch.save(model.state_dict(), f'best_model_{element}.pth')
    
    print(f"The best Precision: {Precision_best:.4f}, Recall: {Recall_best:.4f}, F1: {F1_best:.4f}, Acc: {Acc_best:.4f}")
        
if __name__ == '__main__':
    dataset = StatesDataset(SpecTensor, TargetsTensor)
    dataset_random_split(dataset)
    train_dataset = Subset(dataset, np.load('train_idx.npy'))
    val_dataset = Subset(dataset, np.load('val_idx.npy'))
    test_dataset = Subset(dataset, np.load('test_idx.npy'))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(100)