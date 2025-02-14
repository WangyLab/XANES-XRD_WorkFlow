import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data_josn_path = 'Cr_pure.json'
df_pure = pd.read_json(data_josn_path)
ySpecTensor_pure = torch.tensor(df_pure['input'])
TargetsTensor_pure = torch.tensor(df_pure['targets'])

class PureDataset(Dataset):
    def __init__(self, specs, targets):
        assert len(specs) == len(targets)
        self.targets = targets
        self.specs = specs
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        spec = self.specs[idx]
        return target, spec

'''Net'''
class Spec2CN(nn.Module):
    def __init__(self):
        super(Spec2CN, self).__init__()
        self.dense1 = nn.Linear(200, 512)
        self.dense2 = nn.Linear(512, 3)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x

def train(model, train_loader, val_loader, test_loader, criterion_pure, optimizer, epochs):
    best_acc = float('0')
    best_model_path = f'Cr_{best_acc}.pth'
    
    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        for targets, ySpec in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}'):
            targets, ySpec = targets.to(device), ySpec.to(device)
            optimizer.zero_grad()
            outputs = model(ySpec)
            loss = criterion_pure(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        all_targets = np.concatenate(all_targets).ravel()
        all_preds = np.concatenate(all_preds).ravel()
        train_acc = accuracy_score(all_targets, all_preds)
        train_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        train_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        train_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        print(f'Train Acc: {train_acc:.4f}, Train Precision: {train_precision:.4f}, Train F1: {train_f1:.4f}, Train Recall: {train_recall:.4f}')
        
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for targets, ySpec in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}'):
                targets, ySpec = targets.to(device), ySpec.to(device)
                
                outputs = model(ySpec)
                _, preds = torch.max(outputs, 1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            
            all_targets = np.concatenate(all_targets).ravel()
            all_preds = np.concatenate(all_preds).ravel()
            val_acc = accuracy_score(all_targets, all_preds)
            val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            print(f'Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
        if val_acc > best_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with val Acc: {best_val_acc:.4f}')
    
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []              
    all_targets = []
    for targets, ySpec in tqdm(test_loader, desc='Testing'):
        targets, ySpec = targets.to(device), ySpec.to(device)
        outputs = model(ySpec)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds)
        all_targets.append(targets)
    
    all_targets = np.concatenate([target.cpu().numpy() for target in all_targets]).ravel()
    all_preds = np.concatenate([pred.detach().cpu().numpy() for pred in all_preds]).ravel()
    test_acc = accuracy_score(all_targets, all_preds)
    test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    print(f'Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')


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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_pure = PureDataset(ySpecTensor_pure, TargetsTensor_pure)
    dataset_random_split(dataset_pure)
    train_loader = Subset(dataset_pure, np.load('train_idx.npy'))
    val_dataset = Subset(dataset_pure, np.load('val_idx.npy'))
    test_dataset = Subset(dataset_pure, np.load('test_idx.npy'))
    
    train_loader = DataLoader(train_loader, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model = Spec2CN().to(device)
    train(model, train_loader, val_loader, test_loader, criterion_pure=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters(), lr=0.003), epochs=150)