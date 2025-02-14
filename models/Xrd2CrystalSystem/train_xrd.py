from data_loader_xrd import prepare_dataset, dataset_random_split
from net_lstm import LSTMModel
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs):
    best_f1 = float('0.1')
    best_model_path = f'best_models_{best_f1:.4f}.pth'
    
    for epoch in range(epochs):
        # Training
        model.train()
        all_preds = []
        all_targets = []
        for sequences, lengths, labels in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}'):
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.cpu()
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        # Val
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for sequences, lengths, labels in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}'):
                sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
            val_acc = accuracy_score(all_targets, all_preds)
            val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            print(f'Val Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with val f1: {best_f1:.4f}')
    
    # Test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for sequences, lengths, labels in tqdm(test_loader, desc=f'Test Epoch {epoch+1}/{epochs}'):
            sequences, lengths, labels = sequences.to(device), lengths.cpu(), labels.to(device)
            outputs = model(sequences, lengths)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
        
        test_acc = accuracy_score(all_targets, all_preds)
        test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        test_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        test_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        print(f'Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

if __name__ == '__main__':
    dataset = prepare_dataset('xrd.csv', 'data_moreProperties.json') 
    train_loader, val_loader, test_loader = dataset_random_split(dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_dim=1, d_model=256, num_layers=6, dropout=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = FocalLoss(gamma=2, alpha=0.25)
    train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs=100)
