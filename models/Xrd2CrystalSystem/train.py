import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader_xrd import SortedXRD_Dataset
from cnn_transformer import CNNTransformerClassifier
from dataset_random_split import dataset_random_split

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs):
    best_val_f1 = float('0.2')
    best_model_path = './best_models.pth'
    for epoch in range(epochs):
        model.train()
        all_preds = []
        all_targets = []
        
        # Training
        for xrd, targets in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}'):
            targets, xrd = targets.to(device), xrd.to(device)
            optimizer.zero_grad()
            outputs = model(xrd)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
                        
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Val
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xrd, targets in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}'):
                targets, xrd = targets.to(device), xrd.to(device)
                outputs = model(xrd)
                loss = criterion(outputs, targets)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
            val_acc = accuracy_score(all_targets, all_preds)
            val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            print(f'Val Epoch {epoch+1}/{epochs}, Val Acc: {val_acc:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with val f1: {best_val_f1:.4f}')
            
    # Test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    all_preds = []
    all_targets = []
    for xrd, targets in tqdm(test_loader, desc=f'Val Epoch {epoch+1}/{epochs}'):
        targets, xrd = targets.to(device), xrd.to(device)
        outputs = model(xrd)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    test_acc = accuracy_score(all_targets, all_preds)
    test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    test_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    test_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    print(f'Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall: .4f}, Test F1: {test_f1: .4f}')
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SortedXRD_Dataset(xrd_csv_path='xrd_data.csv', json_path='data.json')
    train_loader, val_loader, test_loader = dataset_random_split('.', dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=True)
    model = CNNTransformerClassifier(in_channels=1, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=100
    )