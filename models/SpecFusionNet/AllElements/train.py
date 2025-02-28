from data_loader import load_and_filter_data, preprocess_data, MyDataset
from dataset_random_split import dataset_random_split
from net import MyNet
from tqdm import tqdm
import math
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt


def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, epochs, target_name):
    best_rmse = float('5')
    best_model_path = f'best_{target_name}.pth'
    
    for epoch in range(epochs):
        # Training
        model.train()
        all_preds = []
        all_targets = []
        for ySpec, Elements, targets, ySpec_mask, Elements_mask in tqdm(train_loader, desc=f'Train Epoch {epoch+1}/{epochs}'):
            ySpec, Elements, targets = ySpec.to(device), Elements.to(device), targets.to(device)
            ySpec_mask, Elements_mask = ySpec_mask.to(device), Elements_mask.to(device)
            
            optimizer.zero_grad()
            outputs = model(ySpec, ySpec_mask, Elements, Elements_mask)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            all_preds.append(outputs.detach().squeeze().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
        
        # Val
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for ySpec, Elements, targets, ySpec_mask, Elements_mask in tqdm(val_loader, desc=f'Val Epoch {epoch+1}/{epochs}'):
                ySpec, Elements, targets = ySpec.to(device), Elements.to(device), targets.to(device)
                ySpec_mask, Elements_mask = ySpec_mask.to(device), Elements_mask.to(device)
            
                outputs = model(ySpec, ySpec_mask, Elements, Elements_mask)
                loss = criterion(outputs.squeeze(), targets)
                val_preds.append(outputs.squeeze().float().cpu().numpy())
                val_targets.append(targets.float().cpu().numpy())
        
        val_preds = np.concatenate(val_preds, axis=0).flatten()
        val_targets = np.concatenate(val_targets, axis=0).flatten()
        rmse = math.sqrt(mean_squared_error(val_targets, val_preds))
        mae = mean_absolute_error(val_targets, val_preds)
        r2 = r2_score(val_targets, val_preds)
        print(f'Val RMSE: {rmse:.4f}, Val MAE: {mae:.4f}, Val R2: {r2:.4f}')
            
        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_r2 = r2
            torch.save(model.state_dict(), best_model_path)
            print(f'Best RMSE: {best_rmse:.4f}, Best MAE: {best_mae:.4f}, Best R2: {best_r2:.4f}')
            
    # Test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for ySpec, Elements, targets, ySpec_mask, Elements_mask in tqdm(test_loader, desc=f'Test'):
            ySpec, Elements, targets = ySpec.to(device), Elements.to(device), targets.to(device)
            ySpec_mask, Elements_mask = ySpec_mask.to(device), Elements_mask.to(device)
            
            outputs = model(ySpec, ySpec_mask, Elements, Elements_mask)
            test_preds.append(outputs.squeeze().float().cpu().numpy())
            test_targets.append(targets.float().cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0).flatten()
    test_targets = np.concatenate(test_targets, axis=0).flatten()
    test_rmse = math.sqrt(mean_squared_error(test_targets, test_preds))
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    print(f'Test RMSE: {test_rmse:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')


    flat_preds = test_preds.tolist()
    flat_targets = test_targets.tolist()
    plt.figure(figsize=(8, 6))
    plt.scatter(flat_targets, flat_preds, alpha=0.5)
    plt.plot([min(flat_targets), max(flat_targets)], [min(flat_targets), max(flat_targets)], 'k--', lw=2)
    plt.xlabel(f'Actual {target_name}')
    plt.ylabel(f'Predicted {target_name}')
    plt.title(f'Actual vs. Predicted {target_name}')
    plt.savefig(f'Predict_{target_name}.png')


if __name__ == '__main__':
    '''
    Regression: 'E_Formation', 'efermi', 'Density', 'BandGap' (control bandgap>0.5).
    Classification: 'BandGap' (conductor: bandgap=0, non-conductor: bandgap>0), 'isGapDirect', 'isMagnetic', 'Ordering'.
    In the classification task, the first three are two-category classification, and the last one is four-catefory.
    
    If you want to train for classification tasks, you need to change:
    1. data_loader, make the target index mapping
    2. net, change the out_dim of FinalRegressor
    3. train_model, change the evaluation index
    '''
    
    df = load_and_filter_data('D:/OneDrive - USTC/Code/XAS/9-25_StrucInfo/data_moreProperties.json')
    target_name = 'E_Formation'
    Elements_info, padded_ySpec, targets, Length_max = preprocess_data(df, target_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MyDataset(Elements_info, padded_ySpec, targets)
    train_loader, val_loader, test_loader = dataset_random_split(".", dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=True)
    model = MyNet(num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.001)
    train_model(model, train_loader, val_loader, test_loader, torch.nn.L1Loss(), optimizer, 100, target_name)  # regression: torch.nn.L1Loss(), classification: torch.nn.CrossEntropyLoss()
    