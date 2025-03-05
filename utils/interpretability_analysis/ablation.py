from models.SpecFusionNet.TransitionMetals.data_loader import load_and_filter_data, preprocess_data, MyDataset
from models.SpecFusionNet.TransitionMetals.dataset_random_split import dataset_random_split
from models.SpecFusionNet.TransitionMetals.net import MyNet
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import math
        
def mask_module_inference(model, test_loader, mask_xanes=False, mask_nottm=False):            
    model.load_state_dict(torch.load("checkpoints\SpecFusionNet\TM\Ef.pth"))
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for ySpec, TMElements, NotTMElements, targets, ySpec_mask, TMElements_mask, NotTMElements_mask in tqdm(test_loader, desc=f'Test'):
            ySpec, TMElements, NotTMElements, targets = ySpec.to(device), TMElements.to(device), NotTMElements.to(device), targets.to(device)
            ySpec_mask, TMElements_mask, NotTMElements_mask = ySpec_mask.to(device), TMElements_mask.to(device), NotTMElements_mask.to(device)
            
            if mask_xanes:
                ySpec = torch.zeros_like(ySpec).to(device)
                TMElements = torch.zeros_like(TMElements).to(device)
                
            if mask_nottm:
                NotTMElements = torch.zeros_like(NotTMElements).to(device)
            
            outputs, attn_weights = model(ySpec, ySpec_mask, TMElements, TMElements_mask, NotTMElements, NotTMElements_mask, return_attn_weights=True)
            test_preds.append(outputs.squeeze().float().cpu().numpy())
            test_targets.append(targets.float().cpu().numpy())

    test_preds = np.concatenate(test_preds, axis=0).flatten()
    test_targets = np.concatenate(test_targets, axis=0).flatten()
    test_rmse = math.sqrt(mean_squared_error(test_targets, test_preds))
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)
    return test_rmse, test_mae, test_r2


def evaluate_module_importance(model, test_loader):
    """
    Ablation experiments were performed on the three modules respectively.
    """
    # 1) Baseline with all modules enabled
    base_rmse, base_mae, base_r2 = mask_module_inference(model, test_loader, False, False)
    print(f"[Baseline] RMSE={base_rmse:.4f}, MAE={base_mae:.4f}, R2={base_r2:.4f}")

    # 2) Mask TM Fused module
    rx, mx, r2x = mask_module_inference(model, test_loader, True, False)
    print(f"[Mask TM] RMSE={rx:.4f}, MAE={mx:.4f}, R2={r2x:.4f}")

    # 4) Mask Non-TM Fused module
    rn, mn, r2n = mask_module_inference(model, test_loader, False, True)
    print(f"[Mask NonTM] RMSE={rn:.4f}, MAE={mn:.4f}, R2={r2n:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_and_filter_data('data.json')
    target_name = 'E_Formation'
    TMElements_info, NotTMElements_info, padded_ySpec, targets, TMLength_max, NotTMLength_max = preprocess_data(df, target_name)
    dataset = MyDataset(TMElements_info, NotTMElements_info, padded_ySpec, targets)
    train_loader, val_loader, test_loader = dataset_random_split(".", dataset, train_size=0.7, val_size=0.15, test_size=0.15, new_split=False)
    model = MyNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.L1Loss()
    evaluate_module_importance(model, test_loader)