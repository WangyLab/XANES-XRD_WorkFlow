from models.XANES_XRD_Properties.data_loader import load_and_filter_data, preprocess_data, MyDataset
from models.XANES_XRD_Properties.dataset_random_split import dataset_random_split
from models.XANES_XRD_Properties.net import MyNet
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def feature_dim_ablation(model, test_loader, which="TM"):
    """
    Demonstrate how to mask a specific feature dimension of TM elements or NotTM elements,
    and observe its impact on the final prediction.
    """
    model.eval()
    all_ySpec, all_TM, all_NotTM, all_targets = [], [], [], []
    all_ySpec_mask, all_TM_mask, all_NotTM_mask = [], [], []
    for ySpec, TMElements, NotTMElements, targets, ySpec_mask, TMElements_mask, NotTMElements_mask in test_loader:
        all_ySpec.append(ySpec)
        all_TM.append(TMElements)
        all_NotTM.append(NotTMElements)
        all_targets.append(targets)
        all_ySpec_mask.append(ySpec_mask)
        all_TM_mask.append(TMElements_mask)
        all_NotTM_mask.append(NotTMElements_mask)

    ySpec = torch.cat(all_ySpec, dim=0).to(device)
    TMinfo = torch.cat(all_TM, dim=0).to(device)
    NotTMinfo = torch.cat(all_NotTM, dim=0).to(device)
    targets = torch.cat(all_targets, dim=0).to(device)
    ySpec_mask = torch.cat(all_ySpec_mask, dim=0).to(device)
    TM_mask = torch.cat(all_TM_mask, dim=0).to(device)
    NotTM_mask = torch.cat(all_NotTM_mask, dim=0).to(device)

    # baseline performance
    with torch.no_grad():
        outputs = model(ySpec, ySpec_mask, TMinfo, TM_mask, NotTMinfo, NotTM_mask)
    base_preds = outputs.squeeze()
    base_mae = mean_squared_error(targets.cpu(), base_preds.cpu())
    print(f"Baseline MAE = {base_mae:.4f}")

    if which == "TM":
        num_feats = TMinfo.shape[-1]
    else:
        num_feats = NotTMinfo.shape[-1]

    for d in range(num_feats):
        ySpec_copy = ySpec.clone()
        TM_copy = TMinfo.clone()
        NotTM_copy = NotTMinfo.clone()

        if which == "TM":
            TM_copy[:,:,d] = 0.0
        else:
            NotTM_copy[:,:,d] = 0.0

        with torch.no_grad():
            new_out = model(ySpec_copy, ySpec_mask, TM_copy, TM_mask, NotTM_copy, NotTM_mask).squeeze()
        new_mae = mean_squared_error(targets.cpu(), new_out.cpu())
        print(f"Feature dim {d} masked => RMSE {new_mae:.4f}, Δ={new_mae-base_mae:.4f}")


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
    checkpoint_path = "checkpoints/XANES_XRD/formation.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    feature_dim_ablation(model, test_loader, which="NotTM")  # or "TM"