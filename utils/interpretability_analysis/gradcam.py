from models.SpecFusionNet.TransitionMetals.data_loader import load_and_filter_data, preprocess_data, MyDataset
from models.SpecFusionNet.TransitionMetals.dataset_random_split import dataset_random_split
from models.SpecFusionNet.TransitionMetals.net import MyNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

class GradCamHook:
    def __init__(self):
        self.forward_output = None
        self.backward_output = None
    
    def forward_hook(self, module, input, output):
        self.forward_output = output

    def backward_hook(self, module, grad_input, grad_output):
        self.backward_output = grad_output[0]


def register_gradcam_hooks(layer):
    hook_obj = GradCamHook()
    fwd_handle = layer.register_forward_hook(hook_obj.forward_hook)
    bwd_handle = layer.register_backward_hook(hook_obj.backward_hook)
    return hook_obj, fwd_handle, bwd_handle


def compute_1d_gradcam(model, 
                       ySpec_single, 
                       ySpec_mask_single,
                       TMElements_single,
                       NotTMElements_single,
                       TMElements_mask_single,
                       NotTMElements_mask_single,
                       target_layer):
    model.eval()

    # 1) Register hooks
    hook_obj, fwd_handle, bwd_handle = register_gradcam_hooks(target_layer)

    # 2) Forward pass
    output = model(
        TM_spec=ySpec_single,
        TM_spec_mask=ySpec_mask_single,
        TM_info=TMElements_single,
        TM_mask=TMElements_mask_single,
        NotTM_info=NotTMElements_single,
        NotTM_mask=NotTMElements_mask_single
    )
    scalar_out = output[0,0]

    # 3) Backward pass
    model.zero_grad()
    scalar_out.backward()

    # 4) Get activations and gradients
    activations = hook_obj.forward_output
    grads = hook_obj.backward_output

    B, C, L_prime = activations.shape
    
    gradcam_maps = []
    for i in range(B):
        alpha = grads[i].mean(dim=1, keepdim=True)
        gradcam_1d = F.relu((alpha * activations[i]).sum(dim=0))
        
        gradcam_1d = gradcam_1d / (gradcam_1d.max() + 1e-8)
        gradcam_1d = gradcam_1d.unsqueeze(0).unsqueeze(0)
        gradcam_1d = F.interpolate(gradcam_1d, size=200, mode='linear', align_corners=False)
        gradcam_1d = gradcam_1d.squeeze().detach().cpu().numpy()

        gradcam_maps.append(gradcam_1d)

    fwd_handle.remove()
    bwd_handle.remove()

    return gradcam_maps


def gradcam_demo(model, test_loader, a):
    batch = next(iter(test_loader))
    (ySpec, TMElements, NotTMElements, targets, 
     ySpec_mask, TMElements_mask, NotTMElements_mask) = batch

    single_ySpec = ySpec[a:a+1].to(device)
    single_ySpec_mask = ySpec_mask[a:a+1].to(device)
    single_TMElements = TMElements[a:a+1].to(device)
    single_TMElements_mask = TMElements_mask[a:a+1].to(device)
    single_NotTMElements = NotTMElements[a:a+1].to(device)
    single_NotTMElements_mask = NotTMElements_mask[a:a+1].to(device)

    target_layer = model.xanes_extractor.branch1_conv2
    
    gradcam_maps = compute_1d_gradcam(
        model, 
        single_ySpec, single_ySpec_mask,
        single_TMElements, single_NotTMElements,
        single_TMElements_mask, single_NotTMElements_mask,
        target_layer=target_layer
    )

    plt.figure(figsize=(10, 5))
    for i, gradcam_map in enumerate(gradcam_maps):
        plt.plot(gradcam_map, label=f"Grad-CAM for TM {i+1}")
    
    plt.xlabel("Energy index")
    plt.ylabel("Importance")
    plt.title("Grad-CAM for multiple TM elements")
    plt.legend()
    plt.show()


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
    gradcam_demo(model, test_loader, 0)   # a = any index