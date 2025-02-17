import torch
from pymatgen.core.periodic_table import Element
from models.XANES_XRD_Properties.net import MyNet
from interp import standardized_df

def get_elements_info(compound_elements, max_length):
    """
    Get element information for the given elements.

    Parameters:
    compound_elements (list): List of element symbols.
    max_length (int): Maximum number of elements to consider.

    Returns:
    list: List of element information.
    """
    compound_elements_info = []
    for element in compound_elements:
        compound_z = Element(element).Z  # Atomic number
        compound_x = Element(element).X if Element(element).X is not None else 0  # Electronegativity
        compound_atomic_radius = Element(element).atomic_radius if Element(element).atomic_radius is not None else 0  # Atomic radius
        compound_ionization_energies = Element(element).ionization_energies[:3] + [0]*(3 - len(Element(element).ionization_energies[:3]))  # Ionization energies
        compound_type = []  # Atomic type | 1: Transition metal 2: Lanthanoid/actinoid metal 3: Normal metal 4: Metal-loids 5: Non-metal
        if Element(element).is_metal:
            if element in transition_metals:
                compound_type.append(1)
            elif Element(element).is_lanthanoid or Element(element).is_actinoid:
                compound_type.append(2)
            else:
                compound_type.append(3)
        else:
            if Element(element).is_metalloid:
                compound_type.append(4)
            else:
                compound_type.append(5)
        element_info = [compound_x, compound_atomic_radius] + compound_ionization_energies + [compound_z] + compound_type
        compound_elements_info.append(element_info)
    if len(compound_elements_info) < max_length:
        compound_elements_info += [[0] * 7] * (max_length - len(compound_elements_info))
    return compound_elements_info

# Data
transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                     'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
elements_list_tm = ['Cu']  # Transition metals
elements_list_not_tm = ['In', 'S']  # Non-transition metals

tm_length_max = 3
if len(elements_list_tm) < tm_length_max:
    elements_list_tm += [''] * (tm_length_max - len(elements_list_tm))

not_tm_length_max = 4
if len(elements_list_not_tm) < not_tm_length_max:
    elements_list_not_tm += [''] * (not_tm_length_max - len(elements_list_not_tm))

tm_elements_info = get_elements_info(elements_list_tm, tm_length_max)
not_tm_elements_info = get_elements_info(elements_list_not_tm, not_tm_length_max)

# Input to tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xanes_tensor = torch.tensor(standardized_df['mu'].tolist(), dtype=torch.float32).unsqueeze(0).to(device)
tm_elements_tensor = torch.tensor(tm_elements_info, dtype=torch.float32).unsqueeze(0).to(device)
not_tm_elements_tensor = torch.tensor(not_tm_elements_info, dtype=torch.float32).unsqueeze(0).to(device)

tm_spec_mask = (xanes_tensor.abs().sum(dim=-1) > 0).to(device)
tm_info_mask = (tm_elements_tensor.abs().sum(dim=-1) > 0).to(device)
not_tm_info_mask = (not_tm_elements_tensor.abs().sum(dim=-1) > 0).to(device)


# Regression
model1 = MyNet(num_classes=1).to(device)
ckp_path = ["checkpoints/XANES_XRD/formation.pth",
            "checkpoints/XANES_XRD/bandgap.pth",
            "checkpoints/XANES_XRD/density.pth",
            "checkpoints/XANES_XRD/efermi.pth"]

for i in range(4):
    print("Loading model from: ", ckp_path[i])
    model1.load_state_dict(torch.load(ckp_path[i], map_location=device))
    model1.eval()
    with torch.no_grad():
        outputs, attn_weights = model1(
            xanes_tensor, tm_spec_mask,
            tm_elements_tensor, tm_info_mask,
            not_tm_elements_tensor, not_tm_info_mask,
            return_attn_weights=True
        )
    print(outputs)


# Classification(2)
model2 = MyNet(num_classes=2).to(device)
ckp_path = ["checkpoints/XANES_XRD/isconductor.pth",
            "checkpoints/XANES_XRD/isGapDirect.pth",
            "checkpoints/XANES_XRD/isMagnetic.pth"]

for i in range(3):
    print("Loading model from: ", ckp_path[i])
    model2.load_state_dict(torch.load(ckp_path[i], map_location=device))
    model2.eval()
    with torch.no_grad():
        outputs, attn_weights = model2(
            xanes_tensor, tm_spec_mask,
            tm_elements_tensor, tm_info_mask,
            not_tm_elements_tensor, not_tm_info_mask,
            return_attn_weights=True
        )
    print(torch.max(outputs, 1))
    

# Classification(4)
model3 = MyNet(num_classes=4).to(device)
ckp_path = ["checkpoints/XANES_XRD/ordering.pth"]
print("Loading model from: ", ckp_path[0])
model3.load_state_dict(torch.load(ckp_path[i], map_location=device))
model3.eval()
with torch.no_grad():
    outputs, attn_weights = model2(
        xanes_tensor, tm_spec_mask,
        tm_elements_tensor, tm_info_mask,
        not_tm_elements_tensor, not_tm_info_mask,
        return_attn_weights=True
    )
print(torch.max(outputs, 1))