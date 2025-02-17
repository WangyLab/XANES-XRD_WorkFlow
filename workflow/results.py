import torch
from pymatgen.core.periodic_table import Element
import sys
import os
import torch.nn.functional as F
import torch.nn as nn
from interp import standardized_df
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(project_root)
from models.XANES_XRD_Properties.net import MyNet



'''Input'''
XANES_list = [standardized_df['mu'].tolist()]
Elements_list_TM = ['Cu']
Elements_list_notTM = ['In', 'S']


'''Predict CNs'''
class Spec2CN(nn.Module):
    def __init__(self):
        super(Spec2CN, self).__init__()
        self.dense1 = nn.Linear(200, 512)
        self.dense2 = nn.Linear(512, 3)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
cns_classes = {
   '4':0, '5':1, '6':2 
}
model_cns = Spec2CN()
model_cns.load_state_dict(torch.load('checkpoints/CNs/Cu.pth'))
model_cns.eval()
outputs = model_cns(torch.tensor(XANES_list))
predicted_indices = torch.max(outputs, 1)[1]
reverse_cns_classes = {v: k for k, v in cns_classes.items()}
predicted_cns = [reverse_cns_classes[idx.item()] for idx in predicted_indices]
print(f'Predict {Elements_list_TM[0]} CNs: ', predicted_cns[0])


'''Predict Oxistates'''
element_classes = {
    'Sc': {'3+':0, '2+':1},
    'Ti': {'3+':0, '4+':1, '2+':2}, 'V': {'3+':0, '2+':1, '4+':2, '5+':3}, 'Cr': {'3+':0, '6+':1, '4+':2, '2+':3},
    'Mn': {'3+':0, '4+':1, '2+':2}, 'Fe': {'2+':0, '3+':1}, 'Co': {'2+':0, '3+':1}, 'Ni': {'3+':0, '2+':1},
    'Cu': {'+':0, '2+':1, '3+': 3},
    'Y': {'3+':0, '2+':1}, 'Zr': {'4+':0, '2+':1},
    'Nb': {'5+':0, '4+':1}, 'Mo': {'6+':0, '3+':1, '4+':2}, 'Pd': {'2+':0, '4+':1},
    'Ag': {'+':0, '2+':1}
}
class Spec2States(nn.Module):
    def __init__(self):
        super(Spec2States, self).__init__()
        self.dense1 = nn.Linear(200, 512)
        self.dense2 = nn.Linear(512, 3)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
model_states = Spec2States()
model_states.load_state_dict(torch.load('checkpoints/OxidationStates/Cu.pth'))
model_states.eval()
outputs = model_states(torch.tensor(XANES_list))
predicted_indices = torch.max(outputs, 1)[1]
reverse_element_classes = {v: k for k, v in element_classes[Elements_list_TM[0]].items()}
predicted_states = [reverse_element_classes[idx.item()] for idx in predicted_indices]
print(f'Predict {Elements_list_TM[0]} Oxistates: ', predicted_states[0])




'''Padding | Input2Tensor'''
TMLength_max = 3
if len(XANES_list) < TMLength_max:
    XANES_list += [[0]*200] * (TMLength_max-len(XANES_list))

transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                     'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']

def GetElementsInfo(compound_elements, MaxLenthType):
    compound_elements_info = []
    for element in compound_elements:
        compound_z = Element(element).Z  # int
        compound_x = Element(element).X if Element(element).X is not None else 0  # float
        compound_atomic_radius = Element(element).atomic_radius if Element(element).atomic_radius is not None else 0  # float
        compound_ionization_energies = Element(element).ionization_energies[:3] + [0]*(3 - len(Element(element).ionization_energies[:3]))  # float
        compound_type = []  # Atomic type | 1: Transition metal 2: Lanthanoid/actinoid metal 3: Normal metal 4: Metal-loids 5: Non-metal | int
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
    if len(compound_elements_info) < MaxLenthType:
        compound_elements_info += [[0] *7] * (MaxLenthType-len(compound_elements_info))
    return compound_elements_info

TMElements_info = GetElementsInfo(Elements_list_TM, TMLength_max)
NotTMLength_max = 4
NotTMElements_info = GetElementsInfo(Elements_list_notTM, NotTMLength_max)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XANES_tensor = torch.tensor(XANES_list, dtype=torch.float32).unsqueeze(0).to(device)
TMElements_tensor = torch.tensor(TMElements_info, dtype=torch.float32).unsqueeze(0).to(device)
NotTMElements_tensor = torch.tensor(NotTMElements_info, dtype=torch.float32).unsqueeze(0).to(device)

TM_spec_mask = (XANES_tensor.abs().sum(dim=-1) > 0).to(device)
TM_info_mask = (TMElements_tensor.abs().sum(dim=-1) > 0).to(device)
NotTM_info_mask = (NotTMElements_tensor.abs().sum(dim=-1) > 0).to(device)


'''Classification(2)'''
model1 = MyNet(num_classes=2).to(device)
ckp_path = ["checkpoints/XANES_XRD/isconductor.pth",
            "checkpoints/XANES_XRD/isGapDirect.pth",
            "checkpoints/XANES_XRD/isMagnetic.pth"]

for i in range(3):
    model1.load_state_dict(torch.load(ckp_path[i], map_location=device))
    model1.eval()
    with torch.no_grad():
        outputs, attn_weights = model1(
            XANES_tensor, TM_spec_mask,
            TMElements_tensor, TM_info_mask,
            NotTMElements_tensor, NotTM_info_mask,
            return_attn_weights=True
        )
    if i == 0:
        if torch.max(outputs, 1)[1] == 0:
            print('yes')
        else:
            print('no')
    else:
        if torch.max(outputs, 1)[1] == 0:
            print('no')
        else:
            print('yes')
            

'''Classification(4)'''
model2 = MyNet(num_classes=4).to(device)
model2.load_state_dict(torch.load("checkpoints/XANES_XRD/ordering.pth", map_location=device))
model2.eval()
with torch.no_grad():
    outputs, attn_weights = model2(
        XANES_tensor, TM_spec_mask,
        TMElements_tensor, TM_info_mask,
        NotTMElements_tensor, NotTM_info_mask,
        return_attn_weights=True
    )
    if torch.max(outputs, 1)[1] == 0:
        print('AFM')
    elif torch.max(outputs, 1)[1] == 1:
        print('FM')
    elif torch.max(outputs, 1)[1] == 2:
        print('FiM')
    else:
        print('NM')
        

'''Regression'''
model3 = MyNet(num_classes=1).to(device)
ckp_path = ["checkpoints/XANES_XRD/formation.pth",
            "checkpoints/XANES_XRD/bandgap.pth",
            "checkpoints/XANES_XRD/density.pth",
            "checkpoints/XANES_XRD/efermi.pth"]

for i in range(4):
    print("Loading model from: ", ckp_path[i])
    model3.load_state_dict(torch.load(ckp_path[i], map_location=device))
    model3.eval()
    with torch.no_grad():
        outputs, attn_weights = model3(
            XANES_tensor, TM_spec_mask,
            TMElements_tensor, TM_info_mask,
            NotTMElements_tensor, NotTM_info_mask,
            return_attn_weights=True
        )
    print(outputs)