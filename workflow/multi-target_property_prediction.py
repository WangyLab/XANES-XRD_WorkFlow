import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pymatgen.core.periodic_table import Element
import torch
from net import MyNet
import torch.nn as nn
import torch.nn.functional as F

# Input: an unknown material's XANES spectra (origin)

def InterpolateAndResample(csv_file):
    df = pd.read_csv(csv_file)
    x0 = df['omega'].min()
    x1 = x0 + 56
    x_new = np.linspace(x0, x1, num=200, endpoint=True)
    f = interp1d(df['omega'], df['mu'], kind='cubic', fill_value='extrapolate')
    y_new = f(x_new)
    new_df = pd.DataFrame({'omega': x_new, 'mu': y_new})
    return new_df


def GetTMStandardizeData(csv_files):
    XANES_list = []
    for csv_file in csv_files:
        df_interpolated = InterpolateAndResample(csv_file)
        mean_mu = df_interpolated['mu'].mean()
        std_mu = df_interpolated['mu'].std()
        df_interpolated['mu'] = (df_interpolated['mu'] - mean_mu) / std_mu
        XANES = df_interpolated['mu'].tolist()
        XANES_list.append(XANES)
    return XANES_list


def GetElementsInfo(compound_elements, MaxLenthType):
    transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                     'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
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


def Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list):
    TMLength_max = 3
    if len(XANES_list) <= TMLength_max:
        XANES_list += [[0] *200] * (TMLength_max-len(XANES_list))
    else:
        print("TM elements need to be limited to 3")
    
    TM_info_embedding = GetElementsInfo(TM_elements_list, TMLength_max)
    
    NotTMLength_max = 4
    if len(NotTM_elements_list) < NotTMLength_max:
        NotTM_info_embedding = GetElementsInfo(NotTM_elements_list, NotTMLength_max)
    else:
        print("Non-TM elements need to be limited to 4")
    
    XANES_tensor = torch.tensor(XANES_list, dtype=torch.float32).unsqueeze(0).to(device)
    TM_embedding_tensor = torch.tensor(TM_info_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    Non_TM_embedding_tensor = torch.tensor(NotTM_info_embedding, dtype=torch.float32).unsqueeze(0).to(device)
    
    XANES_mask = (XANES_tensor.abs().sum(dim=-1) > 0).to(device)
    TM_embedding_mask = (TM_embedding_tensor.abs().sum(dim=-1) > 0).to(device)
    Non_TM_embedding_mask = (Non_TM_embedding_tensor.abs().sum(dim=-1) > 0).to(device)
    
    return XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask


def PredictClassification_BandGap(XANES_list, TM_elements_list, NotTM_elements_list):
    XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
    model = MyNet(num_classes=2).to(device)
    ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\isMetal.pth"
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs, attn_weights = model(
            XANES_tensor, XANES_mask,
            TM_embedding_tensor, TM_embedding_mask,
            Non_TM_embedding_tensor, Non_TM_embedding_mask,
            return_attn_weights=True
        )
    if torch.max(outputs, 1)[1] == 0:
        pred = 'No'
        # print('This material has no band gap.')     
    else:
        pred = 'Yes'
        # print('This material has a band gap.')
    return pred


def PredictClassification_isMagnetic(XANES_list, TM_elements_list, NotTM_elements_list):
    XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
    model = MyNet(num_classes=2).to(device)
    ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\isMagnetic.pth"
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs, attn_weights = model(
            XANES_tensor, XANES_mask,
            TM_embedding_tensor, TM_embedding_mask,
            Non_TM_embedding_tensor, Non_TM_embedding_mask,
            return_attn_weights=True
        )
    if torch.max(outputs, 1)[1] == 0:
        pred = 'No'
        # print('This material has no magnetic.')
    else:
        pred = 'Yes'
        # print('This material has magnetic.')
    return pred


def PredictClassification_isGapDirect(XANES_list, TM_elements_list, NotTM_elements_list):
    if PredictClassification_BandGap(XANES_list, TM_elements_list, NotTM_elements_list) == 'Yes':
        XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
        model = MyNet(num_classes=2).to(device)
        ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\isGapDirect.pth"
        model.load_state_dict(torch.load(ckp_path, map_location=device))
        model.eval()
        with torch.no_grad():
            outputs, attn_weights = model(
                XANES_tensor, XANES_mask,
                TM_embedding_tensor, TM_embedding_mask,
                Non_TM_embedding_tensor, Non_TM_embedding_mask,
                return_attn_weights=True
            )
        if torch.max(outputs, 1)[1] == 0:
            pred = 'No'
            # print('This material has no GapDirect.')    
        else:
            pred = 'Yes'
            # print('This material has GapDirect.') 
    return pred


def PredictClassification_MagneticOrder(XANES_list, TM_elements_list, NotTM_elements_list):
    if PredictClassification_isMagnetic(XANES_list, TM_elements_list, NotTM_elements_list) == 'Yes':
        XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
        model = MyNet(num_classes=3).to(device)
        ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\MagneticOrder.pth"
        model.load_state_dict(torch.load(ckp_path, map_location=device))
        model.eval()
        with torch.no_grad():
            outputs, attn_weights = model(
                XANES_tensor, XANES_mask,
                TM_embedding_tensor, TM_embedding_mask,
                Non_TM_embedding_tensor, Non_TM_embedding_mask,
                return_attn_weights=True
            )
        if torch.max(outputs, 1)[1] == 0:
            pred = 'AFM'
        elif torch.max(outputs, 1)[1] == 1:
            pred = 'FM'
        elif torch.max(outputs, 1)[1] == 2:
            pred = 'FiM'
    return pred


def PredictRegression(XANES_list, TM_elements_list, NotTM_elements_list):
    XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
    model = MyNet(num_classes=1).to(device)
    
    ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\Ef.pth"
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs_Ef, attn_weights = model(
            XANES_tensor, XANES_mask,
            TM_embedding_tensor, TM_embedding_mask,
            Non_TM_embedding_tensor, Non_TM_embedding_mask,
            return_attn_weights=True
        )
    
    ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\Efermi.pth"
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs_Efermi, attn_weights = model(
            XANES_tensor, XANES_mask,
            TM_embedding_tensor, TM_embedding_mask,
            Non_TM_embedding_tensor, Non_TM_embedding_mask,
            return_attn_weights=True
        )
    
    ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\density.pth"
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    model.eval()
    with torch.no_grad():
        outputs_density, attn_weights = model(
            XANES_tensor, XANES_mask,
            TM_embedding_tensor, TM_embedding_mask,
            Non_TM_embedding_tensor, Non_TM_embedding_mask,
            return_attn_weights=True
        )
    return outputs_Ef.item(), outputs_Efermi.item(), outputs_density.item()
    
def PredictRegression_BandGap(XANES_list, TM_elements_list, NotTM_elements_list):
    if PredictClassification_BandGap(XANES_list, TM_elements_list, NotTM_elements_list) == 'Yes':
        XANES_tensor, TM_embedding_tensor, Non_TM_embedding_tensor, XANES_mask, TM_embedding_mask, Non_TM_embedding_mask = Preprocessing(XANES_list, TM_elements_list, NotTM_elements_list)
        model = MyNet(num_classes=1).to(device)
        ckp_path = "C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\SpecFusionNet\\TM\\Eg.pth"
        model.load_state_dict(torch.load(ckp_path, map_location=device))
        model.eval()
        with torch.no_grad():
            outputs_Eg, attn_weights = model(
                XANES_tensor, XANES_mask,
                TM_embedding_tensor, TM_embedding_mask,
                Non_TM_embedding_tensor, Non_TM_embedding_mask,
                return_attn_weights=True
            )
    return outputs_Eg.item()


class Spec2CN_States(nn.Module):
    def __init__(self, out_dim):
        super(Spec2CN_States, self).__init__()
        self.dense1 = nn.Linear(200, 512)
        self.dense2 = nn.Linear(512, out_dim)
    
    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


def Predict_TM_CNs(XANES_list, TM_elements_list):
    cns_classes = {'4':0, '5':1, '6':2}
    
    if len(TM_elements_list) == 1:
        model = Spec2CN_States(out_dim=3)
        model.load_state_dict(torch.load(f"C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\CNs\\{TM_elements_list[0]}.pth"))
        model.eval()
        outputs = model(torch.tensor(XANES_list[0]))
        predicted_indices = torch.max(outputs, 0)[1].item()
        reverse_cns_classes = {v: k for k, v in cns_classes.items()}
        predicted_cns = reverse_cns_classes[predicted_indices]
        return[int(predicted_cns)]
    
    if len(TM_elements_list) > 1:
        CNs_list = []
        for i in range(len(TM_elements_list)):
            model = Spec2CN_States(out_dim=3)
            model.load_state_dict(torch.load(f"C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\CNs\\{TM_elements_list[i]}.pth"))
            model.eval()
            outputs = model(torch.tensor(XANES_list[i]))
            predicted_indices = torch.max(outputs, 0)[1].item()
            reverse_cns_classes = {v: k for k, v in cns_classes.items()}
            predicted_cns = reverse_cns_classes[predicted_indices]
            CNs_list.append(int(predicted_cns))
        return CNs_list
        

def Predict_Oxistates(XANES_list, TM_elements_list, Non_TM_elements_list):
    element_classes = {
    'Sc': {'3':0, '2':1},
    'Ti': {'3':0, '4':1, '2':2}, 'V': {'3':0, '2':1, '4':2, '5':3}, 'Cr': {'3':0, '6':1, '4':2, '2':3},
    'Mn': {'3':0, '4':1, '2':2}, 'Fe': {'2':0, '3':1}, 'Co': {'2':0, '3':1}, 'Ni': {'3':0, '2':1},
    'Cu': {'1':0, '2':1, '3': 3},
    'Y': {'3':0, '2':1}, 'Zr': {'4':0, '2':1},
    'Nb': {'5':0, '4':1}, 'Mo': {'6':0, '3':1, '4':2}, 'Pd': {'2':0, '4':1},
    'Ag': {'1':0, '2':1}}

    non_multi_state_elements = {
        'H': 1, 'Li': 1, 'Be': 2, 'B':3, 'C': [-4, 2, 4], 'N': [-3, 3, 5], 'O':-2, 'F':-1,
        'Na':1, 'Mg':2, 'Al':3, 'Si':4, 'P': [-3, 3, 5], 'S': -2, 'Cl': [-1, 1, 3, 5, 7],
        'K':1, 'Ca':2, 'Zn': 2, 'Ga': 3, 'Ge': [2, 4], 'As': [-3, 3, 5], 'Se': [-2, 4, 6], 'Br': [-1, 1, 3, 5, 7],
        'Rb': 1, 'Sr': 2, 'Cd': 2, 'In': 3, 'Sn': 4, 'Sb':[3, 5], 'Te': [-2, 4, 6], 'I': [-1, 1, 3, 5, 7],
        'Cs': 1, 'Ba': 2, 'Tl': [1, 3], 'Pb': [2, 4], 'Bi': [3, 5], 'Po': [2, 4], 'At': [1, 3, 5, 7],
        'Fr': 1, 'Ra': 2
    }
    
    TMstates_list = []
    for i in range(len(TM_elements_list)):
        if TM_elements_list[i] in element_classes:
            model = Spec2CN_States(out_dim=len(element_classes[TM_elements_list[i]]))
            model.load_state_dict(torch.load(f"C:\\Users\\wangy\\Desktop\\XANES课题\\git_code\\checkpoints\\OxidationStates\\{TM_elements_list[i]}.pth"))
            model.eval()
            outputs = model(torch.tensor(XANES_list[i]))
            predicted_indices = torch.max(outputs, 0)[1].item()
            reverse_element_classes = {v: k for k, v in element_classes[TM_elements_list[i]].items()}
            predicted_oxidation_state = reverse_element_classes[predicted_indices]
            TMstates_list.append(int(predicted_oxidation_state))
        else:
            TMstates_list.append(non_multi_state_elements[TM_elements_list[i]])
    
    Non_TMstates_list = []
    for i in range(len(Non_TM_elements_list)):
        if Non_TM_elements_list[i] in non_multi_state_elements:
            Non_TMstates_list.append(non_multi_state_elements[Non_TM_elements_list[i]])
    
    return TMstates_list, Non_TMstates_list



if __name__ == '__main__':
    # input area
    TM_elements = ['Cu', 'Zn']
    Non_TM_elements = ['Sn', 'S']
    csv_file_Cu = "C:\\Users\\wangy\\Desktop\\FEFF_test_inp\\Zn2Cu4Sn2S8_mp_1025500-ok\\data1.csv"
    csv_file_Zn = "C:\\Users\\wangy\\Desktop\\FEFF_test_inp\\Zn2Cu4Sn2S8_mp_1025500-ok\\data2.csv"
    
    # preprocessing area
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    csv_files = [csv_file_Cu, csv_file_Zn]
    XANES_list = GetTMStandardizeData(csv_files)
    
    # properties_prediction area
    bandgap_cls = PredictClassification_BandGap(XANES_list, TM_elements, Non_TM_elements)
    magnetic_cls = PredictClassification_isMagnetic(XANES_list, TM_elements, Non_TM_elements)
    if bandgap_cls == 'Yes':
        gapdirect_cls = PredictClassification_isGapDirect(XANES_list, TM_elements, Non_TM_elements)
        Eg = PredictRegression_BandGap(XANES_list, TM_elements, Non_TM_elements)
    else:
        gapdirect_cls = 'No Band Gap'
        Eg = 'No Band Gap'
        
    if magnetic_cls == 'Yes':
        magnetic_order_cls = PredictClassification_MagneticOrder(XANES_list, TM_elements, Non_TM_elements)
    else:
        magnetic_order_cls = 'No Magnetic'
    Ef, Efermi, density = PredictRegression(XANES_list, TM_elements, Non_TM_elements)
    
    # CNs&States prediction area
    TM_CNs_list = Predict_TM_CNs(XANES_list, TM_elements)
    TM_States_list, Non_TM_States_list = Predict_Oxistates(XANES_list, TM_elements, Non_TM_elements)
    
    print(f'This material has band gap: {bandgap_cls}, has magnetic: {magnetic_cls}, has gapdirect: {gapdirect_cls}, magnetic order: {magnetic_order_cls}')
    print(f'Ef: {Ef}, Efermi: {Efermi}, Eg: {Eg}, density: {density}')
    print(f'TM: {TM_elements}, CNs: {TM_CNs_list}, States: {TM_States_list}')
    print(f'Non-TM: {Non_TM_elements}, States: {Non_TM_States_list}')
    