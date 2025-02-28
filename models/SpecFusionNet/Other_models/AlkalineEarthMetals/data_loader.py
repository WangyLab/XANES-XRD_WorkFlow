import pandas as pd
import ast
from torch.utils.data import Dataset
from pymatgen.core.periodic_table import Element
import torch
import numpy as np

transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']

alkaline_earth_metals = ['Ga', 'In', 'Sn', 'Hg', 'Tl', 'Pb', 'Bi']

def load_and_filter_data(data_path):
    df = pd.read_json(data_path)
    Compounds = [ast.literal_eval(i) for i in df['Elements']]

    def filter(compound):
        rare_gases = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}
        if any(element in rare_gases for element in compound):
            return False
        return 2 <= len(compound) <= 5

    filtered_df = df[[filter(compound) for compound in Compounds]]
    filtered_df2 = filtered_df[filtered_df['Elements'].apply(
        lambda x: not any(tm in ast.literal_eval(x) for tm in transition_metals))].reset_index(drop=True)
    filtered_df2 = filtered_df2[filtered_df2['Elements'].apply(
        lambda x: any(tm in ast.literal_eval(x) for tm in alkaline_earth_metals))].reset_index(drop=True)
    return filtered_df2


def preprocess_data(filtered_df, target_name):
    XANES_list = []
    Elements_list_EM = []
    Elements_list_notEM = []

    for _, row in filtered_df.iterrows():
        elements = ast.literal_eval(row['Elements'])
        matching_metals = [metal for metal in alkaline_earth_metals if metal in elements]
        non_matching_metals = list(set(elements) - set(matching_metals))

        ySpec_EM = []
        for metal in matching_metals:
            element_idx = elements.index(metal)
            ySpec = ast.literal_eval(row['ySpec'])[element_idx]
            ySpec_EM.append(ySpec)
        
        XANES_list.append(ySpec_EM)
        Elements_list_EM.append(matching_metals)
        Elements_list_notEM.append(non_matching_metals)

    EMLength_max = max(len(i) for i in Elements_list_EM)
    padded_ySpec = [i + [[0] * 200] * (EMLength_max - len(i)) for i in XANES_list]

    def GetElementsInfo(compound_elements, MaxLenthType):
        compound_elements_info = []
        for element in compound_elements:
            compound_z = Element(element).Z
            compound_x = Element(element).X if Element(element).X is not None else 0
            compound_atomic_radius = Element(element).atomic_radius if Element(element).atomic_radius is not None else 0
            compound_ionization_energies = Element(element).ionization_energies[:3] + [0] * (3 - len(Element(element).ionization_energies[:3]))
            compound_type = []
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
            compound_elements_info += [[0] * 7] * (MaxLenthType - len(compound_elements_info))
        return compound_elements_info

    EMElements_info = [GetElementsInfo(compound, EMLength_max) for compound in Elements_list_EM]
    NotEMLength_max = max(len(i) for i in Elements_list_notEM)
    NotTMElements_info = [GetElementsInfo(compound, NotEMLength_max) for compound in Elements_list_notEM]

    targets = filtered_df[target_name].tolist()

    return EMElements_info, NotTMElements_info, padded_ySpec, targets, EMLength_max, NotEMLength_max

class MyDataset(Dataset):
    def __init__(self, EMElements_info, NotEMElements_info, padded_ySpec, targets):
        valid_data = [
            (xas, em, nottm, target) 
            for xas, em, nottm, target in zip(padded_ySpec, EMElements_info, NotEMElements_info, targets) 
            if target is not None and not (isinstance(target, float) and np.isnan(target))
        ]
        self.padded_ySpec, self.EMElements_info, self.NotEMElements_info, self.targets = zip(*valid_data)

        self.padded_ySpec = torch.stack([torch.tensor(xas, dtype=torch.float32) for xas in self.padded_ySpec])
        self.EMElements_info = torch.stack([torch.tensor(tm, dtype=torch.float32) for tm in self.EMElements_info])
        self.NotEMElements_info = torch.stack([torch.tensor(nottm, dtype=torch.float32) for nottm in self.NotEMElements_info])
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def generate_mask(self, spec, em, notem):
        ySpec_mask = (spec.sum(dim=-1) > 0)
        EMElements_mask = (em.sum(dim=-1) > 0)
        NotEMElements_mask = (notem.sum(dim=-1) > 0)
        return ySpec_mask, EMElements_mask, NotEMElements_mask

    def __getitem__(self, idx):
        ySpec = self.padded_ySpec[idx]
        EMElements = self.EMElements_info[idx]
        NotEMElements = self.NotEMElements_info[idx]
        target = self.targets[idx]
        ySpec_mask, EMElements_mask, NotEMElements_mask = self.generate_mask(ySpec, EMElements, NotEMElements)
        return ySpec, EMElements, NotEMElements, target, ySpec_mask, EMElements_mask, NotEMElements_mask
