import pandas as pd
import ast
from torch.utils.data import Dataset
from pymatgen.core.periodic_table import Element
import torch
import numpy as np

transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']

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
        lambda x: any(tm in ast.literal_eval(x) for tm in transition_metals))].reset_index(drop=True)
    return filtered_df2

def preprocess_data(filtered_df, target_name):
    XANES_list = []
    Elements_list_all = []

    for _, row in filtered_df.iterrows():
        elements = ast.literal_eval(row['Elements'])
        ySpec_all = ast.literal_eval(row['ySpec'])

        XANES_list.append(ySpec_all)
        Elements_list_all.append(elements)

    ElementsLength_max = max(len(i) for i in Elements_list_all)
    padded_ySpec = [i + [[0] * 200] * (ElementsLength_max - len(i)) for i in XANES_list]

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

    Elements_info = [GetElementsInfo(compound, ElementsLength_max) for compound in Elements_list_all]

    targets = filtered_df[target_name].tolist()

    return Elements_info, padded_ySpec, targets, ElementsLength_max

class MyDataset(Dataset):
    def __init__(self, Elements_info, padded_ySpec, targets):
        valid_data = [
            (xas, elements, target) 
            for xas, elements, target in zip(padded_ySpec, Elements_info, targets) 
            if target is not None and not (isinstance(target, float) and np.isnan(target))
        ]
        self.padded_ySpec, self.Elements_info, self.targets = zip(*valid_data)

        self.padded_ySpec = torch.stack([torch.tensor(xas, dtype=torch.float32) for xas in self.padded_ySpec])
        self.Elements_info = torch.stack([torch.tensor(elements, dtype=torch.float32) for elements in self.Elements_info])
        self.targets = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def generate_mask(self, spec, elements):
        ySpec_mask = (spec.sum(dim=-1) > 0)
        Elements_mask = (elements.sum(dim=-1) > 0)
        return ySpec_mask, Elements_mask

    def __getitem__(self, idx):
        ySpec = self.padded_ySpec[idx]
        Elements = self.Elements_info[idx]
        target = self.targets[idx]
        ySpec_mask, Elements_mask = self.generate_mask(ySpec, Elements)
        return ySpec, Elements, target, ySpec_mask, Elements_mask