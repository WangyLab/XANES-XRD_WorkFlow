import pandas as pd
import numpy as np
import ast
import re
import json
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset

def load_data(xrd_csv_path, json_path):
    df1 = pd.read_csv(xrd_csv_path)
    df2 = pd.read_json(json_path)
    df = pd.merge(df2, df1[['material_id', 'y_smooth']], on='material_id', how='left')
    return df

def filter(compound):
        rare_gases = {'He', 'Ne', 'Ar', 'Kr', 'Xe', 'Rn'}
        if any(element in rare_gases for element in compound):
            return False
        return 2 <= len(compound) <= 5

def load_and_filter_data(df):
    Compounds = [ast.literal_eval(i) for i in df['Elements']]
    transition_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
                     'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
    filtered_df = df[[filter(compound) for compound in Compounds]]
    filtered_df = filtered_df[filtered_df['Elements'].apply(
        lambda x: any(tm in ast.literal_eval(x) for tm in transition_metals))].reset_index(drop=True)
    return filtered_df

def preprocess_target(df):
    crystal_systems = []
    for item in df['symmetry']:
        match = re.search(r"crystal_system=<CrystalSystem\.(.*?):", item)
        if match:
            crystal_systems.append(match.group(1))
        else:
            crystal_systems.append('unknown')
            
    group_mapping = {
        'cubic': 'group1',
        'hex_': 'group2',
        'trig': 'group3',
        'ortho': 'group4',
        'tet': 'group5',
        'mono': 'group6',
        'tri': 'group7'}
    
    grouped_crystal_systems = []
    for crystal in crystal_systems:
        if crystal in group_mapping:
            grouped_crystal_systems.append(group_mapping[crystal])
        else:
            grouped_crystal_systems.append('unknown')
            
    unique_groups = sorted(set(grouped_crystal_systems))
    group_to_index = {group: index for index, group in enumerate(unique_groups)}
    mapped_targets = [group_to_index[group] for group in grouped_crystal_systems]
    targets_tensor = torch.tensor(mapped_targets)
    return targets_tensor
    
def preprocess_xrd_data(df):
    y_xrd_list = df['y_smooth'].apply(json.loads).tolist()
    y_xrd_combined = np.vstack(y_xrd_list)

    scaler = MinMaxScaler()
    normalized_combined = scaler.fit_transform(y_xrd_combined)
    normalized_y_xrd = np.split(normalized_combined, len(y_xrd_list))
    yXRD_tensors = torch.tensor(normalized_y_xrd, dtype=torch.float32)  # Size: (b, 801)
    return yXRD_tensors
    
class SortedXRD_Dataset(Dataset):
    def __init__(self, xrd_csv_path, json_path):
        df = load_and_filter_data(load_data(xrd_csv_path, json_path))
        self.targets = preprocess_target(df)
        self.xrds = preprocess_xrd_data(df)
        assert len(self.targets) == len(self.xrds)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        xrd = self.xrds[idx]
        return xrd, target