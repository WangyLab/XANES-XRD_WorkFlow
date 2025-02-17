import pandas as pd
import numpy as np
import ast
import re
import torch
from sklearn.preprocessing import MinMaxScaler
from models.Xrd2CrystalSystem.net_lstm import LSTMModel


df1 = pd.read_csv("data/xrd.csv")
df2 = pd.read_json("data/data.json")
df3 = pd.read_json("data/species.json")
df = df1.merge(df2, on="material_id", how="left").merge(df3, on="material_id", how="left")
test_idx = np.load("./test_idx.npy")
df_test = df.iloc[test_idx]
df_filtered = df_test[df_test["possible_species"].apply(lambda x: 2 <= len(x) < 6)]


test_elements = [
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"
]

def filter_test_elements(elements):
    return any(elem in test_elements for elem in elements)

def has_duplicate(species):
    symbols = [e.rstrip("+-0123456789") for e in species]
    return len(symbols) == len(set(symbols))

df_filtered = df_filtered[
    df_filtered["Elements"].apply(ast.literal_eval).apply(filter_test_elements)
]

df_filtered = df_filtered[
    df_filtered["possible_species"].apply(ast.literal_eval).apply(has_duplicate)
].reset_index(drop=True)

element_classes = {
    "Sc": {"3+": 0, "2+": 1},
    "Ti": {"3+": 0, "4+": 1, "2+": 2},
    "V": {"3+": 0, "2+": 1, "4+": 2, "5+": 3},
    "Cr": {"3+": 0, "6+": 1, "4+": 2, "2+": 3},
    "Mn": {"3+": 0, "4+": 1, "2+": 2},
    "Fe": {"2+": 0, "3+": 1},
    "Co": {"2+": 0, "3+": 1},
    "Ni": {"3+": 0, "2+": 1},
    "Cu": {"+": 0, "2+": 1, "3+": 2},
    "Y": {"3+": 0, "2+": 1},
    "Zr": {"4+": 0, "2+": 1},
    "Nb": {"5+": 0, "4+": 1},
    "Mo": {"6+": 0, "3+": 1, "4+": 2},
    "Pd": {"2+": 0, "4+": 1},
    "Ag": {"+": 0, "2+": 1},
}

def filter_species(species):
    pattern = re.compile(r"^([A-Za-z]+)(\d*[+-])$")
    for elem in species:
        match = pattern.match(elem)
        if not match:
            return False
        base, valence = match.groups()
        if base in element_classes and valence not in element_classes[base]:
            return False
    return True

df_filtered = df_filtered[
    df_filtered["possible_species"].apply(ast.literal_eval).apply(filter_species)
].reset_index(drop=True)

N = 4  # 2,3,4,5, N elemental compound
K_xrd = 2  # first K predictions

xrd_list = df_filtered["2theta_list"].apply(ast.literal_eval).tolist()
crystal_system_list = df_filtered["symmetry_x"].apply(lambda x: x.split()).tolist()
element_list = df_filtered["Elements"].apply(ast.literal_eval).tolist()

xrd_N_Elements = []
system_N_Elements = []

for elements, xrd, system in zip(element_list, xrd_list, crystal_system_list):
    if len(elements) == N:
        scaler = MinMaxScaler()
        scaler.fit(np.array([10, 90]).reshape(-1, 1))
        peaks_2theta = scaler.transform(np.array(xrd).reshape(-1, 1)).flatten().tolist()
        xrd_N_Elements.append(peaks_2theta)
        system_N_Elements.append(system)


group_mapping = {
    "cubic": "group1",
    "hex_": "group2",
    "trig": "group3",
    "ortho": "group4",
    "tet": "group5",
    "mono": "group6",
    "tri": "group7",
}

grouped_crystal_systems = []
for system in system_N_Elements:
    match = re.search(r"crystal_system=<CrystalSystem\.(.*?):", system[0])
    grouped_crystal_systems.append(group_mapping.get(match.group(1), "unknown") if match else "unknown")

unique_groups = sorted(set(grouped_crystal_systems))
group_to_index = {group: index for index, group in enumerate(unique_groups)}
mapped_targets = [group_to_index[group] for group in grouped_crystal_systems]

model_path = "checkpoints/XANES_XRD/xrd2crystal_system.pth"
model = LSTMModel(input_dim=1, d_model=256, num_layers=6, dropout=0.1)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

preds_system = []
with torch.no_grad():
    for xrd, target in zip(xrd_N_Elements, mapped_targets):
        xrd_tensor = torch.tensor(xrd, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        lengths = torch.tensor([len(xrd)], dtype=torch.int64)
        outputs = model(xrd_tensor, lengths)
        _, topk_indices = torch.topk(outputs, K_xrd, dim=-1)
        predicted = topk_indices.squeeze(0).tolist()
        preds_system.append(1 if target in predicted else 0)

acc_system = sum(preds_system) / len(preds_system)
print(f"{N}_elements, Top{K_xrd}_Acc: {acc_system:.4f}")