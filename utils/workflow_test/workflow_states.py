import pandas as pd
import numpy as np
import ast
import re
import torch
import torch.nn.functional as F
from itertools import product
from workflow_xrd import preds_system, K_xrd, N


data_csv = "data/xrd.csv"
data_json = 'data/data.json'
species_json = 'data/species.json'
test_idx_path = './test_idx.npy'

df1 = pd.read_csv(data_csv)
df2 = pd.read_json(data_json)
df3 = pd.read_json(species_json)
df = df2.merge(df1[['material_id', '2theta_list', 'intensity_list']], on='material_id', how='left').merge(df3, on='material_id', how='left')
test_idx = np.load(test_idx_path)
df_test = df.iloc[test_idx]


test_elements = [
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'
]

element_classes = {
    'Sc': {'3+':0, '2+':1},
    'Ti': {'3+':0, '4+':1, '2+':2}, 'V': {'3+':0, '2+':1, '4+':2, '5+':3}, 'Cr': {'3+':0, '6+':1, '4+':2, '2+':3},
    'Mn': {'3+':0, '4+':1, '2+':2}, 'Fe': {'2+':0, '3+':1}, 'Co': {'2+':0, '3+':1}, 'Ni': {'3+':0, '2+':1},
    'Cu': {'+':0, '2+':1, '3+': 3},
    'Y': {'3+':0, '2+':1}, 'Zr': {'4+':0, '2+':1},
    'Nb': {'5+':0, '4+':1}, 'Mo': {'6+':0, '3+':1, '4+':2}, 'Pd': {'2+':0, '4+':1},
    'Ag': {'+':0, '2+':1}
}

def filter_test_elements(elements):
    return any(elem in test_elements for elem in elements)

def has_duplicate(species):
    symbols = [e.rstrip('+-0123456789') for e in species]
    return len(symbols) == len(set(symbols))

def filter_species(species):
    pattern = re.compile(r'^([A-Za-z]+)(\d*[+-])$')
    for elem in species:
        match = pattern.match(elem)
        if not match:
            return False
        base, valence = match.groups()
        if base in element_classes and valence not in element_classes[base]:
            return False
    return True

df_test = df_test[df_test['Elements'].apply(ast.literal_eval).apply(filter_test_elements)]
df_test = df_test[df_test['possible_species'].apply(filter_species) & df_test['possible_species'].apply(has_duplicate)].reset_index(drop=True)


elements_list = df_test['Elements'].apply(ast.literal_eval).tolist()
ySpec_list = df_test['ySpec'].apply(ast.literal_eval).tolist()
possible_species_list = df_test['possible_species'].tolist()


pattern = re.compile(r'^([A-Za-z]+)(\d*[+-])$')
SpecList = []
FormulaList = []

for element_states, elements, ySpec in zip(possible_species_list, elements_list, ySpec_list):
    spec = []
    formula_list = []
    for element_state in element_states:
        match = pattern.match(element_state)
        if not match:
            continue
        base, valence = match.groups()
        formula_list.append(element_state)
        idx = elements.index(base)
        spec.append(ySpec[idx])
    SpecList.append(spec)
    FormulaList.append(formula_list)


class Spec2State(torch.nn.Module):
    def __init__(self, class_num):
        super(Spec2State, self).__init__()
        self.dense1 = torch.nn.Linear(200, 512)
        self.dense2 = torch.nn.Linear(512, class_num)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        return self.dense2(x)


predict_elements = [
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Y',
    'Zr', 'Nb', 'Mo', 'Pd', 'Ag'
]
class_nums = [2, 3, 4, 4, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2]

models_dict = {}
for elem, num in zip(predict_elements, class_nums):
    model_path = f'checkpoints/OxidationStates/{elem}.pth'
    model = Spec2State(num)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    models_dict[elem] = model

def predict_formula(formula_states, spec_list, models_dict, K=1):
    topk_results = []
    for state, spec_val in zip(formula_states, spec_list):
        match = pattern.match(state)
        if not match:
            continue
        element_name, element_state = match.groups()
        if element_name not in models_dict:
            continue
        model = models_dict[element_name]
        spec_tensor = torch.tensor(spec_val, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = model(spec_tensor)
            probs = F.softmax(outputs, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, min(K, len(probs)), dim=-1)
            topk_results.append((topk_probs.squeeze(0).tolist(), topk_indices.squeeze(0).tolist()))
    return topk_results


def calculate_accuracy(true_indices, topk_predictions, K=1):
    return any(true_indices in pred for pred in topk_predictions[:K])


K = 1  # 2,3,4,5. N elemental compound
N = 5  # N元化合物
preds_formula_level = []

for formula_states, spec_list in zip(FormulaList, SpecList):
    if len(formula_states) != N:
        continue
    topk_results = predict_formula(formula_states, spec_list, models_dict, K)
    if not topk_results:
        preds_formula_level.append(1)
        continue
    true_indices = [element_classes[match.group(1)][match.group(2)] for state in formula_states if (match := pattern.match(state)) and match.group(1) in models_dict]
    all_combinations = product(*[zip(probs, indices) for probs, indices in topk_results])
    formula_probs = [(np.prod([p for p, _ in comb]), tuple([i for _, i in comb])) for comb in all_combinations]
    formula_probs.sort(key=lambda x: x[0], reverse=True)
    topk_predictions = [indices for _, indices in formula_probs[:K]]
    preds_formula_level.append(1 if calculate_accuracy(true_indices, topk_predictions, K) else 0)


acc_formula = sum(preds_formula_level) / len(preds_formula_level)
print(f'{N}_elements, Top{K}_formula_Accuracy: {acc_formula:.4f}')

acc_system = preds_system.count(1) / len(preds_system)
print(f'{N}_elements, Top{K_xrd}_system_Acc: {acc_system:.4f}')

K_all = K * K_xrd
all_acc = [1 if formula_pred and system_pred else 0 for formula_pred, system_pred in zip(preds_formula_level, preds_system)]
print(f'{N}_elements, Top{K_all}_Acc: {sum(all_acc) / len(all_acc):.4f}')