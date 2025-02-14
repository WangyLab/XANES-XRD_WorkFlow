from pymatgen.analysis.local_env import CrystalNN
import pandas as pd
import ast
from mp_api.client import MPRester
import json

data_json = "data.json"
df = pd.read_json(data_json)
choose_metal = 'Cr'  # Choose the Metal Element

compounds = df['Elements'].tolist()
compounds = [ast.literal_eval(compounds[i]) for i in range(len(compounds))]

def filter_elements(compound):
    elements = [choose_metal]
    return all(element in compound for element in elements)

filtered_df = df[[filter_elements(compound) for compound in compounds]].reset_index(drop=True)
filtered_df['Elements'] = [ast.literal_eval(filtered_df['Elements'][i]) for i in range(len(filtered_df['Elements']))]
filtered_df['ySpec'] = [ast.literal_eval(filtered_df['ySpec'][i]) for i in range(len(filtered_df['ySpec']))]
print(f'Total Number of {choose_metal}: {len(filtered_df)}')


'''Make Dataset'''
ySpectra = []
for compound, spec in zip(filtered_df['Elements'], filtered_df['ySpec']):
    idx = compound.index(choose_metal)
    Metal_spec = spec[idx]
    ySpectra.append(Metal_spec)
material_ids = filtered_df['material_id'].tolist()

api_key = 'This is API Key'
CN_index_mapping = {4:0, 5:1, 6:2}  # CN:index
with MPRester(api_key) as mpr:
    docs = mpr.materials.summary.search(material_ids=material_ids, fields=['material_id', 'structure'])
    material_id_dataset = []
    CN_dataset = []
    CN_pure = []
    ySpectra_pure = []
    
    for doc in docs:
        structure = doc.structure
        crystal_nn = CrystalNN()
        metal_indices = [index for index, species in enumerate(structure.species) if species.symbol == choose_metal]
        
        CNs = []
        for indice in metal_indices:
            near_sites = crystal_nn.get_nn_info(structure, indice)
            coordination_number = len(near_sites)
            CNs.append(coordination_number)
        
        if all(cn in CN_index_mapping for cn in CNs):
            material_id_dataset.append(doc.material_id)
            metal_idx = filtered_df[filtered_df['material_id'] == doc.material_id]['Elements'].values[0].index(choose_metal)
            metal_ySpec = filtered_df[filtered_df['material_id'] == doc.material_id]['ySpec'].values[0][metal_idx]      
            if len(set(CNs)) == 1:
                ySpectra_pure.append(metal_ySpec)
                CN_value = CNs[0]
                if CN_value in CN_index_mapping:
                    index = CN_index_mapping[CN_value]
                    a = [1 if i == index else 0 for i in range(len(CN_index_mapping))]
                    CN_dataset.append(a)
                    CN_pure.append(index)

pure_data = {'input': ySpectra_pure,
             'targets': CN_pure}

with open(f'{choose_metal}_pure.json', 'w') as f:
    json.dump(pure_data, f)