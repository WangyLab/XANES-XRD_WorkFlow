from pymatgen.analysis.diffraction.xrd import XRDCalculator
from mp_api.client import MPRester
import pandas as pd
import csv
import concurrent.futures

def calculate_xrd_for_material(doc):
    material_id = doc.material_id
    structure = doc.structure

    # 初始化 XRD 计算器并计算 XRD 模式
    xrd_calculator = XRDCalculator()
    xrd_pattern = xrd_calculator.get_pattern(structure)
    xrd_x = xrd_pattern.x.tolist()
    xrd_y = xrd_pattern.y.tolist()

    return (material_id, xrd_x, xrd_y)

def fetch_and_calculate_xrd(api_key, material_ids, output_file, batch_size=1000):
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=material_ids, 
            fields=['material_id', 'structure']
        )
    
    print(f"Total materials to process: {len(docs)}")

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_xrd_for_material, doc) for doc in docs]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            
            if i % batch_size == 0 or i == len(docs):
                print(f"Processed {i}/{len(docs)} materials...")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['material_id', '2theta_list', 'intensity_list'])
        writer.writerows(results)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    api_key = "This is my API key"
    data_json = "data_moreProperties.json"
    output_file = "xrd.csv"

    df = pd.read_json(data_json)
    material_ids = df['material_id'].tolist()

    # 调用主函数
    fetch_and_calculate_xrd(api_key, material_ids, output_file)
