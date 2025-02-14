from pymatgen.analysis.diffraction.xrd import XRDCalculator
from mp_api.client import MPRester
import pandas as pd
import csv
import concurrent.futures

def calculate_xrd_for_material(doc):
    """
    计算单个材料的 XRD 模式

    参数：
        doc (dict): 包含 material_id 和结构的字典

    返回：
        tuple: (material_id, 2theta_list, intensity_list)
    """
    material_id = doc.material_id
    structure = doc.structure

    # 初始化 XRD 计算器并计算 XRD 模式
    xrd_calculator = XRDCalculator()
    xrd_pattern = xrd_calculator.get_pattern(structure)
    xrd_x = xrd_pattern.x.tolist()
    xrd_y = xrd_pattern.y.tolist()

    return (material_id, xrd_x, xrd_y)

def fetch_and_calculate_xrd(api_key, material_ids, output_file, batch_size=1000):
    """
    从 Materials Project 获取材料结构并计算 XRD 数据，结果保存为 CSV 文件。

    参数：
        api_key (str): Materials Project API 密钥。
        material_ids (list): 包含材料 ID 的列表。
        output_file (str): 保存结果的 CSV 文件路径。
        batch_size (int): 批量提示进度的材料数量。
    """
    with MPRester(api_key) as mpr:
        # 从 API 获取材料结构
        docs = mpr.materials.summary.search(
            material_ids=material_ids, 
            fields=['material_id', 'structure']
        )
    
    print(f"Total materials to process: {len(docs)}")

    # 使用并行计算加速 XRD 模式计算
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 并行计算每个材料的 XRD 模式
        futures = [executor.submit(calculate_xrd_for_material, doc) for doc in docs]
        
        # 收集计算结果
        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            
            # 每处理 batch_size 个材料输出进度
            if i % batch_size == 0 or i == len(docs):
                print(f"Processed {i}/{len(docs)} materials...")

    # 将结果保存为 CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['material_id', '2theta_list', 'intensity_list'])
        writer.writerows(results)

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    # 配置参数
    api_key = "This is my API key"
    data_json = "data_moreProperties.json"
    output_file = "xrd.csv"

    # 加载材料 ID
    df = pd.read_json(data_json)
    material_ids = df['material_id'].tolist()

    # 调用主函数
    fetch_and_calculate_xrd(api_key, material_ids, output_file)
