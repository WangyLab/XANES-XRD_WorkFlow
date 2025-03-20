import numpy as np
import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import re
from collections import defaultdict

def OxiState2Formula(oxi_state_list, max_count=15):
    found_combinations = []    
    for a in range(1, max_count + 1):
        for b in range(1, max_count + 1):
            for c in range(1, max_count + 1):
                partial_sum = oxi_state_list[0]*a + oxi_state_list[1]*b + oxi_state_list[2]*c
                for d in range(1, max_count + 1):
                    total_sum = partial_sum + oxi_state_list[3]*d
                    if total_sum == 0:
                        combination = (a,b,c,d)
                        is_integer_multiple = False
                        for prev_combination in found_combinations:
                            if all(c % prev_c == 0 for c, prev_c in zip(combination, prev_combination) if prev_c != 0):
                                ratio = [c / prev_c if prev_c != 0 else 0 for c, prev_c in zip(combination, prev_combination)]
                                if len(set(ratio)) == 1 and ratio[0] > 0:
                                    is_integer_multiple = True
                                    break
                        
                        if not is_integer_multiple:
                            found_combinations.append(combination)
                        
                        # for i in range(len(found_combinations)):
                        #     # print(found_combinations[i])
                        
    print(len(found_combinations))
    return found_combinations

def get_element_category(element):
    e = Element(element)
    if e.is_metalloid:
        return 'alkaline_earth'
    elif e.is_transition_metal:
        return 'transition_metal'
    elif element in ['B', 'Al', 'Ga', 'In', 'Tl']:
        return 'group13'
    elif element in ['O', 'S', 'Se', 'Te', 'Po']:
        return 'chalcogen'
    else:
        return 'other'

def check_type_correspondence(listA, listB):
    if len(listA) != len(listB):
        return False
    
    categories_A = sorted([get_element_category(ele) for ele in listA])
    categories_B = sorted([get_element_category(ele) for ele in listB])
    return categories_A == categories_B


def get_average_cn(structure, element_symbol):
    cnn = CrystalNN()
    cn_list = []

    for i, site in enumerate(structure):
        if element_symbol in site.species_string:
            try:
                cn = cnn.get_cn(structure, i)
                cn_list.append(cn)
            except Exception as e:
                print(f"index {i} {element_symbol}'s CNs is error: {e}")

    if cn_list:
        avg_cn = int(sum(cn_list) / len(cn_list))
        return avg_cn
    else:
        return None


def get_ionic_radius(element, charge):
    atomic_radius = Element(element).atomic_radius
    ionic_radius = atomic_radius - 0.3 * abs(charge)
    return ionic_radius

def calculate_radii_diff(target_formula, candidate_formula):
    target_elements = Composition(target_formula).as_dict()
    candidate_elements = Composition(candidate_formula).as_dict()

    target_radii = {el: get_ionic_radius(el, target_elements[el]) for el in target_elements}
    candidate_radii = {el: get_ionic_radius(el, candidate_elements[el]) for el in candidate_elements}

    diff = sum((target_radii[el] - candidate_radii.get(el, 0)) ** 2 for el in target_elements)
    return diff


def StrucTempleSearch(crystal_system, formula_combination_number, elements_list, TM_element, TM_CN):
    '''
    crystal_system: 'Cubic'
    formula_combination_number: [2, 1, 1, 5]
    elements_list = [Ca, Mn, Al, O]
    TM_CNs = [4] or [4, 6]
    '''
    formula = ""
    for i, (count, element) in enumerate(zip(formula_combination_number, elements_list)):
        if count == 1:
            formula += element
        else:
            formula += f"{element}{count}"
    
    compostion = Composition(formula)
    anonymous_formula = compostion.anonymized_formula
    
    temples_pretty_list = []
    temples_strcuture_list = []
    with MPRester(API_KEY) as mpr:
        properties = ["material_id", "formula_pretty", "structure", "Elements"]
        results = mpr.summary.search(elements=['O', 'Al'], crystal_system=crystal_system, formula=anonymous_formula, fields=properties)
        if results:
            for doc in results:
                elements_temple = list(Composition(doc.formula_pretty).as_dict().keys())
                if check_type_correspondence(elements_temple, elements_list):
                    structure = doc.structure
                    tm_elements = list({el for el in elements_temple if Element(el).is_transition_metal})
                    if len(tm_elements) == 1:
                        if get_average_cn(structure, TM_element) == TM_CN:
                            temples_pretty_list.append(doc.formula_pretty)
                            temples_strcuture_list.append(doc.structure)
    
    if temples_pretty_list:
        target_formula = formula
        sorted_materials = sorted(zip(temples_pretty_list, temples_strcuture_list), key=lambda x: calculate_radii_diff(target_formula, x[0]))
        temples_pretty_list, temples_strcuture_list = zip(*sorted_materials)
        temples_pretty_best = [temples_pretty_list[0]]
        temples_strcuture_best = [temples_strcuture_list[0]]
    else:
        temples_pretty_best = []
        temples_strcuture_best = []
    return temples_pretty_best, temples_strcuture_best
                

def replace_elements_by_category(original_formula, final_target_elements):
    composition = Composition(original_formula)
    pattern = r'([A-Z][a-z]?)\d*'
    elems_in_order = re.findall(pattern, original_formula)

    category_in_order = []
    for elem in elems_in_order:
        cat = get_element_category(elem)
        if cat not in category_in_order:
            category_in_order.append(cat)
    
    cat_map = {}
    for i, cat in enumerate(category_in_order):
        if i < len(final_target_elements):
            cat_map[cat] = final_target_elements[i]
        else:
            cat_map[cat] = final_target_elements[-1]

    replaced_formula = ""
    for elem in elems_in_order:
        cat = get_element_category(elem)
        new_elem = cat_map[cat]
        count = composition[elem]
        if count == 1:
            replaced_formula += f"{new_elem}"
        else:
            replaced_formula += f"{new_elem}{int(count)}"
    
    return replaced_formula


def replace_elements_by_category_dynamic(original_formula, candidate_elements):
    pattern = r'([A-Z][a-z]?)\d*'
    elems_in_order = re.findall(pattern, original_formula)
    category_in_order = []
    for elem in elems_in_order:
        cat = get_element_category(elem)
        if cat not in category_in_order:
            category_in_order.append(cat)

    cat_to_candidates = defaultdict(list)
    for el in candidate_elements:
        cat_el = get_element_category(el)
        cat_to_candidates[cat_el].append(el)
    
    final_target_elements = []
    for cat in category_in_order:
        if cat_to_candidates[cat]:
            chosen = cat_to_candidates[cat].pop(0)
            final_target_elements.append(chosen)
        else:
            if final_target_elements:
                final_target_elements.append(final_target_elements[-1])
            else:
                final_target_elements.append(candidate_elements[-1])
    replaced_formula = replace_elements_by_category(original_formula, final_target_elements)
    return replaced_formula


def update_structure(structure, original_formula, replaced_formula):
    original_composition = Composition(original_formula)
    replaced_composition = Composition(replaced_formula)
    for site in structure:
        original_element = site.species_string
        if original_element in original_composition:
            category = get_element_category(original_element)
            replaced_element = None
            for elem in replaced_composition:
                if get_element_category(elem.symbol) == category:
                    replaced_element = elem.symbol
                    break
            if replaced_element:
                site.species = {replaced_element: 1.0}
    
    return structure

def generate_new_dict(result_dict, candidate_elements):
    new_dict = {}
    for key, value in result_dict.items():
        original_formula = value[0]
        original_structure = value[1]
        replaced_formula = replace_elements_by_category_dynamic(original_formula, candidate_elements)
        updated_structure = update_structure(original_structure, original_formula, replaced_formula)
        new_dict[key] = [replaced_formula, updated_structure]
    return new_dict


def generate_cif_files(new_dict, output_dir):
    for key, value in new_dict.items():
        replaced_formula = value[0]
        structure = value[1]
        cif_filename = f"{output_dir}/{replaced_formula}.cif"
        structure.to(filename=cif_filename)
        print(f"Generate CIF file: {cif_filename}")

if __name__ == "__main__":
    API_KEY = "your api key"
    oxistates = [2,3,3,-2]
    elements = ['Ca', 'Mn', 'Al', 'O']
    TM_element = 'Mn'
    TM_CN = 6
    formula_list = OxiState2Formula(oxistates)
    result_dict = {}
    for i in formula_list:
        temple_pretty, temple_struc = StrucTempleSearch('Orthorhombic', i, elements, TM_element, TM_CN)
        if temple_pretty:
            result_dict[i] = [temple_pretty[0], temple_struc[0]]

    new_dict = generate_new_dict(result_dict, elements)
    generate_cif_files(new_dict, "your output directory")
    print("Finished!")