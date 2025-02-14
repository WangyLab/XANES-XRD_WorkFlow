from mp_api.client import MPRester
import pandas as pd
from emmet.core.summary import HasProps

API_Key = "This is API key"
with MPRester(API_Key) as mpr:
    docs_properties = mpr.summary.search(has_props=[HasProps.xas], fields=["material_id", "formula_pretty", "formula_anonymous", "density", # screened materials' ids
                                                    "is_stable", "energy_above_hull", "formation_energy_per_atom", # Thermodynamic Stability
                                                    "band_gap", "is_gap_direct",  # Electronic Structure
                                                    "ordering", "total_magnetization", "is_magnetic",
                                                    'nelements', 'composition', 'composition_reduced', 'volume', 'density_atomic', 'chemsys',
                                                    'symmetry', 'structure', 'energy_per_atom',
                                                    'cbm', 'vbm', 'efermi', 'bandstructure', 'dos',
                                                    'total_magnetization_normalized_vol', 'total_magnetization_normalized_formula_units',
                                                    'homogeneous_poisson', 'possible_species'])

material_id, formula, elements, xSpec, ySpec = [], [], [], [], []
bandgap, isGapDirect, isStable, E_AboveHull, E_Formation = [], [], [], [], []
formula_anonymous, density = [], []
ordering, total_magnetization, isMagnetic = [], [], []
XASerror_list = []

for i in range(len(docs_properties)):
    results_material_id = docs_properties[i].material_id.string
    try:
        xas_Doc = mpr.xas.search(material_ids=[results_material_id], spectrum_type='XANES', edge='K')
        pass

    except ValueError:
        print("{name} XAS spectrum is error.".format(name=results_material_id))
        XASerror_list.append(results_material_id)
        pass

    else:
        elements_this_material, xSpec_this_material, ySpec_this_material = [], [], []
        material_id.append(results_material_id)
        formula.append(docs_properties[i].formula_pretty)

        for j in range(len(xas_Doc)):
            x_xas = xas_Doc[j].spectrum.x
            y_xas = xas_Doc[j].spectrum.y
            elements_this_material.append(xas_Doc[j].absorbing_element.value)
            xSpec_this_material.append(x_xas)
            ySpec_this_material.append(y_xas)
        elements.append(elements_this_material)
        xSpec.append(xSpec_this_material)
        ySpec.append(ySpec_this_material)

        bandgap.append(docs_properties[i].band_gap)
        isGapDirect.append(docs_properties[i].is_gap_direct)

        isStable.append(docs_properties[i].is_stable)
        E_AboveHull.append(docs_properties[i].energy_above_hull)
        E_Formation.append(docs_properties[i].formation_energy_per_atom)

        formula_anonymous.append(docs_properties[i].formula_anonymous)
        density.append(docs_properties[i].density)
        ordering.append(docs_properties[i].ordering)
        total_magnetization.append(docs_properties[i].total_magnetization)
        isMagnetic.append(docs_properties[i].is_magnetic)

Data = {'Material_id': material_id, 'Formula': formula, 'Formula_anonymous': formula_anonymous, 'Elements': elements,
        'Density':density,'isStable': isStable, 'BandGap':bandgap, 'isGapDirect': isGapDirect,
        'E_AboveHull': E_AboveHull, 'E_Formation': E_Formation,
        'Ordering': ordering, 'Total_magnetization': total_magnetization, 'isMagnetic': isMagnetic,
        'xSpec': xSpec, 'ySpec': ySpec}
df = pd.DataFrame(Data)
df.to_json('OriginalSpectra.json')

# file = open('XASerror.txt', 'w')
# for item in XASerror_list:
#     file.write(item+'\n')
# file.close()