# XANES-XRD Integrated Machine Learning: A High-throughput Workflow for Inorganic Material Screening and Structure Inference

## Project Description
The XANES-XRD Integrated Machine Learning Workflow is a high-throughput workflow for inorganic material screening and structure inference. This project combines X-ray Absorption Near Edge Structure (XANES) spectroscopy analysis and Powder X-ray Diffraction (PXRD) data processing, using machine learning methods to achieve efficient material analysis and prediction.

This repository integrates the major modules used in the workflow, including XANES-based property prediction, PXRD-related crystal-system modeling, and structure candidate generation.

## What this repository does
This repository combines three main capabilities:

1. **XANES-based multi-target prediction**
   - Input: one CSV spectrum for each transition-metal absorption edge.
   - Output: predicted band-gap existence, magnetic existence, direct/indirect gap label, magnetic order, Ef, Efermi, density, oxidation states, and coordination numbers.
   - Main script: `workflow/multi-target_property_prediction.py`

2. **PXRD-based crystal-system modeling**
   - The repository includes the model code for crystal-system classification.
   - Relevant directory: `models/Xrd2CrystalSystem/`

3. **Structure candidate generation**
   - Input: crystal system, oxidation states, coordination number, and element list.
   - Output: candidate crystal structures exported as CIF files.
   - Main script: `workflow/structure_inference.py`

## Workflow overview
In the current version of the repository, the workflow is organized as follows:

- **Step 1:** use XANES spectra to predict oxidation states, coordination numbers, and related material properties.
- **Step 2:** determine the crystal-system label from PXRD data or prior knowledge.
- **Step 3:** combine the predicted/local chemical information with the crystal-system label to generate candidate crystal structures.

For the currently documented XANES-to-structure-candidate path, the main user-facing scripts are:
- `workflow/multi-target_property_prediction.py`
- `workflow/structure_inference.py`

These two scripts cover the demonstration workflow currently documented in this repository.

## Modules
- **XANES Spectroscopy Analysis**: Provides tools for preprocessing, feature extraction, and analysis of XANES spectra.

- **PXRD Data Processing**: Provides PXRD preprocessing, feature extraction, and adds Gaussian noise to simulate experimental conditions.

- **Multi-target Property Prediction Module**: Provides a variety of machine learning models (SpecFusionNet, MLP), allowing for quick prediction of multi-target properties, element oxidation states, and coordination numbers from XANES spectra.

- **Crystal System Prediction Module**: Provides hybrid CNN-Transformer model to predict crystal-system categories from PXRD data.

- **Structure Inference Module**: Provides a method for inferring the structure of unknown materials. The oxidation-state category, coordination number, and crystal-system category are combined to retrieve and adapt likely structure templates.

## Project Structure
```
XANES-XRD_WorkFlow/
├─ checkpoints  # Model checkpoints
│  ├─ CNs
│  ├─ CrystalSystem
│  ├─ OxidationStates
│  └─ SpecFusionNet
├─ models
│  ├─ OxidationStates_CNs  # MLP
│  ├─ SpecFusionNet
│  └─ Xrd2CrystalSystem  # Hybrid CNN-Transformer
├─ README.md
├─ utils
│  ├─ interpretability_analysis  # Ablation Experiments & Grad-CAM
│  └─ proprecessing  # Preprocessing scripts for XANES and PXRD data
└─ workflow
   ├─ multi-target_property_prediction.py
   ├─ net.py
   ├─ structure_inference.py
   └─ ZnCu2SnS4  # Folder containing example XANES data
```

## Environment Dependencies
- python 3.8
- pandas 1.5.3
- numpy 1.23.5
- pytorch 2.2.1
- pymatgen 2023.8.10
- tqdm 4.65.0
- scikit-learn 1.3.2
- matplotlib 3.7.2
- scipy 1.10.1
- mp_api 0.35.1

## Input requirements
### 1. XANES input
The script `workflow/multi-target_property_prediction.py` expects one CSV file for each transition-metal spectrum.

Each CSV file should contain at least two columns:
- `omega`
- `mu`

The script internally interpolates each spectrum to 200 points and standardizes the `mu` values before inference.

### 2. Element list
Before running inference, the user needs to edit the following variables in `workflow/multi-target_property_prediction.py`:
- `TM_elements`
- `Non_TM_elements`
- `csv_file_*` paths
- `csv_files`

### 3. Structure-inference inputs
Before running `workflow/structure_inference.py`, the user needs to set:
- `API_KEY`
- `oxistates`
- `elements`
- `TM_element`
- `TM_CN`
- crystal system passed into `StrucTempleSearch(...)`
- output directory passed into `generate_cif_files(...)`

## Quickstart
### Step 1. Prepare XANES CSV files
Prepare one CSV file per transition-metal absorption edge. Example data are included in:
- `workflow/ZnCu2SnS4/data1.csv`
- `workflow/ZnCu2SnS4/data2.csv`

### Step 2. Edit the XANES inference script
Open `workflow/multi-target_property_prediction.py` and edit the variables in the `if __name__ == '__main__':` block:

```python
TM_elements = ['Co']
Non_TM_elements = ['O']
csv_file_Cu = "path/to/your/spectrum.csv"
csv_files = [csv_file_Cu]
```

If your material contains multiple transition metals, provide one CSV file for each transition-metal edge and list the corresponding elements in `TM_elements`.

### Step 3. Run XANES inference
From the repository root, run:

```bash
python workflow/multi-target_property_prediction.py
```

This script prints predicted:
- band-gap existence
- magnetic existence
- direct/indirect gap label
- magnetic order
- Ef
- Efermi
- Eg
- density
- transition-metal coordination numbers
- transition-metal oxidation states
- non-transition-metal oxidation states

### Step 4. Prepare structure-inference inputs
The structure-inference script requires:
- a crystal-system label such as `Cubic` or `Orthorhombic`
- oxidation states
- target elements
- transition-metal coordination number
- a valid Materials Project API key

### Step 5. Edit the structure inference script
Open `workflow/structure_inference.py` and update the `if __name__ == '__main__':` block.

Example fields to update:

```python
API_KEY = "your api key"
oxistates = [2, 3, 3, -2]
elements = ['Ca', 'Mn', 'Al', 'O']
TM_element = 'Mn'
TM_CN = 6
```

Also update:
- the crystal system passed into `StrucTempleSearch(...)`
- the output directory passed into `generate_cif_files(...)`

### Step 6. Run structure inference
From the repository root, run:

```bash
python workflow/structure_inference.py
```

The script searches Materials Project templates, adapts the matched structure to the target composition, and writes candidate CIF files to the specified output directory.

## Outputs
### From `workflow/multi-target_property_prediction.py`
Printed predictions include:
- electronic-property labels
- magnetic-property labels
- regression targets
- oxidation states
- coordination numbers

### From `workflow/structure_inference.py`
Generated outputs include:
- candidate formulas
- candidate structures
- CIF files for the inferred structure candidates

## Notes
- The repository includes the major modules required for the workflow in one codebase.
- The current usage pattern is script-based and intended to document how to run the released models on user data.
- `workflow/structure_inference.py` requires a valid Materials Project API key.
