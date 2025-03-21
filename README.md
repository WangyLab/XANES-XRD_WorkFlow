# XANES-XRD Integrated Machine Learning: A High-throughput Workflow for Inorganic Material Screening and Structure Inference

## Project Description
The XANES-XRD Integrated Machine Learning Workflow is a high-throughput workflow for inorganic material screening and structure inference. This project combines X-ray Absorption Near Edge Structure (XANES) spectroscopy analysis and Powder X-ray Diffraction (PXRD) data processing, using machine learning methods to achieve efficient material analysis and prediction.

## Features
- **XANES Spectroscopy Analysis**: Provides tools for preprocessing, feature extraction, and analysis of XANES spectra.

- **PXRD Data Processing**: Provides PXRD preprocessing, feature extraction, and adds Gaussian noise to simulate experimental conditions.

- **Multi-target Property Prediction Module**: Provides a variety of machine learning models (SpecFusionNet, MLP), allowing for quick prediction of multi-target properties, element oxidation states, and coordination numbers from XANES spectra.

- **Crystal System Prediction Module**: Provides hybrid CNN-Transformer model to quickly predict crystal system categories from PXRD data.

- **Structure Inference Module**: Provides a method for inferring the structure of unknown materials. The oxidation state category, coordination number, and crystal system category are obtained through the Multi-target Property Prediction Module and Crystal System Prediction Module. Through this module, the most likely structure of the material can be quickly inferred.

- **Automated Workflow**: Inplements an automated processing workflow from data input to result output, supporting batch operations to improve efficiency.

## Project Structure
```
XANES-XRD_WorkFlow/
├─ checkpoints
│  ├─ CNs
│  ├─ CrystalSystem
│  ├─ OxidationStates
│  └─ SpecFusionNet
├─ models
│  ├─ OxidationStates_CNs
│  ├─ SpecFusionNet
│  └─ Xrd2CrystalSystem
├─ README.md
├─ utils
│  ├─ interpretability_analysis
│  └─ proprecessing
└─ workflow
   ├─ multi-target_property_prediction.py
   ├─ net.py
   ├─ structure_inference.py
   └─ ZnCu2SnS4
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

