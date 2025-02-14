from utils.proprecessing.DelErrorInfo_Interpolate import resampled_df
import numpy as np

def normalize_spectra(df, elements_all):
    for element in elements_all:
        compound_indices = []
        element_spectra = []
        element_indices = []
        
        for i, row in df.iterrows():
            if element in row['Elements']:
                element_idx = row['Elements'].index(element)
                compound_indices.append(i)
                element_indices.append(element_idx)
                element_spectra.append(row['ySpec'][element_idx])
        
        if element_spectra:
            spectra = np.array(element_spectra)
            mean_spectra = np.mean(spectra, axis=1, keepdims=True)
            std_spectra = np.std(spectra, axis=1, keepdims=True)
            std_spectra[std_spectra == 0] = 1
            normalized_spectra = (spectra - mean_spectra) / std_spectra
            for comp_idx, elem_idx, norm_spec in zip(compound_indices, element_indices, normalized_spectra):
                df.at[comp_idx, 'ySpec'][elem_idx] = norm_spec.tolist()

if __name__ == "__main__":
    elements_all = resampled_df['Elements'].explode().unique()
    standardized_df = normalize_spectra(resampled_df, elements_all)
    standardized_df.to_json('data_moreProperties.json')