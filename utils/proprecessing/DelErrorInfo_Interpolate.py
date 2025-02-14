from MakeDatasetFromMP import df
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def DelXspecErrorXANES(element, dataframe):
    element_rows = dataframe[dataframe['Elements'].apply(lambda elements: element in elements)]
    element_indices = element_rows.index
    first_values = [row['xSpec'][row['Elements'].index(element)][0] for _, row in element_rows.iterrows()]
    mid_value = np.median(first_values)
    error_indices = [
        element_indices[i] for i, value in enumerate(first_values)
        if not (mid_value - 20 <= value <= mid_value + 20)
    ]
    return error_indices

def DelErrorRow(dataframe):
    elements_all = dataframe['Elements'].explode().unique()
    error_indices = set()
    for element in elements_all:
        error_indices.update(DelXspecErrorXANES(element, dataframe))
    cleaned_df = dataframe.drop(index=error_indices).reset_index(drop=True)
    negative_indices = cleaned_df.index[
        cleaned_df['ySpec'].apply(lambda yspec: any(any(x < 0 for x in spec) for spec in yspec))
    ]
    return cleaned_df.drop(index=negative_indices).reset_index(drop=True)

def InterpolateAndResample(dataframe):
    """Interpolate and resample xSpec and ySpec for all elements."""
    elements_all = dataframe['Elements'].explode().unique()
    for element in elements_all:
        # Determine xSpec range for the element
        element_last = [
            dataframe['xSpec'][i][dataframe['Elements'][i].index(element)][-1]
            for i in range(len(dataframe)) if element in dataframe['Elements'][i]
        ]
        x_max = min(element_last)
        x_min = x_max - 56
        
        # Modify xSpec and ySpec for the element
        for i, row in dataframe.iterrows():
            if element in row['Elements']:
                idx_element = row['Elements'].index(element)
                x_old = row['xSpec'][idx_element]
                y_old = row['ySpec'][idx_element]
                
                # Generate new xSpec and interpolate ySpec
                x_new = np.linspace(x_min, x_max, num=200, endpoint=True)
                f = interp1d(x_old, y_old, kind='cubic', fill_value='extrapolate')
                y_new = f(x_new)
                
                # Set ySpec to 0 for invalid ranges
                y_new = np.where((x_new < x_old[0]) & (y_new < 0), 0, y_new)
                
                # Update dataframe
                dataframe.at[i, 'xSpec'][idx_element] = x_new.tolist()
                dataframe.at[i, 'ySpec'][idx_element] = y_new.tolist()
    return dataframe

if __name__ == '__main__':
    final_df = (
        df[df['Elements'].str.len() != 0]
        .loc[lambda x: x.apply(lambda row: sum(1 for char in row['Formula_anonymous'] if char.isupper()) == len(row['Elements']), axis=1)]
        .reset_index(drop=True)
    )
    cleaned_df = DelErrorRow(final_df)
    resampled_df = InterpolateAndResample(cleaned_df)
    resampled_df.to_json('newSpec.json')