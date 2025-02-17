import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def InterpolateAndResample(dataframe):
    """
    Interpolate and resample y for a given range of x.

    This function takes a DataFrame containing x and y columns and interpolates the y values
    from x0 to x0 + 56 to have a total of 200 points.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing x and y columns.

    Returns:
    pd.DataFrame: The modified DataFrame with interpolated y values.
    """
    x0 = dataframe['omega'].min()
    x1 = x0 + 56
    x_new = np.linspace(x0, x1, num=200, endpoint=True)
    f = interp1d(dataframe['omega'], dataframe['mu'], kind='cubic', fill_value='extrapolate')
    y_new = f(x_new)
    new_dataframe = pd.DataFrame({'omega': x_new, 'mu': y_new})
    
    return new_dataframe

def StandardizeData(dataframe):
    """
    Standardize the data in the DataFrame.

    This function takes a DataFrame and standardizes the 'mu' column by subtracting the mean
    and dividing by the standard deviation.

    Parameters:
    dataframe (pd.DataFrame): The input DataFrame containing the 'mu' column.

    Returns:
    pd.DataFrame: The modified DataFrame with standardized 'mu' values.
    """
    mean_mu = dataframe['mu'].mean()
    std_mu = dataframe['mu'].std()
    dataframe['mu'] = (dataframe['mu'] - mean_mu) / std_mu
    
    return dataframe


df = pd.read_csv("xanes_data.csv")  # TM xanes data
standardized_df = StandardizeData(InterpolateAndResample(df))