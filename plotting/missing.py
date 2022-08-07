import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot_perc_missing(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str=r'% of missing values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xticks()
    plt.axvline(x=50, color='g', linestyle='--')
    plt.axvline(x=90, color='r', linestyle='--')
    df\
        .select_dtypes(include=object)\
        .apply(lambda x: x.isnull().mean()*100)\
        .sort_values(ascending=True)\
        .plot(kind='barh')
    plt.show()