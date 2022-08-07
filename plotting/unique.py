import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

def plot_count_unique(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str='Count of Unique Values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    df\
        .select_dtypes(include=dtype_include)\
        .apply(lambda x: len(x.unique()))\
        .sort_values(ascending=True)\
        .plot(kind='barh', *args, **kwargs)
    plt.xlabel('Count of Unique Values')
    plt.show()