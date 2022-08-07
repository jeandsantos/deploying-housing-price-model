import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

custom_params = {
    "axes.spines.right": False, 
    "axes.spines.top": False
    }

def plot_perc_missing(
    df:pd.DataFrame, 
    dtype_include:list=None, 
    title:str=r'% of Missing Values',
    figsize:tuple=(6,12),
    *args, **kwargs) -> None:
    
    if dtype_include is None:
        dtype_include=df.dtypes.unique()
    
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xticks(range(0,101,20))
    plt.xlabel('% Missing Values')
    plt.axvline(x=50, color='g', linestyle='--')
    plt.axvline(x=90, color='r', linestyle='--')
    df\
        .select_dtypes(include=dtype_include)\
        .apply(lambda x: x.isnull().mean()*100)\
        .sort_values(ascending=True)\
        .plot(kind='barh', *args, **kwargs)
    plt.show()

def boxplot_target_missingness_relationship(
    df:pd.DataFrame,
    col_target:str, 
    col_var:str, 
    *args,
    **kwargs):
    
    df_tmp = df.copy()

    col_na = col_var + '_missing'
    df_tmp[col_na] = df_tmp[col_var].isnull()
    
    sns.set_theme(
        style="ticks", 
        palette=sns.color_palette("Set1"), 
        rc=custom_params)
       
    sns.boxplot(
        y=df_tmp[col_na].astype(str), 
        x=df_tmp[col_target], 
        *args, **kwargs)
    plt.title(f'Distribution of \'{col_target}\' \nbased on missingness of \'{col_na}\' values')
    plt.ylabel(col_na.replace('_', ' ').upper())
    plt.xlabel(col_target.replace('_', ' ').upper())
    plt.show()