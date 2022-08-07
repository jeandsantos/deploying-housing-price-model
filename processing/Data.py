
import re

import pandas as pd
import numpy as np
class Data:
    """
    A class to process data 
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = True):
        
        if type(df) is not pd.DataFrame:
            raise TypeError("Wrong data type, a pandas.DataFrame is expected")
        else:
            self._df = df
        
        self.verbose = verbose
        
        self._update_columns()
        self._update_column_types()
        self._update_dimensions()


    def change_column_types(self, dtype_dict:dict):
        self._df.astype(dtype=dtype_dict)
        self._update_column_types()
        
        if self.verbose:
            for col, type in dtype_dict.items():
                print(f'Changed \'{col}\' to \'{type}\' type')

        
    def remove_constant_columns(self):
        constant_bool = self._df.apply(lambda x: len(x.unique()) == 1)
        constant_cols = self._df.columns[constant_bool]

        self._df = self._df.drop(columns=constant_cols)        
        
        if self.verbose:
            print(f'Found {np.sum(constant_bool)} constant column(s). Removing columns:\n{constant_cols.tolist()}')


    def print_column_types(self):
        print(f'There are {len(self._cat_columns)} categorical fields:\n{self._cat_columns}')
        print(f'There are {len(self._num_columns)} numerical fields:\n{self._num_columns}')


    def _update_column_types(self):
        
        self._cat_columns = self._df.select_dtypes(include=['O']).columns.to_list()
        self._num_columns = self._df.select_dtypes(include=['float64', 'int64']).columns.to_list()

        
    def _update_dimensions(self):
        self._nrows, self._ncols  = self._df.shape

        
    def _update_columns(self):
        self._columns = self._df.columns

        
    def __repr__(self):
        return f"Data(df)"
    
    def __str__(self):
        return "A dataframe with {} rows and {} columns".format(self._nrows, self._ncols)
    
    @property
    def df(self):
        return self._df

    @property
    def columns(self):
        return self._columns

    @property
    def ncols(self):
        return self._ncols
    
    @property
    def num_columns(self):
        return self._num_columns
    
    @property
    def cat_columns(self):
        return self._cat_columns
    
    @property
    def nrows(self):
        return self._nrows

    @df.setter
    def df(self, new_df):
        if type(new_df) is not pd.DataFrame:
            raise TypeError("Wrong data type, a pandas.DataFrame is expected")
        else:
            self._df = new_df
            self._columns = self._df.columns

    @columns.setter
    def columns(self, new_columns):
        self._df.columns = new_columns
        self._update_columns()
        self._update_column_types()

    @cat_columns.setter
    def cat_columns(self, new_columns):
        self._cat_columns = new_columns
        self._update_columns()
        self._update_column_types()
            
    @classmethod
    def from_csv(cls, filepath: str, *args, **kwargs):
        df = pd.read_csv(filepath, *args, **kwargs)
        return cls(df)
