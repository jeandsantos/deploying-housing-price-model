import re

import scipy.stats as stats
import pandas as pd
import numpy as np
class Data:
    """
    A class to process data 
    """

    def __init__(self, df: pd.DataFrame, col_target:str = None, verbose: bool = True):
        
        if type(df) is not pd.DataFrame:
            raise TypeError("Wrong data type, a pandas.DataFrame is expected")
        else:
            self._df = df
            
        if col_target is None:
            self._col_target = None
        else:
            self._col_target = col_target
        
        self.verbose = verbose
        
        self._update_metadata()
        
    def _update_metadata(self):
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
        column_bool = self._df.apply(lambda x: len(x.unique()) == 1)
        column_desc = self._df.columns[column_bool]

        self._df = self._df.drop(columns=column_desc)      
        
        self._update_metadata()  
        
        if self.verbose:
            print(f'Found {np.sum(column_bool)} constant column(s).\nRemoved columns:\n{column_desc.tolist()}')
        
    def remove_missing_columns(self, threshold:float):
        
        column_bool = self._df.apply(lambda x: x.isnull().mean() > threshold)
        column_desc = self._df.columns[column_bool]

        self._df = self._df.drop(columns=column_desc)        
        
        self._update_metadata()
        
        if self.verbose:
            print(f'Found {np.sum(column_bool)} column(s) with missing values above the {threshold} threshold.\nRemoved columns:\n{column_desc.tolist()}')

    def add_flag_missing_values(self, 
                                ttest_threshold:float=0.01, 
                                ttest_min_samples:int=30):
        
        cols_flagged = []
        self.col_target = 'sale_price'
        
        col_missing_bool = self._df.apply(lambda x: x.isnull().sum() > 0)
        col_missing_desc = self._df.columns[col_missing_bool].tolist()
        
        # Iterate over columns with missing values
        for col in col_missing_desc:

            df_tmp = self._df.copy()
            
            var = col + '_missing'
            df_tmp[var] = df_tmp[col].isnull()

            if df_tmp[var].sum() > ttest_min_samples:

                _, results = stats.ttest_ind(
                    df_tmp[self._col_target][~df_tmp[var]],
                    df_tmp[self._col_target][df_tmp[var]]
                    )
                
                if results < ttest_threshold:
                    self._df[var] = self._df[col].isnull()
                
                    cols_flagged.append((col, var, results))
                    
                    if self.verbose:
                        print(f'Adding flag for \'{col}\': p-value below the threshold: {results:.7f}')
                        print(f't-test p-value below the threshold: {results:.5f}')
            
            else:
                if self.verbose:
                    print(f'Skipping \'{col}\' due to data size being below threshold {df_tmp[var].sum()}')
        

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
    def col_target(self):
        return self._col_target

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

    @col_target.setter
    def col_target(self, new_column):
        self._col_target = new_column

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
