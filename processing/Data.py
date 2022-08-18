
#TODO Add method to add 'unknown' category for never-seen-before categories
#TODO Add method to aggregate low count categorical features to 'other' category

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
        
        self._verbose = verbose
        
        self._update_metadata()
        
    def _update_metadata(self):
        self._update_columns()
        self._update_column_types()
        self._update_dimensions()  
        
        if self._verbose:
            print('Table metadata updated')
        

    def create_column(self, col_name:str, values:pd.Series):
        
        self._df[col_name] = values
        
        self._update_metadata()
        
        if self._verbose:
            print(f'Created column \'{col_name}\'')
        

    def change_column_types(self, dtype_dict:dict):
        self._df = self._df.astype(dtype=dtype_dict)
        self._update_column_types()
        
        if self._verbose:
            for col, type in dtype_dict.items():
                print(f'Changed \'{col}\' to \'{type}\' type')


    def drop_columns_regex(self, regex:str, *args, **kwargs):
        
        cols = [x for x in self._columns if re.findall(string=x, pattern=regex)]
        
        self.drop_columns(cols=cols)
        

    def drop_columns(self, cols:list, *args, **kwargs):
        
        self._df = self._df.drop(columns=cols, *args, **kwargs)      
        
        if self._verbose:
            print(f'\nRemoved columns:\n{cols}')
        
        self._update_metadata()  
            
        
    def remove_constant_columns(self, *args, **kwargs):
        column_bool = self._df.apply(lambda x: len(x.unique()) == 1)
        column_desc = self._df.columns[column_bool]

        self._df = self._df.drop(columns=column_desc, *args, **kwargs)      
        
        self._update_metadata()  
        
        if self._verbose:
            print(f'Found {np.sum(column_bool)} constant column(s).\nRemoved columns:\n{column_desc.tolist()}')
            
    def group_low_count_categories(self, col: str, new_col: str = None, fill_value:str = 'other', threshold:float=0.01):
        
        if new_col is None:
            new_col = col
            
        cat_counts = self.df[col].value_counts()

        cat_fraction = cat_counts.to_numpy() / cat_counts.sum() 
        cat_counts_bool = pd.Series(cat_fraction > threshold, index=cat_counts.index)

        col_mapping = {}

        for val in cat_counts.index:
            if cat_counts_bool[val]:
                col_mapping[val] = val
            else:
                col_mapping[val] = fill_value
                
        self.df[new_col] = self.df[col].replace(col_mapping)
        
        return col_mapping
        
    def remove_missing_columns(self, threshold:float):
        
        column_bool = self._df.apply(lambda x: x.isnull().mean() > threshold)
        column_desc = self._df.columns[column_bool]

        self._df = self._df.drop(columns=column_desc)        
        
        self._update_metadata()
        
        if self._verbose:
            print(f'Found {np.sum(column_bool)} column(s) with missing values above the {threshold} threshold.\nRemoved columns:\n{column_desc.tolist()}')

    def add_flag_missing_values(self, 
                                ttest_threshold:float=0.01, 
                                ttest_min_samples:int=30):
        
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
                    
                    if self._verbose:
                        print(f'Adding flag for \'{col}\': p-value below the threshold: {results:.7f}')
            
            else:
                if self._verbose:
                    print(f'Skipping \'{col}\' due to data size being below threshold {df_tmp[var].sum()}')
                    
            self._update_metadata()


    def count_unique(self, dtypes: list = None):
        
        func = lambda x: x.unique().size
        
        if dtypes is None:
            return self._df.apply(func)
        
        return self._df.select_dtypes(include=dtypes).apply(func)


    def count_missing(self, dtypes: list = None):
        
        func = lambda x: x.isnull().sum()
        
        if dtypes is None:
            return self._df.apply(func)
        
        return self._df.select_dtypes(include=dtypes).apply(func)
    
    def perc_missing(self, dtypes: list = None):
        
        func = lambda x: x.isnull().mean()
        
        if dtypes is None:
            return self._df.apply(func)
        
        return self._df.select_dtypes(include=dtypes).apply(func)


    def print_column_types(self):
        print(f'There are {len(self._cat_columns)} categorical fields:\n{self._cat_columns}')
        print(f'There are {len(self._num_columns)} numerical fields:\n{self._num_columns}')


    def _update_column_types(self):
        
        self._cat_columns = self._df.select_dtypes(exclude=['float64', 'int64']).columns.to_list()
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
