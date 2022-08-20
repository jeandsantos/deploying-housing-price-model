
#TODO Add method to add 'unknown' category for never-seen-before categories
#TODO Add method to aggregate low count categorical features to 'other' category

import re

from scipy.stats import yeojohnson, ttest_ind, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

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
        
        self.drop_columns(cols=cols, *args, **kwargs)
        

    def drop_columns(self, cols:list, *args, **kwargs):
        
        self._df = self._df.drop(columns=cols, *args, **kwargs)      
        
        if self._verbose:
            print(f'\nRemoved columns:\n{cols}')
        
        self._update_metadata()  

        
    def remove_highly_collinear_features(self, 
                                         threshold:float, 
                                         dtype_include:list=[float,int], 
                                         *args, **kwargs):
        
        high_corr = self._df\
            .select_dtypes(include=dtype_include)\
            .corr()\
            .abs()\
            .drop(self._col_target, axis=0)\
            .drop(self._col_target, axis=1)
        
        tri_df = high_corr.mask(np.triu(np.ones_like(high_corr, dtype=bool)))
        corr_cols_drop = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
                        
        if self._verbose:
            print(f'Removed {len(corr_cols_drop)} column(s) due to high collinearity')
        
        self.drop_columns(corr_cols_drop,*args, **kwargs)
                        
                        
    def remove_correlation_features(self, threshold:float, *args, **kwargs):
                        
        corr_bool = self._df.corr()[self._col_target].abs() < threshold
        corr_cols_drop = corr_bool.index[corr_bool].tolist()
        
        if self._verbose:
            print(f'Removed {np.sum(corr_bool)} column(s) with low correlation with {self._col_target}')
        
        self.drop_columns(corr_cols_drop,*args, **kwargs)
    

    def cramers_v(self, col1, col2):

        cross_tab = pd.crosstab(self._df[col1], self._df[col2])
        chi_sq = chi2_contingency(cross_tab)[0]

        total_observations = cross_tab.values.sum()
        min_dim = np.min(cross_tab.shape) - 1

        return np.sqrt(chi_sq/(total_observations*min_dim))
    
    def remove_highly_correlated_categorical_features(self, 
                                                      cramer_max_cardinality:int,
                                                      cramer_threshold:float=0.95,
                                                      selection_strategy:str='cardinality',
                                                      *args, **kwargs
                                                      ):

        cols_bool = self.count_unique() <= cramer_max_cardinality 
        cols_cardinality = cols_bool.index[cols_bool]
        cols_cardinality = cols_cardinality[cols_cardinality.isin(self._cat_columns)]

        X = self._df[cols_cardinality].copy()
        y = self._df[self._col_target].copy()

        res = pd.DataFrame()
        for col_1 in cols_cardinality:
            for col_2 in cols_cardinality:
                if col_1 != col_2:
                    cramers_v_value = self.cramers_v(col_1, col_2)
                    
                    if cramers_v_value > cramer_threshold:
                        tmp = pd.DataFrame([[col_1, col_2]], columns=['col_1', 'col_2'])
                        res = res.append(tmp)

        to_remove = set()
        cardinality = {col:X[col].unique().size for col in cols_cardinality}

        if selection_strategy == 'cardinality':
            for feature_1 in res.col_1:
                if feature_1 in to_remove:
                    next
                else:
                    block = res[res.col_1 == feature_1]
                    max_cardinality=cardinality[feature_1]
                    
                    for feature_2 in block.col_2:
                        if cardinality[feature_2] > max_cardinality:
                            to_remove.add(feature_1)
        
        elif selection_strategy == 'random_forest':
            
            label_encoder = LabelEncoder()
            for col in X.columns:
                X[col] = label_encoder.fit_transform(X[col])
            
            for feature_1 in res.col_1:
                if feature_1 in to_remove:
                    next
                else:
                    block = res[res.col_1 == feature_1]
                    
                    for feature_2 in block.col_2:
                        if feature_2 in to_remove:
                            next
                        else:
                            rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
                            
                            train_features=[feature_1, feature_2]
                            rf.fit(X[train_features], y)
                            
                            remove_feature = train_features[rf.feature_importances_.argmin()]
                            print(f'Removing {remove_feature}')
                            to_remove.add(remove_feature)
        
        self.drop_columns(cols=to_remove, *args, **kwargs)
        
        return to_remove    
        
        
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
        
        if self._verbose:
            print(f'Found {np.sum(column_bool)} column(s) with missing values above the {threshold} threshold.\nRemoved columns:\n{column_desc.tolist()}')

        self._update_metadata()
        
    def add_flag_missing_values(self, 
                                columns:list=None,
                                ttest_threshold:float=0.01, 
                                ttest_min_samples:int=50,
                                ttest_equal_var:bool=False,
                                yeojohnson_transform: bool=False,
                                ):
        
        if columns is None:
            col_missing_bool = self._df.apply(lambda x: x.isnull().sum() > 0)
            columns = self._df.columns[col_missing_bool].tolist()
        
        # Iterate over columns with missing values
        for col in columns:
            if self._verbose:
                print(f'\n{col}')

            df_tmp = self._df.copy()
            
            col_na = col + '_missing'
            col_na_bool = df_tmp[col].isnull()

            # Check if there sample size is large enough for statistical test
            if (col_na_bool.sum() > ttest_min_samples):
                
                # Perform Yeo-Johnson transform
                if yeojohnson_transform:
                    values, _ = yeojohnson(df_tmp[self._col_target])
                    values_missing = values[col_na_bool]
                    values_not_missing = values[~col_na_bool]
                    
                    _, results = ttest_ind(values_missing, values_not_missing, equal_var=ttest_equal_var)
                else:
                    values_missing = df_tmp[self._col_target][col_na_bool]
                    values_not_missing = df_tmp[self._col_target][~col_na_bool]
                    
                    _, results = ttest_ind(values_missing, values_not_missing, equal_var=ttest_equal_var)
                
                # Create flag for missing values
                if results < ttest_threshold:
                    self._df[col_na] = self._df[col].isnull()
                    
                    if self._verbose:
                        print(f'Adding flag for \'{col}\', p-value below the threshold {ttest_threshold:.5f}: {results:.12f}')
                    
                    self._update_metadata()
                        
                elif self._verbose:
                    print(f'Not adding flag for \'{col}\', p-value above the threshold {ttest_threshold:.5f}: {results:.12f}')
                    
            else:
                if self._verbose:
                    print(f'Skipping \'{col}\' due to data size being below threshold {col_na_bool.sum()}')


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
        
        self._cat_columns = self._df.select_dtypes(exclude=[int, float]).columns.to_list()
        self._num_columns = self._df.select_dtypes(include=[int, float]).columns.to_list()


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
        
        if self._verbose:
            print(f'Set \'{self._col_target}\' as target feature')

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
