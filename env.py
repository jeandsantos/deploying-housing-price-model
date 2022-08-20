DIR_DATA_TRAIN = './data/train.csv'
DIR_DATA_TEST = './data/test.csv'

COL_ID = 'Id'
COL_TARGET = 'sale_price'

MISSING_THRESHOLD_DROP = 0.50
MISSING_THRESHOLD_IMPUTE = 0.50

FLAG_MISSING_PVALUE_TRESHOLD = 0.01
FLAG_MISSING_MIN_SAMPLES = 50

CORRELATION_TARGET_THRESHOLD=0.10
COLLINEARITY_THRESHOLD=0.8

CRAMER_MAX_CARDINALITY = 4
CATEGORICAL_CORRELATION_THRESHOLD = 0.95
CATEGORICAL_CORRELATION_SELECTION_STRATEGY = 'cardinality' # 'random_forest'


REGEX_REPL_COLUMN = [
    ('Bsmt', 'Basement'),
    ('Bldg', 'Building'),
    ('Matl', 'Material'),
    ('Abv', 'Above'),
    ('Cond(?!ition)', 'Condition'),
    ('Mas', 'Masonry'),
    ('Vnr', 'Veneer'),
    ('Exter(?=[A-Z])', 'External'),
    ('Qual', 'Quality'),
    ('Fin(?=[A-Z])', 'Finished'),
    ('SF', 'SqFt'),
    ('QC', 'QualityCondition'),
    ('Flr', 'Floor'),
    ('Grd?', 'Ground'),
    ('Rms', 'Rooms'),
    ('Yr', 'Year'),
    ('Blt', 'Built'),
    ('Ssn', 'Season'),
    ('Yr', 'Year'),
    ('Mo', 'Month'),
    ('Val', 'Value'),
    ('(?<=^[A-Z]{2})(?=[A-Z]{1}[a-z])|(?<=[a-z]{1})(?=[A-Z])|(?=[0-9]$)', "_"),
    ]

