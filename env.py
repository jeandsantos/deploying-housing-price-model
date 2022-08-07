DIR_DATA_TRAIN = './data/train.csv'
DIR_DATA_TEST = './data/test.csv'

COL_ID = 'Id'

MISSING_THRESHOLD_DROP = 0.90
MISSING_THRESHOLD_IMPUTE = 0.50

REGEX_REPL_COLUMN = [
    ('Bsmt', 'Basement'),
    ('Bldg', 'Building'),
    ('Matl', 'Material'),
    ('Abv', 'Above'),
    ('Cond', 'Condition'),
    ('Mas', 'Masonry'),
    ('Vnr', 'Veneer'),
    ('Exter(?=[A-Z])', 'External'),
    ('Qual', 'Quality'),
    ('Fin', 'Finished'),
    ('SF', 'SqFt'),
    ('QC', 'QualityCondition'),
    ('Flr', 'Floor'),
    ('Gr', 'Ground'),
    ('Rms', 'Rooms'),
    ('Yr', 'Year'),
    ('Blt', 'Built'),
    ('Ssn', 'Season'),
    ('Yr', 'Year'),
    ('Mo', 'Month'),
    ('Val', 'Value'),
    ('(?<=^[A-Z]{2})(?=[A-Z]{1}[a-z])|(?<=[a-z]{1})(?=[A-Z])|(?=[0-9]$)', "_"),
    ]

