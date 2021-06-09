import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

import dfencoder
from dfencoder import AutoEncoder

def get_unique_value_lengths(dataframe: pd.DataFrame, col_name: str):
    unique_vals = map(lambda r: str(r), dataframe[col_name].unique())
    unique_vals = filter(lambda r: r != 'nan', unique_vals)
    unique_vals = map(lambda r: len(r), unique_vals)
    unique_vals = list(unique_vals)
    return np.array(unique_vals)

def inferred_type(dataframe: pd.DataFrame, col_name: str, max_cat_value_count: int=1000) -> np.dtype:
    is_datetime_col = dataframe[col_name].str.match('(\d{2,4}(-|\/|\\|\.| )\d{2}(-|\/|\\|\.| )\d{2,4})+').all()
    if is_datetime_col:
        return 'datetime'
    
    is_int32 = dataframe[col_name].str.match('\d{1,8}$').all()
    if is_int32:
        return 'int32'
    
    is_float = dataframe[col_name].str.match(r'\d{1,7}(\.\d{1,5})?$').all()
    if is_float:
        return 'float'
    
    unique_vals = dataframe[col_name].unique()
    n_unique = unique_vals.shape[0]

    if n_unique == 2 or n_unique == 3:
        bool_vals = np.array(['(N/A)', 'N', 'Y'], dtype='str')
        possible_bool_vals = np.array(pd.DataFrame(unique_vals).fillna('(N/A)')[0])
        if np.isin(possible_bool_vals, bool_vals).all():
            return 'bool'
    
    if n_unique >= 2 and n_unique < max_cat_value_count:
        unique_val_lengths = get_unique_value_lengths(dataframe, col_name)
        if np.max(unique_val_lengths) > 300:
            # print(f'{col_name} -> {np.max(unique_val_lengths)}')
            return 'object'
        return 'category'

    return 'object'

def clean_icecat(dataframe: pd.DataFrame):
    # Count number of rows per category
    df_count_by_category = dataframe.groupby('category_name').agg({'id': 'count'}).rename(columns={'id': 'n_rows'})

    # Find categories with at least N amount of rows
    categories = list(df_count_by_category[df_count_by_category.n_rows > 20].index)

    # Delete rows with few that N amount of rows per category
    dataframe = dataframe[dataframe.category_name.isin(categories)]

    # Get columns that specify features of the products
    product_feature_columns = list(dataframe.columns)[26:]

    # Find columns that have too few specified values
    n_rows = dataframe.shape[0]
    small_columns = []
    for col in product_feature_columns:
        n_filled = n_rows - dataframe[col].isna().sum()
        if n_filled < 10:
            small_columns.append(col)

    # Find columns that have enough values
    product_features_to_use = [col for col in product_feature_columns if col not in small_columns]
    
    # Create a copy
    df_cleaned = dataframe[['category_name', 'supplier_name'] + product_features_to_use].copy()
    
    # Use proper dtypes
    for col in df_cleaned.columns:
        dtype = inferred_type(df_cleaned, col)
        if dtype == 'int32':
            df_cleaned[col].fillna(0, inplace=True)
        elif dtype == 'float':
            df_cleaned[col].fillna(0.0, inplace=True)
        elif dtype == 'bool':
            df_cleaned[col].fillna('N', inplace=True)
            df_cleaned[col] = df_cleaned[col].str.replace('N', '0')
            df_cleaned[col] = df_cleaned[col].str.replace('Y', '1')
            df_cleaned[col] = df_cleaned[col].astype('int')
        elif dtype == 'category':
            df_cleaned[col].fillna('(N/A)', inplace=True)
            
        df_cleaned[col] = df_cleaned[col].astype(dtype)

    return df_cleaned

def split_train_test(df):
    train = df.sample(frac=.8, random_state=42)
    test = df.loc[~df.index.isin(train.index)]
    return train, test

def compute_column_stats(dataframe: pd.DataFrame):
    dmap = {
        'column': [],
        'suggested_type': [],
        'n_unique': [],
        'len_total': [],
        'len_min': [],
        'len_max': [],
        'len_avg': [],
        'values': [],
        'n_filled': [],
    }

    n_rows = dataframe.shape[0]

    for col in dataframe.columns:
        dmap['column'].append(col)
        dmap['suggested_type'].append(inferred_type(dataframe, col))

        dmap['n_filled'].append( n_rows - dataframe[col].isna().sum() )

        dmap['n_unique'].append(dataframe[col].unique().shape[0])
        unique_val_lengths = get_unique_value_lengths(dataframe, col)
        dmap['len_total'].append(len(unique_val_lengths))
        dmap['len_min'].append(np.min(unique_val_lengths))
        dmap['len_max'].append(np.max(unique_val_lengths))
        dmap['len_avg'].append(np.mean(unique_val_lengths))

        vals = ' | '.join([str(s) for s in list(dataframe[col].unique())[0:5]])
        dmap['values'].append(vals)

    return pd.DataFrame(dmap)


def main():
    print('Fetching data...')
    df_data = pd.read_csv('ice-cat-office-products.csv.gz', dtype=str, index_col=0)

    print('Cleaning data...')
    df_cleaned_data = clean_icecat(df_data)
    df_train, df_test = split_train_test(df_cleaned_data)

    model = dfencoder.AutoEncoder(
        encoder_layers = [128, 64], #model architecture
        decoder_layers = [64, 128], #decoder optional - you can create bottlenecks if you like
        activation='relu',
        swap_p=0.2, #noise parameter
        lr = 0.01,
        lr_decay=.99,
        batch_size=512,
        logger_name='tensorboard', #special logging for jupyter notebooks
        verbose=True,
        optimizer='sgd',
        scaler='gauss_rank', #gauss rank scaling forces your numeric features into standard normal distributions
        min_cats=3 #Define cutoff for minority categories, default 10
    )

    print('Fitting model...')
    model.fit(df_train, epochs=10, val=df_test)



if __name__ == '__main__':
    main()
