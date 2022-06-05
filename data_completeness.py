import pandas as pd
from sklearn import impute


KNN_IMPUTE_NEIGHBORS = 5
INT_COLS = ['num_of_admin_pages', 'num_of_info_pages',
            'num_of_product_pages', 'Region']
DUR_COL_DICT = {
    'info_page_duration': 'num_of_info_pages',
    'admin_page_duration': 'num_of_admin_pages',
    'product_page_duration': 'num_of_product_pages'
}


def fill_missing_data(df: pd.DataFrame):
    filled_data = df.copy()
    filled_data = fill_duration_zeros(filled_data)
    filled_data = fill_special_cols(filled_data)
    filled_data = impute_knn_missing_data(filled_data)
    filled_data = fill_total_duration(filled_data)
    return filled_data


def fill_duration_zeros(df: pd.DataFrame):
    """
     Fills duration columns with zeros where:
     1. The total duration equals zero.
     2. The relevant page visit number equal zero.
    """
    for key in DUR_COL_DICT:
        df[key] = df.apply(
        lambda row: row[key] == 0 if row['total_duration'] == 0 else row[key],
        axis=1
        )
        df[key] = df.apply(
        lambda row: row[key] == 0 if row[DUR_COL_DICT[key]] == 0 else row[key],
        axis=1
        )
    
    df['info_page_duration'] = df.apply(
        lambda row: row['info_page_duration'] == 0 if row['total_duration'] == row['admin_page_duration'] or row['total_duration'] == row['product_page_duration'] else row['info_page_duration'],
        axis=1
        )
    df['admin_page_duration'] = df.apply(
        lambda row: row['admin_page_duration'] == 0 if row['total_duration'] == row['info_page_duration'] or row['total_duration'] == row['product_page_duration'] else row['admin_page_duration'],
        axis=1
        )
    df['product_page_duration'] = df.apply(
        lambda row: row['product_page_duration'] == 0 if row['total_duration'] == row['admin_page_duration'] or row['total_duration'] == row['info_page_duration'] else row['product_page_duration'],
        axis=1
        )
    # TODO: Better way for this?
    return df


def fill_special_cols(df: pd.DataFrame):
    """
     Fills the colums "A", "C"  with values that won't interrupt with the analysis.
    """
    df['A'] = df['A'].fillna('c_0')
    df['A'] = df['A'].str.extract('(\d+)', expand=False)
    df['A'] = df['A'].astype(float)

    df['C'] = df['C'].str.extract('(\d+)', expand=False)
    df['C'] = df['C'].fillna(df['C'].mode()[0])
    df['C'] = df['C'].astype(float)

    return df


def impute_knn_missing_data(df: pd.DataFrame):
    """
     Fills all missing values with KNN imputer.
    """
    imputer = impute.KNNImputer(n_neighbors=KNN_IMPUTE_NEIGHBORS)
    filled_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    for col in INT_COLS:
        filled_df[col] = filled_df[col].apply(lambda x: round(x))

    return filled_df


def fill_total_duration(df: pd.DataFrame):
    """
     Fills the total duration column with the sum of other duration columns.
    """
    df['total_duration'] = df.apply(
        lambda row: row['info_page_duration'] + row['product_page_duration'] + row[
            'admin_page_duration'], axis=1
        )
    
    return df

