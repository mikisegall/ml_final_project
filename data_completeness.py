import pandas as pd
from sklearn import impute

"""
What I would add:
1. Documentation
2. Maybe seperate into smaller functions or just add comments on what happens
    in each part.
3. After all of this - we still have a lot of rows with nans, 
    can we complete it somehow? (I saw it happend to me but not to you in the 
    notebook so maybe something is missing in this code)
4. C & D consts - I'm not sure how we got the values and if they are correct,
    maybe add documentation? I remember we didn't agree on the D value
"""

D_FILLNA_VAL = -999
C_FILLNA_VAL = 200
KNN_IMPUTE_NEIGHBORS = 5
INT_COLS = ['Month', 'device', 'num_of_admin_pages', 'num_of_info_pages',
            'num_of_product_pages', 'Region']
DUR_COL_DICT = {
    'info_page_duration': 'num_of_info_pages',
    'admin_page_duration': 'num_of_admin_pages',
    'product_page_duration': 'num_of_product_pages'
}


def fill_missing_data(df: pd.DataFrame, int_cols: list, dur_col_dict: dict):

    for key in DUR_COL_DICT:
        df[key] = df.apply(
            lambda row: row[key] == 0 if row["total_duration"] == 0 else row[key],
            axis=1
            )
        df[key] = df.apply(
            lambda row: row[key] == 0 if row[dur_col_dict[key]] == 0 else row[key],
            axis=1
            )

    df["D"] = df["D"].fillna(D_FILLNA_VAL)

    df['C'] = df['C'].fillna(C_FILLNA_VAL)
    df['A'] = df['A'].fillna('c_0')
    df['C'] = df['C'].fillna(df['C'].mode())

    # TODO - maybe for columns with "logX"
    #  it's better to have the expression result instead of X itself?

    df['A'] = df['A'].str.extract('(\d+)', expand=False)
    df['A'] = df['A'].astype(float)

    df['C'] = df['C'].str.extract('(\d+)', expand=False)
    df['C'] = df['C'].astype(float)

    imputer = impute.KNNImputer(n_neighbors=KNN_IMPUTE_NEIGHBORS)
    filled_train_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    for col in int_cols:
        filled_train_data[col] = filled_train_data[col].apply(lambda x: round(x))

    filled_train_data['total_duration'] = filled_train_data.apply(
        lambda row: row['info_page_duration'] + row['product_page_duration'] + row[
            'admin_page_duration'], axis=1
        )

    return df
