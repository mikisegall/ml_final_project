import enum
import re
from time import strptime

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

EXTRACT_FLOAT_COLS = ['info_page_duration', 'product_page_duration']
BOOL_COLS = ['Weekend']
MONTH_COL = 'Month'
BROWSER_COL = 'internet_browser'
CATEGORICAL_COLS = ['user_type', 'browser_name', 'Month']


# TODO - consider running normalization on some of the data (happens in PCA anyways)


class CategoricalEncoder(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one_hot'
    DUMMY = 'dummy'


def standardize_data(
    df: pd.DataFrame, extract_float_cols: list,
    bool_cols: list, categorical_cols: list,
    browser_col: str,
    # Default encoder is OneHot assuming there is no ordinal relation, but it's
    # configurable
    categorical_encoding_method: CategoricalEncoder = CategoricalEncoder.DUMMY,
    month_col: str = None
) -> pd.DataFrame:
    """
    Runs a set of data transformation on columns not standartized by
     default - e.g "5 seconds" instead of 5 as int.
     Missing data is being ignored and will be filled later with different utils.
    """
    for col in extract_float_cols:
        mask = ~df[col].isna()
        df[col][mask] = df[col][mask].apply(extract_float_from_string)

    for col in bool_cols:
        df[col] = df[col].apply(convert_bool_to_int)

    df = parse_browser_col(df, browser_col)
    # Only after we extract the browser clean name, we will encode it

    # if month_col:
    #     mask = ~df[month_col].isna()
    #     df[month_col][mask] = df[month_col][mask].apply(convert_month_name_to_num)

    for col in categorical_cols:
        df = transform_categorical_column(df, col, categorical_encoding_method)

    return df


def parse_browser_col(df: pd.DataFrame, browser_col: str):
    """
    The browser column has a different pattern, and it should be extracted accordingly:
    the pattern is <name>_<version>, where version can be an int (like in safari), or
     a double version like in "browser" ? - browser_4_v15.
     We're under the assumption that "minor" versions like "3_v12"
     or "99.1.4" can be looked as 3 or 99 and this resolution is not importantt enough.
     In case of missing data we add a label of unknown.
    """
    browser_data = df[browser_col].fillna('unknown_0')
    df['browser_name'] = browser_data.apply(lambda x: x.split('_')[0])
    df['browser_version'] = browser_data.apply(extract_browser_version)
    df.drop(columns=[browser_col], inplace=True)
    return df


def extract_browser_version(browser_info: str) -> float:
    version_data = browser_info.split('_')[1]
    version_number = version_data.split('.')[0]
    if version_number.startswith('v'):
        version_number = version_number[1:]
    return float(version_number)


def get_categorical_encoder(encoding_method: CategoricalEncoder):
    if encoding_method == CategoricalEncoder.ONE_HOT:
        return OneHotEncoder()
    else:
        return OrdinalEncoder()


def transform_categorical_column(
    df: pd.DataFrame, col: str,
    encoding_method: CategoricalEncoder
) -> pd.DataFrame:
    """
    Utility to transform categorical in any desired method.
    At first we used ordinal as default but decided instead to use dummy variables.
     The reason is that the data is not continuous and there is no relation of
      big-small and so this way the data represents more accurately what it is.
      The downside is a lot of dimensions that we hope to reduce later.
    """
    df[col].fillna('other', inplace=True)

    if encoding_method == CategoricalEncoder.ONE_HOT:
        encoder = OneHotEncoder()
        encoded_col = encoder.fit_transform(df[[col]])
        encoded_col_df = pd.DataFrame.sparse.from_spmatrix(
            encoded_col,
            columns=encoder.categories_
        )
        df = pd.concat([df, encoded_col_df], axis=1)
        df.drop(columns=[col], inplace=True)

    elif encoding_method == CategoricalEncoder.ORDINAL:
        encoder = OrdinalEncoder()
        df[col] = encoder.fit_transform(df[[col]])

    elif encoding_method == CategoricalEncoder.DUMMY:
        dummy_cols = pd.get_dummies(df[col], prefix=f'{col}')
        df.drop(col, axis=1, inplace=True)
        df = df.join(dummy_cols)

    else:
        raise ValueError("Unknown encoding method!")

    return df


def convert_month_name_to_num(month_name: str) -> int:
    month_number = strptime(month_name[:3], '%b').tm_mon
    return month_number


def convert_bool_to_int(val: bool) -> int:
    """
    Converts True to 1 and False to 0 in order to work better with our models.
    """
    return 1 if val is True else 0


def extract_float_from_string(text: str) -> float:
    nums = re.findall("\d+\.\d+", text)
    if not nums:
        raise ValueError(f"String doesn't contain float in it: {text}")
    return float(nums[0])

#
# if __name__ == "__main__":
#     path = "/Users/mikis/Downloads/ML project files/train.csv"
#     df = pd.read_csv(path)
#     df2 = standardize_data(
#         df, EXTRACT_FLOAT_COLS, BOOL_COLS,
#         CATEGORICAL_COLS, BROWSER_COL,
#         CategoricalEncoder.DUMMY, MONTH_COL
#         )
#
#     df2
#