import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

"""
During the data exploration we already notice that our data containes possible outliers.
Outliers could cause our predictions to be inaccurate, so we would like to remove potential outliers before moving forward.
To make sure we remove only absolute outliers, we will use a zscore test.

Zscore test usually applied on normally distributed data. As we previously seen, not all of our colums are normally distributed, but we believe that this will be covered due to the large amount of train data that we have. In addition, we will take a look at the "before and after" charts, and make sure that only obvious outleirs are filtered out for each column.

We will also exclude some columns from the test, like ID, purchase and column "D" that has very little actual data.

"""


def impute_zscore_test(df: pd.DataFrame, z_threshold: float = 3) -> pd.DataFrame:
    """
    Imputes the z-score test to remove outliers with threshold of 3 standard diviations.
    """
    cols = [col for col in df.columns if col not in ['id', 'purchase']] # we don't want the Zscore test to remove data based on the ID and the purchase columns
    z_scores = stats.zscore(df[cols])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < z_threshold).all(axis=1)
    df_wo_outliers = df[filtered_entries]
    plot_zscore_changes(df, df_wo_outliers)
    return df_wo_outliers


def plot_zscore_changes(original_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    Plots the changes in the data after removing outliers.
    """
    print(f"Original amount of rows: {original_df.shape[0]}")
    print(f"New amount of rows: {new_df.shape[0]}")

    fig, axes = plt.subplots(len(new_df.columns), 1, figsize=(30, 50))
    fig.tight_layout(pad=1.5)
    
    i=0
    for col in new_df.columns:

        plt.hist([original_df[col], new_df[col]])
        axes[i].title.set_text(col)
        i+=1

    axes[0].legend(["With outliers", "Without outliers"])

