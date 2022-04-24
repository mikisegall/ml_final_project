import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_completeness import DUR_COL_DICT
from data_completeness import INT_COLS
from data_completeness import fill_missing_data
from preprocessing_utils import BOOL_COLS
from preprocessing_utils import BROWSER_COL
from preprocessing_utils import CATEGORICAL_COLS
from preprocessing_utils import CategoricalEncoder
from preprocessing_utils import EXTRACT_FLOAT_COLS
from preprocessing_utils import MONTH_COL
from preprocessing_utils import standardize_data

PCA_EXPLAINED_VARIANCE = 0.99
# TODO - I Used RMSE and I'm not sure it is the best metric, we can reconsider.


def compare_pca_and_forward_selection(x_train, y_train, x_test, y_test):
    """
    Before choosing a model, we will compare 2 methods for dimension reduction:
    1. PCA
    2. Forward selection
    We will train both of them and compare results of PCA vs. best forward selection,
     and based on the results choose what model dimension to try in the next phase (training).
    """
    pca_rmse = calculate_rmse_for_pca(x_train, y_train, x_test,
                                      y_test, PCA_EXPLAINED_VARIANCE)
    print(f"PCA RMSE: {pca_rmse}")
    forward_selection_rmse, forward_selection_features = \
        get_best_feature_subset(x_train, y_train, x_test, y_test)
    print(f"Best forward selection RMSE: {forward_selection_rmse}")
    if pca_rmse < forward_selection_rmse:
        print("PCA scored best. Better use PCA to reduce dimensions.")
    else:
        print(f"Forward Selection scored best. "
              f"Features to use: {forward_selection_features}")


def calculate_rmse_for_pca(x_train, y_train, x_test, y_test, explained_variance):
    pca_lr = LinearRegression()
    x_pca_train, x_pca_test = transform_data_with_pca(x_train, x_test,
                                                      explained_variance)

    pca_lr.fit(x_pca_train, y_train)
    pca_test_predictions = pca_lr.predict(x_pca_test)
    accuracy = mean_squared_error(y_test, pca_test_predictions, squared=False)
    return accuracy


def transform_data_with_pca(x_train, x_test, explained_variance: float = 0.95):
    # Normalize the data first
    scaler = StandardScaler()
    normalized_x_train = scaler.fit_transform(x_train)
    normalized_x_test = scaler.transform(x_test)

    pca = PCA(explained_variance)
    pca.fit(normalized_x_train)
    train_components = pca.components_.T
    x_pca_train = np.dot(normalized_x_train, train_components)
    x_pca_test = np.dot(normalized_x_test, train_components)
    return x_pca_train, x_pca_test


def get_best_feature_subset(x_train, y_train, x_test, y_test, cv=10):
    best_rmse = np.inf
    best_features = []
    rmse_lst = []
    for n in range(1, x_train.shape[1]):

        rmse, features = get_rmse_for_best_n_features(x_train, y_train,
                                                          x_test, y_test, n, cv)
        rmse_lst.append(rmse)
        if rmse < best_rmse:
            print(f"New best score! n={n} Best features set: {features}, RMSE: {rmse}")
            best_features = features
            best_rmse = rmse

    plt.plot(range(1, len(rmse_lst) + 1), rmse_lst)
    plt.xlabel("number of features")
    plt.ylabel("RMSE score")
    plt.title("RMSE score VS number of features")
    plt.show()

    return best_rmse, best_features


def get_rmse_for_best_n_features(x_train, y_train, x_test,
                                 y_test, n: int, cv=10) -> (float, list):
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=n, cv=cv,
                                    scoring='neg_root_mean_squared_error')
    sfs.fit(x_train, y_train)
    features_list = sfs.get_feature_names_out()

    reduced_x_train = sfs.transform(x_train)
    reduced_x_test = sfs.transform(x_test)
    lr.fit(reduced_x_train, y_train)
    predictions = lr.predict(reduced_x_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return rmse, features_list


if __name__ == "__main__":
    path = "/Users/mikis/Downloads/ML project files/train.csv"
    df = pd.read_csv(path)
    std_df = standardize_data(
        df, EXTRACT_FLOAT_COLS, BOOL_COLS, CATEGORICAL_COLS, BROWSER_COL,
        CategoricalEncoder.ORDINAL, MONTH_COL
    )
    filled_df = fill_missing_data(std_df, INT_COLS, DUR_COL_DICT)
    filled_df.dropna(inplace=True)  # TODO - temp solution to make it run
    y = filled_df.pop('purchase')
    x_train, x_test, y_train, y_test = train_test_split(
        filled_df, y, test_size=0.2, random_state=42, shuffle=True
    )
    compare_pca_and_forward_selection(x_train, y_train, x_test, y_test)

