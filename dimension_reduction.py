import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
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
from remove_outliers import impute_zscore_test

PCA_EXPLAINED_VARIANCE = 0.99


def compare_pca_and_forward_selection(x_train, y_train, x_test, y_test):
    """
    Before choosing a model, we will compare 2 methods for dimension reduction:
    1. PCA
    2. Forward selection
    We will train both of them and compare results of PCA vs. best forward selection,
     and based on the results choose what model dimension to try in the next phase (
     training).
    """
    pca_auc = calculate_auc_for_pca(
        x_train, y_train, x_test,
        y_test, PCA_EXPLAINED_VARIANCE
        )
    print(f"PCA AUC: {pca_auc}")
    forward_selection_auc, forward_selection_features = \
        get_best_feature_subset(x_train, y_train, x_test, y_test)
    print(f"Best forward selection AUC: {forward_selection_auc}")
    if pca_auc > forward_selection_auc:
        print("PCA scored best. Better use PCA to reduce dimensions.")
    else:
        print(
            f"Forward Selection scored best. "
            f"Features to use: {forward_selection_features}"
            )


def calculate_auc_for_pca(x_train, y_train, x_test, y_test, explained_variance):
    best_params = {'max_depth': 6, 'min_samples_split': 4, 'n_estimators': 50}
    pca_lr = RandomForestClassifier(**best_params)
    x_pca_train, x_pca_test = transform_data_with_pca(
        x_train, x_test,
        explained_variance
        )

    pca_lr.fit(x_pca_train, y_train)
    pca_test_predictions = pca_lr.predict_proba(x_pca_test)
    auc = roc_auc_score(y_test, pca_test_predictions[:, 1])
    return auc


def transform_data_with_pca(
    x_train: pd.DataFrame, x_test: pd.DataFrame,
    explained_variance: float = 0.95
    ) -> (pd.DataFrame, pd.DataFrame):
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


def get_best_feature_subset(x_train, y_train, x_test, y_test, cv=5):
    # Lower bound of features, it makes little sense to use very little subset
    # as it may cause overfitting and loss of variance.
    min_features = 10
    # Upper bound of features - if we are left with a lot of dimensions,
    # the curse of dimensions is not solved.
    max_features = round(x_train.shape[1] / 2)

    best_auc = 0
    best_features = []
    auc_lst = []
    for n in range(min_features, max_features):
        print(n)
        auc, features = get_auc_for_best_n_features(
            x_train, y_train,
            x_test, y_test, n, cv
            )
        auc_lst.append(auc)
        if auc >= best_auc:
            print(f"Found better AUC! {n} features: {features}, AUC: {auc}")
            best_features = features
            best_auc = auc

    plt.plot(range(min_features, max_features), auc_lst)
    plt.xlabel("number of features")
    plt.ylabel("AUC score")
    plt.title("AUC score VS number of features")
    plt.show()

    return best_auc, best_features


def get_auc_for_best_n_features(
    x_train, y_train, x_test,
    y_test, n: int, cv=5
    ) -> (float, list):
    best_params = {'max_depth': 6, 'min_samples_split': 4, 'n_estimators': 50}
    clf = RandomForestClassifier(**best_params)
    sfs = SequentialFeatureSelector(
        clf, n_features_to_select=n, cv=cv,
        scoring='roc_auc'
        )
    sfs.fit(x_train, y_train)
    features_list = sfs.get_feature_names_out()

    reduced_x_train = sfs.transform(x_train)
    reduced_x_test = sfs.transform(x_test)
    clf.fit(reduced_x_train, y_train)
    predictions = clf.predict_proba(reduced_x_test)
    auc = roc_auc_score(y_test, predictions[:, 1])
    return auc, features_list

#
# if __name__ == "__main__":
#     path = "/Users/mikis/Downloads/ML project files/train.csv"
#     df = pd.read_csv(path)
#     std_df = standardize_data(
#         df, EXTRACT_FLOAT_COLS, BOOL_COLS, CATEGORICAL_COLS, BROWSER_COL,
#         CategoricalEncoder.DUMMY, MONTH_COL
#     )
#     filled_df = fill_missing_data(std_df)
#     Z_SCORE_THRESHOLD = 5.5
#     filled_df = impute_zscore_test(filled_df, Z_SCORE_THRESHOLD)
#     filled_df.pop('id') # TODO - make sure it is somewhere in the code
#
#     y = filled_df.pop('purchase')
#
#     x_train, x_test, y_train, y_test = train_test_split(
#         filled_df, y, test_size=0.2, random_state=42, shuffle=True
#     )
#     compare_pca_and_forward_selection(x_train, y_train, x_test, y_test)
