import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PCA_EXPLAINED_VARIANCE = 0.99


def compare_pca_and_forward_selection(model, x_train, y_train, x_test, y_test):
    """
    Before choosing a model, we will compare 2 methods for dimension reduction:
    1. PCA
    2. Forward selection
    We will train both of them and compare results of PCA vs. best forward selection,
     and based on the results choose what model dimension to try in the next phase (training).
    """
    pca_auc = calculate_rmse_for_pca(model, x_train, y_train, x_test,
                                      y_test, PCA_EXPLAINED_VARIANCE)
    print(f"PCA AUC: {pca_auc}")
    forward_selection_auc, forward_selection_features = \
        get_best_feature_subset(model, x_train, y_train, x_test, y_test)
    print(f"Best forward selection AUC: {forward_selection_auc}")
    if pca_auc > forward_selection_auc:
        print("PCA scored best. Better use PCA to reduce dimensions.")
    else:
        print(f"Forward Selection scored best. "
              f"Features to use: {forward_selection_features}")


def calculate_rmse_for_pca(model, x_train, y_train, x_test, y_test, explained_variance):
    x_pca_train, x_pca_test = transform_data_with_pca(x_train, x_test,
                                                      explained_variance)

    model.fit(x_pca_train, y_train)
    pca_test_predictions = model.predict(x_pca_test)
    auc = roc_auc_score(y_test, pca_test_predictions)
    return auc


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


def get_best_feature_subset(model, x_train, y_train, x_test, y_test, cv=5):

    sfs = SequentialFeatureSelector(model, n_features_to_select=None, cv=cv,
                                    scoring='roc_auc')

    sfs.fit(x_train, y_train)
    features_list = sfs.get_feature_names_out()

    reduced_x_train = sfs.transform(x_train)
    reduced_x_test = sfs.transform(x_test)
    model.fit(reduced_x_train, y_train)
    predictions = model.predict(reduced_x_test)
    auc = roc_auc_score(y_test, predictions)
    return auc, features_list



#
# if __name__ == "__main__":
#    path = "/Users/mikis/Downloads/ML project files/train.csv"
#    df = pd.read_csv(path)
#    std_df = standardize_data(
#        df, EXTRACT_FLOAT_COLS, BOOL_COLS, CATEGORICAL_COLS, BROWSER_COL,
#        CategoricalEncoder.ORDINAL, MONTH_COL
#     )
#     filled_df = fill_missing_data(std_df, INT_COLS, DUR_COL_DICT)
#     filled_df.dropna(inplace=True)  # TODO - temp solution to make it run
#     y = filled_df.pop('purchase')
#     x_train, x_test, y_train, y_test = train_test_split(
#        filled_df, y, test_size=0.2, random_state=42, shuffle=True
#     )
#     compare_pca_and_forward_selection(x_train, y_train, x_test, y_test)
