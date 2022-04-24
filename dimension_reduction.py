import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

"""
Before choosing a model, we will compare 2 methods for dimension reduction:
1. PCA
2. Forward selection
We will train both of them and compare results of PCA vs. best forward selection,
 and based on the results choose what model dimension to try in the next phase (training).
"""
PCA_EXPLAINED_VARIANCE = 0.95


def compare_pca_and_forward_selection(x_train, y_train, x_test, y_test):
    pca_accuracy = calculate_accuracy_for_pca(x_train, y_train, x_test,
                                              y_test, PCA_EXPLAINED_VARIANCE)
    print(f"PCA accuracy: {pca_accuracy}")
    forward_selection_accuracy, forward_selection_features = \
        get_best_feature_subset(x_train, y_train, x_test, y_test)
    print(f"Best forward selection accuracy: {forward_selection_accuracy}")
    if pca_accuracy > forward_selection_accuracy:
        print("PCA scored best. Better use PCA to reduce dimensions.")
    else:
        print(f"Forward Selection scored best. "
              f"Features to use: {forward_selection_features}")


def calculate_accuracy_for_pca(x_train, y_train, x_test, y_test, explained_variance):
    pca_lr = LinearRegression()
    x_pca_train, x_pca_test = transform_data_with_pca(x_train, x_test,
                                                      explained_variance)

    pca_lr.fit(x_pca_train, y_train)
    pca_test_predictions = pca_lr.predict(x_pca_test)
    accuracy = accuracy_score(y_test, pca_test_predictions)
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
    best_accuracy = np.inf
    best_features = []
    accuracy_lst = []
    for n in range(x_train.shape[1]):

        accuracy, features = get_accuracy_for_best_n_features(x_train, y_train,
                                                              x_test, y_test, n, cv)
        print(f"K:{n} Best features set: {features}, Accuracy: {accuracy}")
        accuracy_lst.append(accuracy)
        if accuracy > best_accuracy:
            print(f"New best score! n={n}")
            best_features = features
            best_accuracy = accuracy

    plt.plot(range(1, len(accuracy_lst) + 1), accuracy_lst)
    plt.xlabel("number of features")
    plt.ylabel("Accuracy score")
    plt.title("Accuracy score VS number of features")
    plt.show()

    return best_accuracy, best_features


def get_accuracy_for_best_n_features(x_train, y_train, x_test,
                                     y_test, n: int, cv=10) -> (float, list):
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=n, cv=cv,
                                    scoring='accuracy')
    sfs.fit(x_train, y_train)
    features_list = sfs.get_feature_names_out()

    reduced_x_train = sfs.transform(x_train)
    reduced_x_test = sfs.transform(x_test)
    lr.fit(reduced_x_train, y_train)
    predictions = lr.predict(reduced_x_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy, features_list


if __name__ == "__main__":
    path = "/Users/mikis/Downloads/ML project files/train.csv"
    df = pd.read_csv(path)
    y = df.pop('purchase')
    x_train, x_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, shuffle=True
        )
    compare_pca_and_forward_selection(x_train, y_train, x_test, y_test)

