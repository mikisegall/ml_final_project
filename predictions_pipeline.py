import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from data_completeness import fill_missing_data
from dimension_reduction import PCA_EXPLAINED_VARIANCE
from preprocessing_utils import BOOL_COLS
from preprocessing_utils import BROWSER_COL
from preprocessing_utils import CATEGORICAL_COLS
from preprocessing_utils import CategoricalEncoder
from preprocessing_utils import EXTRACT_FLOAT_COLS
from preprocessing_utils import standardize_data
from remove_outliers import impute_zscore_test

GROUP_NO = 35
LABEL_COL = 'purchase'
RESULTS_LABEL_COL = 'predict_prob'
Z_SCORE_THRESHOLD = 5.5


class PredictionsPipeline:
    """
    A pipeline with a single function product prediction results, through data
    normalization and pre-processing, dimension reduction and model application
     with all the parameters and utilities built during the research.
    """

    def __init__(self):
        self._model = MLPClassifier(hidden_layer_sizes=(100,), alpha=1)
        self._pca = PCA(PCA_EXPLAINED_VARIANCE)

    def train(self, train_file_path: str):
        train_set = pd.read_csv(train_file_path)
        train_labels = train_set.pop(LABEL_COL)

        train_set = self._standardize_data(train_set)
        train_set = self._drop_chosen_columns(train_set)
        train_set = self._fill_missing_values(train_set)
        train_set = self._reduce_dimensions(train_set, should_fit=True)
        train_set = self._remove_outliers(train_set)

        self._model.fit(train_set, train_labels)

    def predict_to_file(self, test_file_path: str, output_file_path: str):
        original_test_set = pd.read_csv(test_file_path)
        processed_test_set = self._standardize_data(original_test_set)
        processed_test_set = self._drop_chosen_columns(processed_test_set)
        processed_test_set = self._fill_missing_values(processed_test_set)
        processed_test_set = self._reduce_dimensions(processed_test_set, should_fit=False)
        predictions = self.run_label_predictions(processed_test_set)

        # Writes the output into a CSV in the requested format
        original_test_set[RESULTS_LABEL_COL] = predictions
        output_file = output_file_path if output_file_path \
            else f'Submission_group_{GROUP_NO}.csv'
        original_test_set[['id', RESULTS_LABEL_COL]].to_csv(output_file)

    @staticmethod
    def _standardize_data(df: pd.DataFrame) -> pd.DataFrame:
        return standardize_data(
            df=df,
            extract_float_cols=EXTRACT_FLOAT_COLS,
            bool_cols=BOOL_COLS,
            categorical_cols=CATEGORICAL_COLS,
            browser_col=BROWSER_COL,
            categorical_encoding_method=CategoricalEncoder.DUMMY
        )

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        return fill_missing_data(df)

    @staticmethod
    def _remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        return impute_zscore_test(df, Z_SCORE_THRESHOLD)

    @staticmethod
    def _drop_chosen_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Dropping columns chosen along the way by the following methods:
        - Very high correlation, features that come from same origin
         as other features left in the data.
        - ID - just an index without extra data.
        - D - Mostly empty and doesn't explain our label much.
        """
        cols_to_drop = ['id', 'BounceRates', 'admin_page_duration',
                        'product_page_duration', 'info_page_duration', 'D']
        return df.drop(columns=cols_to_drop)

    def _reduce_dimensions(self, df: pd.DataFrame,
                           should_fit: bool = False) -> pd.DataFrame:
        """
        Using research conclusions, reducing the dataset using PCA with
         explained_variance=0.99, best feature selection method for MLP.
        """
        scaler = StandardScaler()
        normalized_df = scaler.fit_transform(df)

        if should_fit:
            self._pca.fit(normalized_df)
        return self._pca.transform(normalized_df)

    def run_label_predictions(self, df: pd.DataFrame) -> pd.Series:
        return self._model.predict_proba(df)[:, 1]
