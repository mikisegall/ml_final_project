import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_completeness import fill_missing_data
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
        self._model = RandomForestClassifier(n_estimators=90, min_samples_split=3, max_depth=9)
        self._features_to_select = ['num_of_admin_pages', 'PageValues', 'closeness_to_holiday',
         'Weekend', 'device_1.0', 'device_4.0', 'device_5.0', 'device_6.0', 'device_7.0',
          'device_8.0', 'device_other', 'user_type_Other', 'user_type_other', 'browser_name_unknown',
           'Month_Feb', 'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Sep', 'Month_other', 'purchase']

    def train(self, train_file_path: str):
        train_set = pd.read_csv(train_file_path)

        train_set = self._standardize_data(train_set)
        train_set = self._fill_missing_values(train_set)
        train_set = self._drop_chosen_columns(train_set)
        train_set = self._reduce_dimensions(train_set, should_fit=True)
        train_set = self._remove_outliers(train_set)

        train_labels = train_set.pop(LABEL_COL)
        self._model.fit(train_set, train_labels)

    def predict_to_file(self, test_file_path: str, output_file_path: str):
        original_test_set = pd.read_csv(test_file_path)
        processed_test_set = self._standardize_data(original_test_set)
        processed_test_set = self._fill_missing_values(processed_test_set)
        processed_test_set = self._drop_chosen_columns(processed_test_set)
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
        return impute_zscore_test(df, Z_SCORE_THRESHOLD, plot=False)

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
        Using research conclusions, reducing the dataset using the features
        that were chosen by Forward Selection.
        """
        
        if 'purchase' in df.columns:
            df_reduced = df[self._features_to_select]
        else:
            self._features_to_select.remove('purchase')
            df_reduced = df[self._features_to_select]

        return df_reduced

    def run_label_predictions(self, df: pd.DataFrame) -> pd.Series:
        return self._model.predict_proba(df)[:, 1]
