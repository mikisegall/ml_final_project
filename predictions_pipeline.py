import pandas as pd
from sklearn.linear_model import LinearRegression

from preprocessing_utils import BOOL_COLS
from preprocessing_utils import BROWSER_COL
from preprocessing_utils import CATEGORICAL_COLS
from preprocessing_utils import CategoricalEncoder
from preprocessing_utils import EXTRACT_FLOAT_COLS
from preprocessing_utils import MONTH_COL
from preprocessing_utils import standardize_data

GROUP_NO = 35
LABEL_COL = 'purchase'
RESULTS_LABEL_COL = 'predict_prob'


class PredictionsPipeline:
    """
    A pipeline with a single function product prediction results, through data
    normalization and pre-processing, dimension reduction and model application
     with all the parameters and utilities built during the research.
    """

    def __init__(self):
        self._model = None

    def train(self, train_file_path: str):
        train_set = pd.read_csv(train_file_path)
        train_set = self._standardize_data(train_set)
        train_set = self.fill_missing_values(train_set)
        train_set = self.remove_outliers(train_set)
        train_labels = train_set.pop(LABEL_COL)
        train_set = self._reduce_dimensions(train_set)

        # Fit chosen model with chosen params (this is example)
        self._model = LinearRegression()
        self._model.fit(train_set, train_labels)

    def predict_to_file(self, test_file_path: str, output_file_path: str = None):
        original_test_set = pd.read_csv(test_file_path)
        processed_test_set = self._standardize_data(original_test_set)
        processed_test_set = self.fill_missing_values(processed_test_set)
        processed_test_set = self._reduce_dimensions(processed_test_set)
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
            categorical_encoding_method=CategoricalEncoder.ORDINAL,
            month_col=MONTH_COL
        )

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def _reduce_dimensions(df: pd.DataFrame) -> pd.DataFrame:
        """
        Using research conclusions, reducing the dataset to the
        subset of features resulting in the best performance.
        """
        chosen_cols = ['num_of_product_pages', 'total_duration', 'BounceRates',
                       'ExitRates', 'PageValues', 'closeness_to_holiday', 'Month',
                       'device', 'user_type']
        return df[chosen_cols]

    def run_label_predictions(self, df: pd.DataFrame) -> pd.Series:
        pass
