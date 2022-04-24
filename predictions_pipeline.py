import pandas as pd
from sklearn.linear_model import LinearRegression

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
        train_set = self.standardize_data(train_set)
        train_set = self.fill_missing_values(train_set)
        train_set = self.remove_outliers(train_set)
        train_labels = train_set.pop(LABEL_COL)
        train_set = self.reduce_dimensions(train_set)

        # Fit chosen model with chosen params (this is example)
        self._model = LinearRegression()
        self._model.fit(train_set, train_labels)

    def predict_to_file(self, test_file_path: str, output_file_path: str = None):
        original_test_set = pd.read_csv(test_file_path)
        processed_test_set = self.standardize_data(original_test_set)
        processed_test_set = self.fill_missing_values(processed_test_set)
        processed_test_set = self.reduce_dimensions(processed_test_set)
        predictions = self.run_label_predictions(processed_test_set)

        # Writes the output into a CSV in the requested format
        original_test_set[RESULTS_LABEL_COL] = predictions
        output_file = output_file_path if output_file_path \
            else f'Submission_group_{GROUP_NO}.csv'
        original_test_set[['id', RESULTS_LABEL_COL]].to_csv(output_file)

    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def reduce_dimensions(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def run_label_predictions(self, df: pd.DataFrame) -> pd.Series:
        pass
