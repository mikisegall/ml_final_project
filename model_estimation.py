import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate

#TODO: Add auc score for training set

def estimate_model_performance(
    model, model_name: str, data: pd.DataFrame,
    labels: pd.DataFrame, n_splits: int
):
    """
    Estimate each model version with the following output:
    - Print: Accuracy,  MSE
    - Plot: Confusion Matrix, ROC curve for every K-Fold validation
    """
    # Evaluate the predictions
    scores = cross_validate(
        model, data, labels,
        scoring=['accuracy', 'neg_mean_squared_error'], cv=n_splits
    )
    print(
        "%0.2f accuracy with a standard deviation of %0.2f" % (
            scores['test_accuracy'].mean(), scores['test_accuracy'].std())
    )
    print(
        "%0.2f MSE with a standard deviation of %0.2f" % (
            -1 * scores['test_neg_mean_squared_error'].mean(),
            scores['test_neg_mean_squared_error'].std())
    )

    predictions = cross_val_predict(model, data, labels, cv=10)
    confusion_matrix_res = confusion_matrix(labels, predictions)

    plt.figure()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(
        confusion_matrix_res / np.sum(confusion_matrix_res),
        annot=True, annot_kws={"size": 16}
    )

    plot_roc_curve_for_k_fold(model, model_name, data, labels, n_splits)


def plot_roc_curve_for_k_fold(
    model, model_name: str, data: pd.DataFrame,
    labels: pd.DataFrame, n_splits: int
):
    """
    A function that for a given untrained model, splits it into n splits and
    calculated the ROC curve for each validation set, displaying the results all
    in one graph with more data such as the mean ROC results and the
    "Chance" curve - results of 0.5 odds.
    """
    cv = StratifiedKFold(n_splits=n_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 10))
    for i, (train, test) in enumerate(cv.split(data, labels)):
        # Calculate ROC curve per data chunk and plot its curve
        model.fit(data.iloc[list(train)], labels.iloc[list(train)])
        viz = RocCurveDisplay.from_estimator(
            model,
            data.iloc[list(train)],
            labels.iloc[list(train)],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        title=model_name,
    )
    ax.legend(loc="lower right")
    plt.show()
