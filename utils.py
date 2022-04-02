import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def estimate_model_performance(model, model_name, test_data, test_labels, n_splits):
    """
    Estimate each model version with the following output:
    - Print: runtime, Accuracy
    - Plot: Confusion Matrix, ROC curve for every K-Fold validation
    """
    # Run predictions and calculate runtime
    start_time = time.time()
    predictions = model.predict(test_data, test_labels)
    end_time = time.time()
    runtime = end_time - start_time
    print(f'Model predictions runtime: {runtime}s')

    # Evaluate the predictions
    confusion_matrix_res = confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Accuracy score: {accuracy}")

    plt.figure()
    sn.set(font_scale=1.4)  # for label size
    sn.heatmap(
        confusion_matrix_res / np.sum(confusion_matrix_res),
        annot=True, annot_kws={"size": 16}
    )

    plot_roc_curve_for_k_fold(model, model_name, test_data, test_labels, n_splits)


def plot_roc_curve_for_k_fold(model, model_name, test_data, test_labels, n_splits):
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
    for i, (train, test) in enumerate(cv.split(test_data, test_labels)):
        # Calculate ROC curve per data chunch and plot its curve
        model.fit(test_data[train], test_labels[train])
        viz = RocCurveDisplay.from_estimator(
            model,
            test_data[test],
            test_labels[test],
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
