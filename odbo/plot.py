import matplotlib.pyplot as plt
import numpy as np


def plot_cm(true_labels, pred_labels, Y=None, cmap=None):
    outlier = np.array([k for k, x in enumerate(pred_labels) if x == 1])
    inlier = np.array([k for k, x in enumerate(pred_labels) if x == 0])
    out_outlier = outlier[[
        k for k, x in enumerate(true_labels[outlier]) if x == 1
    ]]
    in_outlier = outlier[[
        k for k, x in enumerate(true_labels[outlier]) if x == 0
    ]]
    out_inlier = inlier[[
        k for k, x in enumerate(true_labels[inlier]) if x == 1
    ]]
    in_inlier = inlier[[
        k for k, x in enumerate(true_labels[inlier]) if x == 0
    ]]
    true = ['True Outlier', 'True Inlier']
    pred = ['Pred Outlier', 'Pred Inlier']
    if Y is None:
        avg = np.ones((2, 2)) * np.inf
    else:
        coo, cio, coi, cii = np.mean(Y[out_outlier]), np.mean(
            Y[in_outlier]), np.mean(Y[out_inlier]), np.mean(Y[in_inlier])
        avg = np.array([[coo, cio], [coi, cii]])

    count = np.array([[len(out_outlier), len(in_outlier)],
                      [len(out_inlier), len(in_inlier)]])
    fig, ax = plt.subplots()
    from matplotlib import cm
    im = ax.imshow(avg, cmap=cm.coolwarm)
    if cmap == None:
        cmap = "YlGn"
    ax.set_xticks(np.arange(len(true)))
    ax.set_yticks(np.arange(len(pred)))
    ax.set_xticklabels(true)
    ax.set_yticklabels(pred)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(len(pred)):
        for j in range(len(true)):
            text = ax.text(
                j,
                i,
                str(round(avg[i, j], 4)) + '\nRatio {0:.3%}'.format(
                    count[i, j] / len(pred_labels), 4),
                ha="center",
                va="center",
                color="k")
    if avg.all() != np.inf:
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Measurments', rotation=-90, va="bottom")
    ax.set_title("Confusion matrix plot for this experiment")
    fig.tight_layout()
#    fig.show()
    return out_outlier, in_outlier, out_inlier, in_inlier
