from utils.ECGDataset import ECGDataset
import torch
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, recall_score, precision_score, \
    f1_score
import os
import datetime


def print_scores(y_target, y_predicted):
    # Compute Area Under the (ROC AUC) from prediction scores.
    print("Area Under the Curve(AUC): ", roc_auc_score(y_target, y_predicted))
    # Compute average precision (AP) from prediction scores.
    print("Average Precision: ", average_precision_score(y_target, y_predicted))

    print(accuracy_score(y_target, y_predicted))
    print(recall_score(y_target, y_predicted))
    print(precision_score(y_target, y_predicted))
    print(f1_score(y_target, y_predicted))


def plot_roc_curve(y_target, y_predicted, guideline=False, save_png=True):
    fpr, tpr, threshold = metrics.roc_curve(y_target, y_predicted)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic (ROC)')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    if guideline:
        plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if save_png:
        if not os.path.isdir('./ROC_Curves/'):
            os.mkdir('./ROC_Curves/')
        now = datetime.datetime.now()
        nowDatetime = now.strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'roc_curve_{nowDatetime}.png', bbox_inches='tight')

    plt.show()


def predict(model, data_path='./data.npy', label_path='./label.npy'):
    dataset = ECGDataset(data_path=data_path, label_path=label_path)
    y_pred_list = []
    for i, X in enumerate(dataset):
        y_pred = model(X)
        y_pred_list.append(y_pred)
    y_predicted = torch.cat([*y_pred_list], dim=0)
    y_predicted = (y_predicted >= 0.5).float()  # threshold (round values)
    y_target = dataset.y
    return y_target, y_predicted
