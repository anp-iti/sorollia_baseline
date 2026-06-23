import numpy as np
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
import os
import math


def get_roc_curve(y_true, y_pred, classes,path_to_cm='/panns_implementation/outputs/roc_curve.png'):
    """Obtain roc_curve image per class from ground truth and prediction

    Args:
        y_true (array): True binary labels or binary label indicators for each class.
        y_pred (array): Predicted labels in binary.
        classes (list): List of class labels that correspond to indices in y_true and y_pred.
        path_to_cm (str, optional): The file path where the ROC curve image will be saved. Defaults to '/panns_implementation/outputs/roc_curve.png'.
    """
    # plt.clf()
    n_classes = y_pred.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                    ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve per class and micro-average')
    plt.legend(loc="lower right",fontsize = '7')   
    plt.savefig(path_to_cm)
    plt.close()


def get_f1_score(y_true, y_pred,clipwise_output, classes,path_to_cm):
    """Obtain f1 per class

    Arguments:
        y_true {np.array} -- ground truth i.e. [0,1,2,1,2,2,1]
        y_pred {_type_} -- predicted labels in one hot encoding format (number of samples x number of classes)
                            i.e. [[1,0,0], [0,1,0]] etc
        classes {list} -- classes i.e. ['dog', 'cat', 'elephant']
    """
    with open(path_to_cm, 'w') as file:

        # F-Score per class
        average_precision = average_precision_score(y_true, clipwise_output, average=None)
        file.write(f'mAP: {round(np.mean(average_precision),3)}\n')
        file.write(classification_report(y_true,y_pred,target_names = classes))

        file.close()


def get_confusion_matrix(y_true, y_pred, classes,
                         path_to_cm='/panns_implementation/outputs/confusion_matrix.png'):
    """Obtain confusion matrix

    Arguments:
        y_true {np.array} -- ground truth i.e. [0,1,2,1,2,2,1]
        y_pred {np.array} -- predicted labels in index format i.e. [0,1,1,1,2,2,0]
        classes {list} -- classes i.e. ['dog', 'cat', 'elephant']

    Keyword Arguments:
        path_to_cm {str} -- path to store confusion matrix image 
                            (default: {'/panns_implementation/outputs/confusion_matrix.png'})
    """


    confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    sns.set(font_scale=1.0)
    num_conf_matrix = len(classes)
    num_rows = int(math.sqrt(num_conf_matrix))
    num_cols = math.ceil(num_conf_matrix / num_rows)
    fig, axes = plt.subplots(num_rows,num_cols)
    axes = axes.ravel()
    for i in range(confusion_matrix.shape[0]):
        sns.heatmap(confusion_matrix[i], annot = True, fmt = "d", cmap = "YlOrRd", cbar = False, ax = axes[i], annot_kws={"size": 9})

        axes[i].set_title(f"Class {classes[i]}",fontsize = 8)
        axes[i].set_xlabel('Predicted',fontsize = 8)
        axes[i].set_ylabel('True',fontsize = 8)

    plt.tight_layout()
    plt.savefig(path_to_cm)
    plt.close()


