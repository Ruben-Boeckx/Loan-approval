import matplotlib.pyplot as plt

def plot_precision_recall_curves(recall_list, precision_list, pr_auc_list):
    """
    Plots precision-recall curves for each fold along with AUC values.

    Parameters:
    recall_list (list of lists): List of recall values for each fold.
    precision_list (list of lists): List of precision values for each fold.
    pr_auc_list (list): List of AUC values for each fold.
    """
    plt.figure(figsize=(12, 8))
    
    for fold in range(len(pr_auc_list)):
        plt.plot(recall_list[fold], precision_list[fold], label=f"Fold {fold} (AUC = {pr_auc_list[fold]:.2f})")
    
    plt.axhline(y=0.5, color='k', linestyle='--', label="Random Guess")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid()
    plt.show()

def plot_roc_auc_curves(fpr_list, tpr_list, roc_auc_list):
    """
    Plots ROC-AUC curves for each fold along with AUC values.

    Parameters:
    fpr_list (list of lists): List of false positive rate values for each fold.
    tpr_list (list of lists): List of true positive rate values for each fold.
    roc_auc_list (list): List of AUC values for each fold.
    """
    plt.figure(figsize=(12, 8))
    
    for fold in range(len(roc_auc_list)):
        plt.plot(fpr_list[fold], tpr_list[fold], label=f"Fold {fold} (AUC = {roc_auc_list[fold]:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.title("ROC-AUC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
