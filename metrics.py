import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted

def calculate_woe(df, feature, target, event=1, plot=False, figsize=(12, 6)):
    """
    Calculate and optionally plot Weight of Evidence (WoE) for a feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    feature : str
        Name of the feature column (predictor variable)
    target : str
        Name of the target column (binary: 0/1)
    event : int, default=1
        Value that represents the event in target variable
    plot : bool, default=False
        Whether to create a visualization
    figsize : tuple, default=(12, 6)
        Figure size for the plot
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing bins, counts, proportions, and WoE values
    float
        Total Information Value
    """
    
    try:
        # Validate inputs
        if not all(col in df.columns for col in [feature, target]):
            raise ValueError(f"Columns {feature} and/or {target} not found in DataFrame")
            
        if not set(df[target].unique()).issubset({0, 1}):
            raise ValueError("Target variable should be binary (0/1)")
            
        # Calculate frequencies
        cross_tab = pd.crosstab(df[feature], df[target])
        
        # Calculate proportions
        event_prop = cross_tab[event] / cross_tab[event].sum()
        non_event_prop = cross_tab[1 - event] / cross_tab[1 - event].sum()
        
        # Calculate WoE
        woe = np.log(non_event_prop / event_prop)
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'non_event_count': cross_tab[1 - event],
            'event_count': cross_tab[event],
            'non_event_rate': non_event_prop,
            'event_rate': event_prop,
            'WoE': woe
        }).round(4)
        
        # Calculate Information Value (IV)
        summary['IV'] = (non_event_prop - event_prop) * woe
        total_iv = summary['IV'].sum()
        
        # Sort the index using natural sorting
        summary = summary.loc[natsorted(summary.index)]
        
        if plot:
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Bar plot for event and non-event counts
            counts = pd.DataFrame({
                'Non-Event': summary['non_event_count'],
                'Event': summary['event_count']
            })
            counts.plot(kind='bar', stacked=True, ax=ax1, colormap="coolwarm", alpha=0.7)
            ax1.set_xlabel(feature)
            ax1.set_ylabel("Number of Observations")
            ax1.set_title(f"Distribution and WoE Analysis for {feature}")
            
            # Create second y-axis for WoE values
            ax2 = ax1.twinx()
            ax2.plot(range(len(summary.index)), summary['WoE'].values, 
                    marker='o', color='black', linestyle='-', linewidth=2, 
                    label="Weight of Evidence")
            
            ax2.set_ylabel("Weight of Evidence")
            
            # Show legends
            ax1.legend(["Non-Event (0)", "Event (1)"], loc="upper left")
            ax2.legend(loc="upper right")
            
            # Rotate x-labels if needed
            if len(str(summary.index[0])) > 10:
                plt.xticks(rotation=0)
            
            plt.tight_layout()
            plt.show()
        
        return summary, total_iv
        
    except Exception as e:
        print(f"Error in WoE calculation: {str(e)}")
        return None, None
    
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
