from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted


def create_binned_feature(df, feature, target, max_leaf_nodes, min_samples_leaf=0.05, random_state=42):
    """
    Create bins using a decision tree with enforced monotonicity and a minimum sample size per bin.
    """
    min_samples_leaf_count = int(min_samples_leaf * len(df))
    
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes, 
        min_samples_leaf=min_samples_leaf_count, 
        random_state=random_state,
        criterion='entropy'
    )
    
    tree.fit(df[[feature]], df[target])
    
    df_copy = df.copy()
    df_copy[feature + '_bin'] = tree.apply(df[[feature]])
    
    bin_stats = df_copy.groupby(feature + '_bin')[feature].agg(['min', 'max']).round(2)
    bin_labels = {
        row_idx: f'[{row["min"]:.2f}-{row["max"]:.2f}]'
        for row_idx, row in bin_stats.iterrows()
    }
    
    df_copy[feature + '_bin_label'] = df_copy[feature + '_bin'].map(bin_labels)
    
    return df_copy, bin_stats, bin_labels

def calculate_woe(df, feature, target, event=1, plot=False, figsize=(12, 6), bin_labels=None):
    """
    Calculate and optionally plot Weight of Evidence (WoE) for a feature.
    """
    try:
        feature_bin = feature + '_bin_label'
        
        if feature_bin not in df.columns or target not in df.columns:
            raise ValueError(f"Columns {feature_bin} and/or {target} not found in DataFrame")
            
        if not set(df[target].unique()).issubset({0, 1}):
            raise ValueError("Target variable should be binary (0/1)")
            
        cross_tab = pd.crosstab(df[feature_bin], df[target])
        
        event_prop = cross_tab[event] / cross_tab[event].sum()
        non_event_prop = cross_tab[1 - event] / cross_tab[1 - event].sum()
        
        woe = np.log(non_event_prop / event_prop)
        
        summary = pd.DataFrame({
            'non_event_count': cross_tab[1 - event],
            'event_count': cross_tab[event],
            'non_event_rate': non_event_prop,
            'event_rate': event_prop,
            'WoE': woe
        }).round(4)
        
        summary['IV'] = (non_event_prop - event_prop) * woe
        total_iv = summary['IV'].sum()

        df[f'{feature}_woe'] = df[feature_bin].map(summary['WoE'])
        
        summary = summary.sort_values('WoE', ascending=False)
        
        if plot:
            fig, ax1 = plt.subplots(figsize=figsize)
            
            counts = pd.DataFrame({
                'Non-Event': summary['non_event_count'],
                'Event': summary['event_count']
            })
            counts.plot(kind='bar', stacked=True, ax=ax1, colormap="coolwarm", alpha=0.7)
            
            plt.xticks(range(len(summary.index)), summary.index, rotation=45, ha='right')
            
            ax1.set_xlabel("Bins (Ranges)")
            ax1.set_ylabel("Number of Observations")
            ax1.set_title(f"Distribution and WoE Analysis\n(Sorted by WoE)")
            
            ax2 = ax1.twinx()
            line_plot, = ax2.plot(range(len(summary.index)), summary['WoE'].values, 
                    marker='o', color='black', linestyle='-', linewidth=2, 
                    label="Weight of Evidence")
            
            ax2.set_ylabel("Weight of Evidence")
            
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = [line_plot], ["Weight of Evidence"]
            
            ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=10, framealpha=0.8)
            
            plt.tight_layout()
            plt.show()
        
        df.drop(columns=[feature_bin, feature + '_bin'], inplace=True)

        
        return df, summary, total_iv
        
    except Exception as e:
        print(f"Error in WoE calculation: {str(e)}")
        return None, None


def optimize_decision_tree_leaves(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    max_leaf_nodes_range: List[int],
    min_samples_leaf: float = 0.05,
    cv_folds: int = 5,
    random_state: Optional[int] = 42
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """
    Optimize the max_leaf_nodes parameter for a decision tree classifier using cross-validation.
    
    Args:
        df: DataFrame containing the features and target
        feature_cols: List of feature column names
        target_col: Name of the target column
        max_leaf_nodes_range: List of max_leaf_nodes values to test
        min_samples_leaf: Minimum samples per leaf as a fraction of total samples
        cv_folds: Number of cross-validation folds
        random_state: Random state for reproducibility
    
    Returns:
        Tuple containing:
        - Best max_leaf_nodes value
        - Best mean ROC AUC score
        - List of (max_leaf_nodes, score) tuples for all trials
    
    Raises:
        ValueError: If input parameters are invalid
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame")
    if not all(col in df.columns for col in feature_cols + [target_col]):
        raise ValueError("Some specified columns not found in DataFrame")
    if not (0 < min_samples_leaf < 1):
        raise ValueError("min_samples_leaf must be between 0 and 1")
    
    min_samples_leaf_count = int(min_samples_leaf * len(df))
    best_score = -np.inf
    best_max_leaf_nodes = None
    results = []

    X = df[feature_cols]
    y = df[target_col]

    for max_leaf_nodes in max_leaf_nodes_range:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf_count,
            random_state=random_state,
            criterion='entropy'
        )

        scores = cross_val_score(
            tree, 
            X, 
            y, 
            cv=cv_folds, 
            scoring='roc_auc',
            n_jobs=-1  # Use all available CPU cores
        )
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results.append((max_leaf_nodes, mean_score, std_score))
        
        if mean_score > best_score:
            best_score = mean_score
            best_max_leaf_nodes = max_leaf_nodes

    # Print results with confidence intervals
    print("\nResults for different max_leaf_nodes values (ROC AUC):")
    for max_leaf_nodes, mean, std in results:
        print(f"Max leaf nodes: {max_leaf_nodes:2d}, "
              f"ROC AUC: {mean:.4f} ± {std:.4f}")

    print(f"\nBest max_leaf_nodes: {best_max_leaf_nodes} "
          f"with ROC AUC: {best_score:.4f}")
    
    return best_max_leaf_nodes, best_score, results

def plot_tree_performance(
    results: List[Tuple[int, float, float]], 
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
) -> None:
    """
    Plot the performance metrics of decision tree with error bars.
    
    Args:
        results: List of tuples containing (max_leaf_nodes, mean_score, std_score)
        figsize: Tuple specifying figure dimensions
        save_path: Optional path to save the plot
    """
    leaf_nodes, means, stds = zip(*results)
    
    plt.figure(figsize=figsize)
    
    # Plot mean scores with error bars
    plt.errorbar(
        leaf_nodes, 
        means, 
        yerr=stds,
        fmt='o-',
        capsize=5,
        capthick=1.5,
        elinewidth=1.5,
        markersize=8,
        color='#2E86C1',
        label='Mean ROC AUC with std dev'
    )
    
    # Find and mark the best performer
    best_idx = np.argmax(means)
    plt.plot(
        leaf_nodes[best_idx],
        means[best_idx],
        'r*',
        markersize=15,
        label=f'Best: {leaf_nodes[best_idx]} nodes\nROC AUC: {means[best_idx]:.4f}'
    )
    
    # Customize the plot
    plt.xlabel('Maximum Leaf Nodes', fontsize=12)
    plt.ylabel('ROC AUC Score', fontsize=12)
    plt.title('Decision Tree Performance vs Max Leaf Nodes', fontsize=14, pad=15)
    
    # Add grid with custom style
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize axis
    plt.xticks(leaf_nodes)
    plt.ylim(
        min(means) - max(stds) - 0.02,
        max(means) + max(stds) + 0.02
    )
    
    # Add legend
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1, 0),
        fontsize=10
    )
    
    # Ensure layout fits well
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()