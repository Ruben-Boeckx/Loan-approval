import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional
import warnings

class OptimalBinning:
    def __init__(
        self, 
        min_bins: int = 2, 
        max_bins: int = 10, 
        min_samples_bin: float = 0.05, 
        monotonic_trend: Optional[str] = None, 
        p_value_threshold: float = 0.05,
        special_values: Optional[List[Union[int, float, str]]] = None,
        variable_type: str = "continuous"
    ):
        """
        Initialize the OptimalBinning class for discretizing continuous or categorical variables.
        """
        self._validate_inputs(min_bins, max_bins, min_samples_bin, monotonic_trend, p_value_threshold, variable_type)
        
        self.min_bins = min_bins
        self.max_bins = max_bins
        self.min_samples_bin = min_samples_bin
        self.monotonic_trend = monotonic_trend
        self.p_value_threshold = p_value_threshold
        self.special_values = special_values if special_values is not None else []
        self.variable_type = variable_type
        
        self.bins = None
        self.binning_table = None
        self.is_integer = False
        self.iv_total = None
        self.special_bins = {}
        self._fitted = False

    def _validate_inputs(self, min_bins, max_bins, min_samples_bin, monotonic_trend, p_value_threshold, variable_type):
        """Validate input parameters."""
        if not isinstance(min_bins, int) or min_bins < 2:
            raise ValueError("min_bins must be an integer >= 2")
        if not isinstance(max_bins, int) or max_bins < min_bins:
            raise ValueError("max_bins must be an integer >= min_bins")
        if not 0 < min_samples_bin < 1:
            raise ValueError("min_samples_bin must be between 0 and 1")
        if monotonic_trend not in [None, 'ascending', 'descending', 'peak', 'valley']:
            raise ValueError("monotonic_trend must be one of: None, 'ascending', 'descending', 'peak', 'valley'")
        if not 0 < p_value_threshold < 1:
            raise ValueError("p_value_threshold must be between 0 and 1")
        if variable_type not in ["continuous", "categorical"]:
            raise ValueError("variable_type must be 'continuous' or 'categorical'")

    def _pre_binning(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Pre-binning to create initial split points or groups.
        """
        if self.variable_type == "continuous":
            self.is_integer = np.all(np.equal(np.mod(X, 1), 0))
            return self._pre_binning_continuous(X, y)
        else:
            return self._pre_binning_categorical(X, y)

    def _pre_binning_continuous(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pre-binning for continuous variables."""
        tree = DecisionTreeClassifier(
            max_leaf_nodes=self.max_bins, 
            min_samples_leaf=max(1, int(len(X) * self.min_samples_bin)), 
            random_state=42, 
            criterion='entropy'
        )
        
        try:
            tree.fit(X.reshape(-1, 1), y)
            thresholds = np.sort(tree.tree_.threshold[tree.tree_.threshold != -2])
            if self.is_integer:
                thresholds = np.ceil(thresholds).astype(int)
                thresholds = np.unique(thresholds)
            return thresholds
        except Exception as e:
            warnings.warn(f"Decision tree binning failed: {str(e)}. Falling back to equidistant binning.")
            min_val, max_val = np.min(X), np.max(X)
            if min_val == max_val:
                return np.array([])
            if self.is_integer:
                step = max(1, int((max_val - min_val) / self.min_bins))
                return np.arange(int(min_val) + step, int(max_val), step)
            else:
                return np.linspace(min_val, max_val, self.min_bins + 1)[1:-1]

    def _pre_binning_categorical(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pre-binning for categorical variables."""
        unique_categories = np.unique(X)
        category_stats = [{
            'category': cat,
            'count': np.sum(X == cat),
            'events': np.sum(y[X == cat]),
            'event_rate': np.sum(y[X == cat]) / np.sum(X == cat) if np.sum(X == cat) > 0 else 0
        } for cat in unique_categories]
        
        category_stats.sort(key=lambda x: x['event_rate'])
        return self._chaid_split(category_stats)

    def _chaid_split(self, category_stats, max_depth=3, min_p_value=0.05):
        """CHAID-like algorithm for finding optimal category groupings."""
        if len(category_stats) <= 1 or max_depth == 0:
            return [tuple(item['category'] for item in category_stats)]
        
        best_p_value = 1.0
        best_split_idx = None
        
        for i in range(1, len(category_stats)):
            left = category_stats[:i]
            right = category_stats[i:]
            
            contingency = np.array([
                [sum(item['count'] - item['events'] for item in left), sum(item['events'] for item in left)],
                [sum(item['count'] - item['events'] for item in right), sum(item['events'] for item in right)]
            ])
            
            try:
                _, p_value, _, _ = chi2_contingency(contingency)
                if p_value < best_p_value:
                    best_p_value = p_value
                    best_split_idx = i
            except:
                continue
        
        if best_split_idx is None or best_p_value > min_p_value:
            return [tuple(item['category'] for item in category_stats)]
        
        left_groups = self._chaid_split(category_stats[:best_split_idx], max_depth - 1, min_p_value)
        right_groups = self._chaid_split(category_stats[best_split_idx:], max_depth - 1, min_p_value)
        return left_groups + right_groups

    def _calculate_woe_iv(self, events: int, non_events: int, total_events: int, total_non_events: int) -> Tuple[float, float, float]:
        """Calculate Weight of Evidence (WoE) and Information Value (IV) for a bin."""
        epsilon = 1e-10
        events_smooth = events + epsilon
        non_events_smooth = non_events + epsilon
        woe = np.log((non_events_smooth / total_non_events) / (events_smooth / total_events))
        iv = ((non_events / total_non_events) - (events / total_events)) * woe
        event_rate = events / (events + non_events) if (events + non_events) > 0 else 0
        return woe, iv, event_rate

    def _handle_special_values(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Handle special values specified by the user."""
        special_bins = {}
        if not self.special_values:
            return X, y, special_bins
        
        keep_indices = np.ones(len(X), dtype=bool)
        total_events = np.sum(y)
        total_non_events = len(y) - total_events
        
        for special_value in self.special_values:
            special_mask = np.isclose(X, special_value) if self.variable_type == "continuous" else (X == special_value)
            special_data = y[special_mask]
            
            if len(special_data) > 0:
                events = np.sum(special_data)
                non_events = len(special_data) - events
                woe, iv, event_rate = self._calculate_woe_iv(events, non_events, total_events, total_non_events)
                special_bins[special_value] = {
                    'Bin': f"Special: {special_value}",
                    'Count': len(special_data),
                    'Non-event': non_events,
                    'Event': events,
                    'Event rate': event_rate,
                    'WoE': woe,
                    'IV': iv
                }
                keep_indices = keep_indices & ~special_mask
        
        return X[keep_indices], y[keep_indices], special_bins

    def _merge_bins(self, X: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> Tuple[List[Tuple], List[Dict]]:
        """Create bins based on thresholds or grouped categories."""
        total_events = np.sum(y)
        total_non_events = len(y) - total_events
        bins = []
        binning_table = []

        if self.variable_type == "continuous":
            if len(thresholds) == 0:
                bin_data = y
                woe, iv, event_rate = self._calculate_woe_iv(np.sum(bin_data), len(bin_data) - np.sum(bin_data), total_events, total_non_events)
                bins.append((-np.inf, np.inf))
                binning_table.append({
                    'Bin': f"[-inf, inf)",
                    'Count': len(bin_data),
                    'Non-event': len(bin_data) - np.sum(bin_data),
                    'Event': np.sum(bin_data),
                    'Event rate': event_rate,
                    'WoE': woe,
                    'IV': iv
                })
                return bins, binning_table

            for i in range(len(thresholds) + 1):
                if i == 0:
                    bin_data = y[X < thresholds[i]]
                    lower_bound, upper_bound = -np.inf, thresholds[i]
                elif i == len(thresholds):
                    bin_data = y[X >= thresholds[i - 1]]
                    lower_bound, upper_bound = thresholds[i - 1], np.inf
                else:
                    bin_data = y[(X >= thresholds[i - 1]) & (X < thresholds[i])]
                    lower_bound, upper_bound = thresholds[i - 1], thresholds[i]

                if len(bin_data) == 0:
                    continue
                    
                woe, iv, event_rate = self._calculate_woe_iv(np.sum(bin_data), len(bin_data) - np.sum(bin_data), total_events, total_non_events)
                if self.is_integer:
                    lower_bound = int(lower_bound) if lower_bound != -np.inf else lower_bound
                    upper_bound = int(upper_bound) if upper_bound != np.inf else upper_bound
                        
                bins.append((lower_bound, upper_bound))
                binning_table.append({
                    'Bin': f"[{lower_bound}, {upper_bound})",
                    'Count': len(bin_data),
                    'Non-event': len(bin_data) - np.sum(bin_data),
                    'Event': np.sum(bin_data),
                    'Event rate': event_rate,
                    'WoE': woe,
                    'IV': iv
                })
        else:
            for group in thresholds:
                bin_data = y[np.isin(X, group)]
                if len(bin_data) == 0:
                    continue
                    
                woe, iv, event_rate = self._calculate_woe_iv(np.sum(bin_data), len(bin_data) - np.sum(bin_data), total_events, total_non_events)
                bins.append(tuple(group))
                binning_table.append({
                    'Bin': f"{group}",
                    'Count': len(bin_data),
                    'Non-event': len(bin_data) - np.sum(bin_data),
                    'Event': np.sum(bin_data),
                    'Event rate': event_rate,
                    'WoE': woe,
                    'IV': iv
                })

        return bins, binning_table

    def _enforce_monotonicity(self, binning_table: List[Dict]) -> List[Dict]:
        """Enforce monotonicity constraints by merging violating bins."""
        if self.monotonic_trend not in ['ascending', 'descending', 'peak', 'valley']:
            return binning_table
            
        if self.monotonic_trend in ['peak', 'valley']:
            event_rates = [row['Event rate'] for row in binning_table]
            peak_idx = np.argmax(event_rates) if self.monotonic_trend == 'peak' else np.argmin(event_rates)
            left_bins = binning_table[:peak_idx+1]
            right_bins = binning_table[peak_idx+1:]
            left_bins = self._enforce_simple_monotonicity(left_bins, 'ascending' if self.monotonic_trend == 'peak' else 'descending')
            right_bins = self._enforce_simple_monotonicity(right_bins, 'descending' if self.monotonic_trend == 'peak' else 'ascending')
            return left_bins + right_bins
        else:
            return self._enforce_simple_monotonicity(binning_table, self.monotonic_trend)
            
    def _enforce_simple_monotonicity(self, binning_table: List[Dict], trend: str) -> List[Dict]:
        """Helper method to enforce simple monotonicity (ascending or descending)."""
        if not binning_table:
            return binning_table
            
        merged = True
        while merged and len(binning_table) > 1:
            merged = False
            event_rates = [row['Event rate'] for row in binning_table]
            
            if trend == 'ascending':
                for i in range(len(event_rates) - 1):
                    if event_rates[i] > event_rates[i + 1]:
                        self._merge_adjacent_bins(binning_table, i)
                        merged = True
                        break
            elif trend == 'descending':
                for i in range(len(event_rates) - 1):
                    if event_rates[i] < event_rates[i + 1]:
                        self._merge_adjacent_bins(binning_table, i)
                        merged = True
                        break
        
        self._recalculate_metrics(binning_table)
        return binning_table
        
    def _merge_adjacent_bins(self, binning_table: List[Dict], idx: int) -> None:
        """Merge two adjacent bins in the binning table."""
        binning_table[idx]['Count'] += binning_table[idx + 1]['Count']
        binning_table[idx]['Non-event'] += binning_table[idx + 1]['Non-event']
        binning_table[idx]['Event'] += binning_table[idx + 1]['Event']
        binning_table[idx]['Event rate'] = binning_table[idx]['Event'] / binning_table[idx]['Count'] if binning_table[idx]['Count'] > 0 else 0
        
        bin1_range = binning_table[idx]['Bin'].strip('[]()')
        bin2_range = binning_table[idx + 1]['Bin'].strip('[]()')
        try:
            lower1 = bin1_range.split(',')[0].strip()
            upper2 = bin2_range.split(',')[1].strip()
            binning_table[idx]['Bin'] = f"[{lower1}, {upper2})"
        except:
            binning_table[idx]['Bin'] = f"Merged: {binning_table[idx]['Bin']} + {binning_table[idx + 1]['Bin']}"
        
        del binning_table[idx + 1]

    def _recalculate_metrics(self, binning_table: List[Dict]) -> None:
        """Recalculate WoE and IV for all bins in the binning table."""
        total_events = sum(row['Event'] for row in binning_table)
        total_non_events = sum(row['Non-event'] for row in binning_table)
        
        for row in binning_table:
            events = row['Event']
            non_events = row['Non-event']
            woe, iv, _ = self._calculate_woe_iv(events, non_events, total_events, total_non_events)
            row['WoE'] = woe
            row['IV'] = iv

    def _check_p_value(self, binning_table: List[Dict]) -> List[Dict]:
        """Check p-value constraints and merge bins with similar distributions."""
        if len(binning_table) <= 1:
            return binning_table
            
        merged_bins = set()
        i = 0
        
        while i < len(binning_table) - 1:
            bin1 = binning_table[i]
            bin2 = binning_table[i + 1]
            
            if i in merged_bins or i + 1 in merged_bins:
                i += 1
                continue
                
            contingency_table = [
                [bin1['Non-event'], bin1['Event']], 
                [bin2['Non-event'], bin2['Event']]
            ]
            
            row_sums = [sum(row) for row in contingency_table]
            col_sums = [sum(col) for col in zip(*contingency_table)]
            total = sum(row_sums)
            
            if total == 0:
                i += 1
                continue
                
            expected_too_low = any(row_sums[r] * col_sums[c] / total < 5 for r in range(2) for c in range(2))
            if expected_too_low:
                self._merge_adjacent_bins(binning_table, i)
                merged_bins.add(i + 1)
                continue
            
            try:
                _, p_value, _, _ = chi2_contingency(contingency_table)
                if p_value > self.p_value_threshold:
                    self._merge_adjacent_bins(binning_table, i)
                    merged_bins.add(i + 1)
                else:
                    i += 1
            except Exception:
                self._merge_adjacent_bins(binning_table, i)
                merged_bins.add(i + 1)
        
        self._recalculate_metrics(binning_table)
        return binning_table

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OptimalBinning':
        """Fit the optimal binning model."""
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if self.variable_type == "continuous":
            if np.isnan(X).any():
                raise ValueError("X contains NaN values. Please handle missing values before fitting.")
        else:
            if any(pd.isna(X)):
                raise ValueError("X contains missing values. Please handle missing values before fitting.")
        
        if np.isnan(y).any():
            raise ValueError("y contains NaN values. Please handle missing values before fitting.")
        
        unique_y = np.unique(y)
        if len(unique_y) > 2 or not np.all(np.isin(unique_y, [0, 1])):
            raise ValueError("y must be binary (0 or 1)")
        
        filtered_X, filtered_y, self.special_bins = self._handle_special_values(X, y)
        
        if len(filtered_X) == 0:
            self.bins = [(-np.inf, np.inf)]
            self.binning_table = [{
                'Bin': "[-inf, inf)",
                'Count': 0,
                'Non-event': 0,
                'Event': 0,
                'Event rate': 0,
                'WoE': 0,
                'IV': 0
            }]
            self._fitted = True
            self.iv_total = 0
            return self
            
        if self.variable_type == "continuous":
            self.is_integer = np.all(np.equal(np.mod(filtered_X, 1), 0))
        
        thresholds = self._pre_binning(filtered_X, filtered_y)
        self.bins, self.binning_table = self._merge_bins(filtered_X, filtered_y, thresholds)
        
        if self.monotonic_trend:
            self.binning_table = self._enforce_monotonicity(self.binning_table)
        
        self.binning_table = self._check_p_value(self.binning_table)
        
        for special_info in self.special_bins.values():
            self.binning_table.append(special_info)
        
        if len(self.binning_table) < self.min_bins and len(np.unique(X)) >= self.min_bins:
            warnings.warn(
                f"Resulting number of bins ({len(self.binning_table)}) "
                f"is less than min_bins ({self.min_bins}). "
                "Consider adjusting parameters."
            )
    
        self.bins = []
        for row in self.binning_table:
            if "Special:" in row['Bin'] or "Merged:" in row['Bin']:
                continue
                
            if self.variable_type == "continuous":
                bin_range = row['Bin'].strip('[]()')
                lower, upper = bin_range.split(',')
                lower = float(lower.strip()) if lower.strip() != '-inf' else -np.inf
                upper = float(upper.strip()) if upper.strip() != 'inf' else np.inf
                
                if self.is_integer:
                    if lower != -np.inf:
                        lower = int(lower)
                    if upper != np.inf:
                        upper = int(upper)
                        
                self.bins.append((lower, upper))
            else:
                self.bins.append(row['Bin'])
        
        self.iv_total = sum(row['IV'] for row in self.binning_table)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data into binned values (bin labels)."""
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'transform'.")
            
        X = np.asarray(X).flatten()
        binned_X = np.full_like(X, fill_value="", dtype=object)
        
        for special_value in self.special_values:
            special_mask = np.isclose(X, special_value)
            if special_value in self.special_bins:
                binned_X[special_mask] = self.special_bins[special_value]['Bin']
        
        for lower, upper in self.bins:
            if lower == -np.inf:
                mask = (X < upper) & (binned_X == "")
            elif upper == np.inf:
                mask = (X >= lower) & (binned_X == "")
            else:
                mask = (X >= lower) & (X < upper) & (binned_X == "")
                
            binned_X[mask] = f"[{lower}, {upper})"
        
        remaining_mask = binned_X == ""
        if np.any(remaining_mask):
            warnings.warn(f"{np.sum(remaining_mask)} values did not fit into any bin and will be labeled as 'Other'.")
            binned_X[remaining_mask] = "Other"
            
        return binned_X

    def apply_binning(self, X: np.ndarray) -> np.ndarray:
        """Apply binning and return WoE values."""
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'apply_binning'.")
            
        X = np.asarray(X).flatten()
        binned_X = self.transform(X)
        woe_dict = {row['Bin']: row['WoE'] for row in self.binning_table}
        woe_X = np.array([woe_dict.get(bin_value, 0) for bin_value in binned_X])
        return woe_X

    def get_binning_table(self) -> pd.DataFrame:
        """Return the binning table as a pandas DataFrame with summary row."""
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'get_binning_table'.")
            
        df = pd.DataFrame(self.binning_table)
        
        # Calculate totals for summary row
        total_count = df['Count'].sum()
        total_non_event = df['Non-event'].sum()
        total_event = df['Event'].sum()
        
        # Calculate overall event rate
        total_event_rate = total_event / total_count if total_count > 0 else 0
        
        # Create IV percentage column
        df['IV_percentage'] = (df['IV'] / self.iv_total * 100).round(2) if self.iv_total > 0 else 0.0
        
        # Create summary row
        summary_row = pd.DataFrame([{
            'Bin': 'Total',
            'Count': total_count,
            'Non-event': total_non_event,
            'Event': total_event,
            'Event rate': total_event_rate,
            'WoE': "",
            'IV': self.iv_total,
            'IV_percentage': 100.0  # Total IV is 100% of total IV
        }])
        
        # Append the summary row to the dataframe
        df = pd.concat([df, summary_row], ignore_index=True)
        
        return df

    def plot_binning(self, figsize=(10, 6)):
        """Visualize the binning results with a bar plot for event/non-event distribution and a line plot for WoE."""
        if not self._fitted:
            raise ValueError("Model not fitted yet. Call 'fit' before using 'plot_binning'.")
        
        if not self.binning_table:
            warnings.warn("Empty binning table, nothing to plot.")
            return
        
        summary = pd.DataFrame(self.binning_table)
        rounded_bin_labels = []
        for bin_range in summary['Bin']:
            if "Special:" in bin_range or "Merged:" in bin_range or "Other" in bin_range:
                rounded_bin_labels.append(bin_range)
                continue
            
            try:
                lower, upper = bin_range.strip('[]()').split(',')
                lower = lower.strip()
                upper = upper.strip()
                
                lower_fmt = "-∞" if lower == '-inf' else format(float(lower), '.2g') if not self.is_integer or lower == '-inf' else str(int(float(lower)))
                upper_fmt = "∞" if upper == 'inf' else format(float(upper), '.2g') if not self.is_integer or upper == 'inf' else str(int(float(upper)))
                rounded_bin_labels.append(f"[{lower_fmt}, {upper_fmt})")
            except:
                rounded_bin_labels.append(bin_range)
        
        fig, ax1 = plt.subplots(figsize=figsize)
        counts = pd.DataFrame({
            'Non-Event': summary['Non-event'],
            'Event': summary['Event']
        })
        counts.plot(kind='bar', stacked=True, ax=ax1, colormap="coolwarm", alpha=0.7)
        plt.xticks(range(len(summary.index)), rounded_bin_labels, rotation=45, ha='right')
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

    def __repr__(self) -> str:
        """Return a string representation of the fitted binning model."""
        if not self._fitted:
            return "OptimalBinning (not fitted)"
            
        summary = f"OptimalBinning (fitted)\n"
        summary += f"  Number of bins: {len(self.binning_table)}\n"
        summary += f"  Total IV: {self.iv_total:.4f}\n"
        if self.monotonic_trend:
            summary += f"  Monotonic Trend: {self.monotonic_trend}\n"
        if self.special_values:
            summary += f"  Special Values: {self.special_values}\n"

        return summary