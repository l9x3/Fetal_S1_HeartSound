"""
Production-Ready Improved Median-Based Learner
==============================================

Implements recommended improvements with minimal complexity:
1. Robust feature selection (Spearman correlation)
2. Class weighting for imbalance
3. Optimal decision threshold tuning
4. Clean, production-ready code
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class RobustFeatureSelector:
    """Select features using rank-based correlation (robust to outliers)."""
    
    def __init__(self, n_features=60, min_variance=0.01):
        self.n_features = n_features
        self.min_variance = min_variance
        self.selected_features_ = None
        self.feature_correlations_ = None
        self.scaler_ = None
    
    def fit(self, X, y):
        """Select robust features."""
        # Scale features
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Compute Spearman correlations (rank-based, robust to outliers)
        correlations = []
        for i in range(X_scaled.shape[1]):
            corr, _ = spearmanr(X_scaled[:, i], y)
            correlations.append(np.abs(corr))
        
        self.feature_correlations_ = np.array(correlations)
        
        # Select top N features by correlation
        top_indices = np.argsort(correlations)[-self.n_features:]
        
        # Remove low-variance features
        variances = np.var(X_scaled[:, top_indices], axis=0)
        high_var_mask = variances > self.min_variance
        self.selected_features_ = top_indices[high_var_mask]
        
        return self
    
    def transform(self, X):
        """Transform using selected features."""
        if self.selected_features_ is None:
            raise ValueError("Not fitted")
        return X[:, self.selected_features_]
    
    def fit_transform(self, X, y):
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class ImprovedMedianBasedRobustLearner:
    """Production-ready median-based learner with improvements."""
    
    def __init__(self, n_iterations=5, robust_blend=0.7, 
                 regularization=0.001, class_weight='balanced'):
        self.n_iterations = n_iterations
        self.robust_blend = robust_blend
        self.regularization = regularization
        self.class_weight = class_weight
        
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
        self.class_weights_ = None
        self.feature_importance_ = None
    
    def _compute_class_weights(self, y):
        """Compute weights for imbalanced data."""
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            weights = len(y) / (2 * counts)
            return dict(zip(unique, weights / np.sum(weights)))
        else:
            return {-1: 1.0, 1: 1.0}
    
    def _median_absolute_deviation(self, residuals):
        """Calculate MAD."""
        return np.median(np.abs(residuals - np.median(residuals)))
    
    def _weight_bisquare(self, standardized_residuals):
        """Tukey's bisquare weight function."""
        c = 4.685
        u = standardized_residuals / c
        return np.where(np.abs(u) <= 1, (1 - u**2)**2, 0)
    
    def fit(self, X, y):
        """Fit the learner."""
        # Convert labels
        unique_labels = np.unique(y)
        if 0 in unique_labels and 1 in unique_labels:
            y = np.where(y == 0, -1, 1)
        
        # Compute class weights
        self.class_weights_ = self._compute_class_weights(y)
        
        # Scale features
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        
        # Solve robust problem with iterative re-weighting
        w_robust = np.zeros(n_features)
        b_robust = np.median(y)
        
        sample_weights = np.ones(n_samples)
        
        for iteration in range(self.n_iterations):
            # Apply class weights
            effective_weights = sample_weights.copy()
            for label in [-1, 1]:
                mask = y == label
                if np.any(mask):
                    effective_weights[mask] *= self.class_weights_.get(label, 1.0)
            
            # Normalize
            effective_weights = effective_weights / np.mean(effective_weights)
            
            # Weighted least squares
            sqrt_w = np.sqrt(effective_weights)
            X_weighted = X_scaled * sqrt_w[:, np.newaxis]
            y_weighted = y * sqrt_w
            X_with_bias = np.column_stack([X_weighted, sqrt_w])
            
            try:
                params, _, _, _ = np.linalg.lstsq(X_with_bias, y_weighted, rcond=None)
                w_robust = params[:-1]
                b_robust = params[-1]
            except:
                continue
            
            # Update weights
            z = X_scaled.dot(w_robust) + b_robust
            predictions = np.sign(z)
            errors = np.abs(y - predictions)
            
            mad = self._median_absolute_deviation(errors)
            if mad > 1e-10:
                standardized_errors = errors / mad
                sample_weights = self._weight_bisquare(standardized_errors)
            
            sample_weights = np.clip(sample_weights, 1e-4, 1.0)
        
        # Solve standard problem
        X_with_bias = np.column_stack([X_scaled, np.ones(n_samples)])
        params_std, _, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        w_std = params_std[:-1]
        b_std = params_std[-1]
        
        # Blend
        self.coef_ = self.robust_blend * w_robust + (1 - self.robust_blend) * w_std
        self.intercept_ = self.robust_blend * b_robust + (1 - self.robust_blend) * b_std
        self.feature_importance_ = np.abs(self.coef_)
        
        return self
    
    def decision_function(self, X):
        """Compute decision function."""
        if self.coef_ is None:
            raise ValueError("Not fitted")
        X_scaled = self.scaler_.transform(X)
        return X_scaled.dot(self.coef_) + self.intercept_
    
    def predict(self, X):
        """Predict with default threshold."""
        return np.sign(self.decision_function(X))
    
    def predict_with_threshold(self, X, threshold=0.0):
        """Predict with custom threshold."""
        scores = self.decision_function(X)
        return np.where(scores >= threshold, 1, -1)


def optimize_threshold(y_true, y_scores, metric='f1'):
    """Find optimal decision threshold."""
    thresholds = np.linspace(np.min(y_scores), np.max(y_scores), 100)
    best_score = 0
    best_threshold = 0.0
    
    for threshold in thresholds:
        y_pred = np.where(y_scores >= threshold, 1, -1)
        
        if metric == 'f1':
            score = f1_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
        elif metric == 'accuracy':
            score = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
        else:
            score = accuracy_score((y_true > 0).astype(int), (y_pred > 0).astype(int))
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def load_and_prepare_data(csv_path):
    """Load data."""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    unique_vals = np.unique(y)
    if len(unique_vals) == 2 and set(unique_vals) == {-1, 1}:
        y_binary = y
    else:
        median_target = np.median(y)
        y_binary = np.where(y >= median_target, 1, -1)
    
    return X, y_binary


def main():
    """Main execution."""
    
    print("=" * 80)
    print("PRODUCTION-READY IMPROVED MEDIAN-BASED LEARNER")
    print("=" * 80)
    
    # Load data
    csv_path = '/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/dataset_experimental_200ms_50ms.csv'
    X, y = load_and_prepare_data(csv_path)
    
    print(f"\nDataset: {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    results = []
    
    # 1. Original median learner
    print("=" * 80)
    print("MODEL 1: Original Median-Based Learner")
    print("=" * 80)
    
    original = ImprovedMedianBasedRobustLearner()
    original.fit(X_train, y_train)
    y_pred_orig = original.predict(X_test)
    acc_orig = accuracy_score((y_test > 0).astype(int), (y_pred_orig > 0).astype(int))
    
    print(f"Accuracy: {acc_orig:.4f}")
    results.append(('Original', acc_orig, None))
    
    # 2. With robust feature selection
    print("\n" + "=" * 80)
    print("MODEL 2: With Robust Feature Selection")
    print("=" * 80)
    
    selector = RobustFeatureSelector(n_features=60)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"Selected {len(selector.selected_features_)} / {X_train.shape[1]} features")
    print(f"Reduced dimensionality: {X_train.shape[1]} → {X_train_selected.shape[1]}")
    
    model2 = ImprovedMedianBasedRobustLearner()
    model2.fit(X_train_selected, y_train)
    y_pred2 = model2.predict(X_test_selected)
    acc2 = accuracy_score((y_test > 0).astype(int), (y_pred2 > 0).astype(int))
    
    print(f"Accuracy: {acc2:.4f} ({(acc2-acc_orig)*100:+.2f}%)")
    results.append(('Feature Selection', acc2, None))
    
    # 3. With threshold optimization
    print("\n" + "=" * 80)
    print("MODEL 3: With Threshold Optimization")
    print("=" * 80)
    
    model3 = ImprovedMedianBasedRobustLearner()
    model3.fit(X_train, y_train)
    y_scores_train = model3.decision_function(X_train)
    
    opt_threshold, opt_score = optimize_threshold(y_train, y_scores_train, metric='f1')
    print(f"Optimal threshold: {opt_threshold:.4f} (F1: {opt_score:.4f})")
    
    y_scores_test = model3.decision_function(X_test)
    y_pred3 = model3.predict_with_threshold(X_test, opt_threshold)
    acc3 = accuracy_score((y_test > 0).astype(int), (y_pred3 > 0).astype(int))
    
    print(f"Accuracy: {acc3:.4f} ({(acc3-acc_orig)*100:+.2f}%)")
    results.append(('Threshold Optimization', acc3, opt_threshold))
    
    # 4. Combined: Feature selection + Threshold optimization
    print("\n" + "=" * 80)
    print("MODEL 4: Combined (Feature Selection + Threshold)")
    print("=" * 80)
    
    model4 = ImprovedMedianBasedRobustLearner()
    model4.fit(X_train_selected, y_train)
    y_scores_train4 = model4.decision_function(X_train_selected)
    
    opt_threshold4, _ = optimize_threshold(y_train, y_scores_train4, metric='f1')
    
    y_scores_test4 = model4.decision_function(X_test_selected)
    y_pred4 = model4.predict_with_threshold(X_test_selected, opt_threshold4)
    acc4 = accuracy_score((y_test > 0).astype(int), (y_pred4 > 0).astype(int))
    
    print(f"Accuracy: {acc4:.4f} ({(acc4-acc_orig)*100:+.2f}%)")
    results.append(('Combined Improvements', acc4, opt_threshold4))
    
    # 5. Standard SVM (baseline)
    print("\n" + "=" * 80)
    print("MODEL 5: Standard SVM (Baseline)")
    print("=" * 80)
    
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score((y_test > 0).astype(int), (y_pred_svm > 0).astype(int))
    
    print(f"Accuracy: {acc_svm:.4f}")
    results.append(('SVM Baseline', acc_svm, None))
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    for model_name, acc, threshold in results:
        vs_orig = (acc - acc_orig) * 100
        vs_svm = (acc - acc_svm) * 100
        marker = "🏆" if acc == max([r[1] for r in results]) else "  "
        threshold_str = f" (T={threshold:.4f})" if threshold is not None else ""
        print(f"{marker} {model_name:30s} {acc:.4f}  ({vs_orig:+6.2f}% vs orig, {vs_svm:+6.2f}% vs SVM){threshold_str}")
    
    best_improved, best_acc_imp, _ = max([(r[0], r[1], r[2]) for r in results if r[0] != 'SVM Baseline'], key=lambda x: x[1])
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"\nOriginal Median Learner: {acc_orig:.4f}")
    print(f"Best Improved Version: {best_acc_imp:.4f}")
    print(f"Total Improvement: {(best_acc_imp - acc_orig)*100:+.2f}%")
    print(f"vs SVM Baseline: {(best_acc_imp - acc_svm)*100:+.2f}%")
    
    if best_acc_imp > acc_orig:
        print(f"\n✓ Successfully improved median learner by {(best_acc_imp - acc_orig)*100:.2f}%!")
    
    print("\n" + "=" * 80)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    names = [r[0] for r in results]
    accs = [r[1] for r in results]
    colors = ['gold' if acc == best_acc_imp else 'green' for acc in accs]
    colors[-1] = 'purple'  # SVM color
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.75, 0.83)
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=10)
    plt.axhline(acc_orig, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Original')
    plt.grid(True, alpha=0.3, axis='y')
    plt.title('Model Accuracy Comparison', fontweight='bold', fontsize=13)
    plt.legend(fontsize=10)
    
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Improvements
    plt.subplot(1, 2, 2)
    improvements = [(acc - acc_orig) * 100 for acc in accs]
    colors_imp = ['gold' if imp == max(improvements[:-1]) else 'green' if imp > 0 else 'orange' for imp in improvements]
    colors_imp[-1] = 'purple'
    
    plt.barh(range(len(names)), improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Improvement vs Original (%)', fontsize=12)
    plt.yticks(range(len(names)), names)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.title('Performance Improvement', fontweight='bold', fontsize=13)
    
    for i, imp in enumerate(improvements):
        plt.text(imp + 0.15 if imp > 0 else imp - 0.15, i, f'{imp:+.2f}%',
                va='center', ha='left' if imp > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/final_improved_learner.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved: final_improved_learner.png")
    
    # Save results
    output = {
        'Original_Median': float(acc_orig),
        'Best_Improved': best_improved,
        'Best_Accuracy': float(best_acc_imp),
        'SVM_Baseline': float(acc_svm),
        'All_Results': {name: float(acc) for name, acc, _ in results},
        'Improvements': {
            'vs_original_percent': float((best_acc_imp - acc_orig) * 100),
            'vs_svm_percent': float((best_acc_imp - acc_svm) * 100)
        },
        'Strategies_Applied': [
            'Robust feature selection (Spearman correlation)',
            'Decision threshold optimization',
            'Class weighting for imbalanced data'
        ]
    }
    
    with open('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/final_improved_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("✓ Results saved: final_improved_results.json\n")


if __name__ == "__main__":
    main()
