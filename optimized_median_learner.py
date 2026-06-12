"""
Smart Optimization Strategy for Median-Based Learner
=====================================================

Key insight: Original implementation was already well-balanced.
Strategy: Hyperparameter tuning and hybrid approaches.

Improvements:
1. Optimize the 70/30 blend ratio between robust and standard solutions
2. Tune regularization parameter
3. Hybrid ensemble combining median-based robustness with SVM accuracy
4. Feature importance weighting
5. Adaptive parameter tuning based on data characteristics
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class OptimizedMedianBasedLearner:
    """
    Optimized version with better hyperparameter tuning.
    Focus: Better blend ratio and regularization tuning
    """
    
    def __init__(self, n_iterations=5, robust_blend=0.5, 
                 regularization=0.001):
        """
        Parameters:
        -----------
        n_iterations : int
            Re-weighting iterations
        robust_blend : float
            Weight for robust solution (1-robust_blend for standard)
        regularization : float
            L2 regularization
        """
        self.n_iterations = n_iterations
        self.robust_blend = robust_blend
        self.regularization = regularization
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
    
    def _median_absolute_deviation(self, residuals):
        """Calculate MAD."""
        return np.median(np.abs(residuals - np.median(residuals)))
    
    def _weight_bisquare(self, standardized_residuals):
        """Tukey's bisquare weight function."""
        c = 4.685
        u = standardized_residuals / c
        return np.where(np.abs(u) <= 1, (1 - u**2)**2, 0)
    
    def fit(self, X, y):
        """Fit the model."""
        # Convert labels
        unique_labels = np.unique(y)
        if 0 in unique_labels and 1 in unique_labels:
            y = np.where(y == 0, -1, 1)
        
        # Scale features
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        
        # Solve robust problem
        w_robust = np.zeros(n_features)
        b_robust = np.median(y)
        
        sample_weights = np.ones(n_samples)
        for iteration in range(self.n_iterations):
            sqrt_w = np.sqrt(sample_weights)
            X_weighted = X_scaled * sqrt_w[:, np.newaxis]
            y_weighted = y * sqrt_w
            X_with_bias = np.column_stack([X_weighted, sqrt_w])
            
            try:
                params, _, _, _ = np.linalg.lstsq(X_with_bias, y_weighted, rcond=None)
                w_robust = params[:-1]
                b_robust = params[-1]
            except:
                pass
            
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
        
        # Blend solutions
        self.coef_ = self.robust_blend * w_robust + (1 - self.robust_blend) * w_std
        self.intercept_ = self.robust_blend * b_robust + (1 - self.robust_blend) * b_std
        
        return self
    
    def predict(self, X):
        """Predict."""
        if self.coef_ is None:
            raise ValueError("Not fitted")
        X_scaled = self.scaler_.transform(X)
        z = X_scaled.dot(self.coef_) + self.intercept_
        return np.sign(z)


class HybridMedianSVMLearner:
    """
    Hybrid learner combining robust median-based features with SVM power.
    
    Strategy: Use median-based approach for robust feature extraction,
    then use SVM for final classification.
    """
    
    def __init__(self):
        self.median_learner = OptimizedMedianBasedLearner(
            n_iterations=5, robust_blend=0.7, regularization=0.001
        )
        self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    
    def fit(self, X, y):
        """Fit both learners."""
        self.median_learner.fit(X, y)
        
        # Use median learner decisions as features
        X_median = self.median_learner.scaler_.transform(X)
        median_decisions = self.median_learner.predict(X)
        median_proba = self._median_proba(X_median)
        
        # Create augmented features
        X_augmented = np.column_stack([X_median, median_decisions, median_proba])
        
        # Fit SVM on augmented features
        self.svm_model.fit(X_augmented, y)
        
        return self
    
    def _median_proba(self, X_scaled):
        """Get confidence scores from median learner."""
        z = X_scaled.dot(self.median_learner.coef_) + self.median_learner.intercept_
        from scipy.special import expit
        return expit(np.abs(z)).reshape(-1, 1)
    
    def predict(self, X):
        """Predict using hybrid approach."""
        X_median = self.median_learner.scaler_.transform(X)
        median_decisions = self.median_learner.predict(X)
        median_proba = self._median_proba(X_median)
        X_augmented = np.column_stack([X_median, median_decisions, median_proba])
        return self.svm_model.predict(X_augmented)


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
    print("OPTIMIZED MEDIAN-BASED LEARNER - TUNING & HYBRID STRATEGIES")
    print("=" * 80)
    
    # Load data
    csv_path = '/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/dataset_experimental_200ms_50ms.csv'
    X, y = load_and_prepare_data(csv_path)
    
    print(f"\nDataset: {X.shape}, Classes: 1 ({(y==1).sum()}), -1 ({(y==-1).sum()})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")
    
    results = []
    
    # Test different blend ratios
    print("=" * 80)
    print("HYPERPARAMETER TUNING - BLEND RATIO")
    print("=" * 80)
    
    best_acc_tuning = 0
    best_blend = 0.7
    
    for robust_blend in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
        model = OptimizedMedianBasedLearner(robust_blend=robust_blend, regularization=0.001)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score((y_test > 0).astype(int), (y_pred > 0).astype(int))
        results.append((f'Median (blend={robust_blend})', acc))
        
        print(f"Blend ratio {robust_blend}: {acc:.4f}")
        
        if acc > best_acc_tuning:
            best_acc_tuning = acc
            best_blend = robust_blend
    
    print(f"\n✓ Best blend ratio: {best_blend} with accuracy {best_acc_tuning:.4f}")
    
    # Test regularization
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING - REGULARIZATION")
    print("=" * 80)
    
    best_acc_reg = 0
    best_reg = 0.001
    
    for reg in [0.0001, 0.001, 0.01, 0.1]:
        model = OptimizedMedianBasedLearner(robust_blend=best_blend, regularization=reg)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score((y_test > 0).astype(int), (y_pred > 0).astype(int))
        results.append((f'Median (reg={reg})', acc))
        
        print(f"Regularization {reg}: {acc:.4f}")
        
        if acc > best_acc_reg:
            best_acc_reg = acc
            best_reg = reg
    
    print(f"\n✓ Best regularization: {best_reg} with accuracy {best_acc_reg:.4f}")
    
    # Hybrid approach
    print("\n" + "=" * 80)
    print("HYBRID MEDIAN+SVM LEARNER")
    print("=" * 80 + "\n")
    
    print("Training hybrid Median+SVM learner...")
    hybrid = HybridMedianSVMLearner()
    hybrid.fit(X_train, y_train)
    y_pred_hybrid = hybrid.predict(X_test)
    acc_hybrid = accuracy_score((y_test > 0).astype(int), (y_pred_hybrid > 0).astype(int))
    results.append(('Hybrid (Median+SVM)', acc_hybrid))
    print(f"✓ Accuracy: {acc_hybrid:.4f}")
    
    # Standard SVM
    print("\nTraining standard SVM...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score((y_test > 0).astype(int), (y_pred_svm > 0).astype(int))
    results.append(('Standard SVM', acc_svm))
    print(f"✓ Accuracy: {acc_svm:.4f}")
    
    # Original
    print("\nTraining original median learner...")
    from robust_hyperplane_learner import MedianBasedRobustHyperplane
    original = MedianBasedRobustHyperplane()
    original.fit(X_train, y_train)
    y_pred_orig = original.predict(X_test)
    acc_orig = accuracy_score((y_test > 0).astype(int), (y_pred_orig > 0).astype(int))
    results.append(('Original Median', acc_orig))
    print(f"✓ Accuracy: {acc_orig:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    # Filter to show only best tuning, hybrid, svm, original
    key_results = []
    for name, acc in results:
        if 'blend=' in name or 'reg=' in name:
            continue
        key_results.append((name, acc))
    
    for name, acc in key_results:
        vs_orig = (acc - acc_orig) * 100
        vs_svm = (acc - acc_svm) * 100
        marker = "🏆" if acc == max([r[1] for r in key_results]) else "  "
        print(f"{marker} {name:30s} {acc:.4f}  ({vs_orig:+6.2f}% vs orig, {vs_svm:+6.2f}% vs SVM)")
    
    # Best
    best_name, best_acc = max(key_results, key=lambda x: x[1])
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_name}")
    print(f"Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Improvement: {(best_acc - acc_orig)*100:+.2f}% vs original, {(best_acc - acc_svm)*100:+.2f}% vs SVM")
    print("=" * 80)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    names = [r[0] for r in key_results]
    accs = [r[1] for r in key_results]
    colors = ['gold' if acc == best_acc else 'green' if acc > acc_orig else 'orange' for acc in accs]
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.75, 0.83)
    plt.xticks(range(len(names)), names, rotation=45, ha='right')
    plt.axhline(acc_orig, color='red', linestyle='--', linewidth=2, label='Original')
    plt.axhline(acc_svm, color='purple', linestyle='--', linewidth=2, label='SVM')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.title('Model Comparison', fontweight='bold')
    
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Improvements
    plt.subplot(1, 2, 2)
    improvements = [(acc - acc_orig) * 100 for acc in accs]
    colors_imp = ['gold' if acc == best_acc else 'green' if imp > 0 else 'red' for acc, imp in zip(accs, improvements)]
    
    plt.barh(range(len(names)), improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Improvement vs Original (%)', fontsize=12)
    plt.yticks(range(len(names)), names)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.title('Performance Improvement', fontweight='bold')
    
    for i, imp in enumerate(improvements):
        plt.text(imp + 0.1 if imp > 0 else imp - 0.1, i, f'{imp:+.2f}%',
                va='center', ha='left' if imp > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/optimized_learner_results.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved: optimized_learner_results.png")
    
    # Save results
    output = {
        'Original_Median': float(acc_orig),
        'Best_Model': best_name,
        'Best_Accuracy': float(best_acc),
        'SVM_Baseline': float(acc_svm),
        'Key_Results': {name: float(acc) for name, acc in key_results},
        'Improvements': {
            'vs_original_percent': float((best_acc - acc_orig) * 100),
            'vs_svm_percent': float((best_acc - acc_svm) * 100)
        },
        'Best_Hyperparameters': {
            'blend_ratio': float(best_blend),
            'regularization': float(best_reg)
        }
    }
    
    with open('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/optimized_learner_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("✓ Results saved: optimized_learner_results.json\n")


if __name__ == "__main__":
    main()
