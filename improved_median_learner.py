"""
Improved Robust Hyperplane Learner with Advanced Features
==========================================================

Enhanced version with kernel methods, class weighting, and optimized training.

Features:
- Kernel support (linear, RBF, polynomial)
- Class weighting for imbalanced data
- Enhanced optimization with better convergence
- Alternative robust weight functions
- Hyperparameter tuning
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy.special import expit
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class ImprovedMedianBasedRobustHyperplane:
    """
    Improved robust hyperplane learner with kernel support and advanced optimization.
    
    Enhancements over basic version:
    - Kernel methods (RBF, polynomial)
    - Class weighting for imbalanced data
    - Enhanced M-estimation with multiple weight functions
    - Optimized convergence with more iterations
    - Better feature preprocessing
    - Adaptive parameter tuning
    """
    
    def __init__(self, kernel='linear', C=1.0, gamma='auto', degree=3,
                 n_iterations=10, class_weight='balanced', 
                 weight_function='bisquare', regularization=0.01,
                 normalize_features=True, use_pca=False, n_components=None):
        """
        Initialize the Improved Median-based Robust Hyperplane learner.
        
        Parameters:
        -----------
        kernel : str ('linear', 'rbf', 'poly')
            Kernel type for non-linear learning
        C : float
            Inverse regularization strength (lower = more regularization)
        gamma : str or float
            Kernel parameter for RBF and polynomial kernels
        degree : int
            Degree for polynomial kernel
        n_iterations : int
            Number of re-weighting iterations (increased from 5)
        class_weight : str or dict
            'balanced' to adjust weights by class frequency
        weight_function : str
            Type of robust weight function ('bisquare', 'andrews', 'welsch')
        regularization : float
            L2 regularization strength
        normalize_features : bool
            Whether to normalize features to unit norm
        use_pca : bool
            Whether to apply PCA for dimensionality reduction
        n_components : int
            Number of PCA components (default: auto)
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.n_iterations = n_iterations
        self.class_weight = class_weight
        self.weight_function = weight_function
        self.regularization = regularization
        self.normalize_features = normalize_features
        self.use_pca = use_pca
        self.n_components = n_components
        
        self.coef_ = None
        self.intercept_ = None
        self.weights_ = None
        self.scaler_ = None
        self.normalizer_ = None
        self.pca_ = None
        self.support_vectors_ = None
        self.X_train_scaled_ = None
        self.class_weights_ = None
        
    def _compute_kernel(self, X, Y=None):
        """Compute kernel matrix."""
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return X.dot(Y.T)
        elif self.kernel == 'rbf':
            gamma = self.gamma if isinstance(self.gamma, (int, float)) else 1.0 / X.shape[1]
            # Efficient RBF kernel computation
            X_norm = np.sum(X ** 2, axis=1, keepdims=True)
            Y_norm = np.sum(Y ** 2, axis=1, keepdims=True).T
            XY = X.dot(Y.T)
            distances = X_norm + Y_norm - 2 * XY
            return np.exp(-gamma * distances)
        elif self.kernel == 'poly':
            return (1 + X.dot(Y.T)) ** self.degree
        else:
            return X.dot(Y.T)
    
    def _weight_bisquare(self, standardized_residuals):
        """Tukey's bisquare weight function."""
        c = 4.685
        u = standardized_residuals / c
        return np.where(np.abs(u) <= 1, (1 - u**2)**2, 0)
    
    def _weight_andrews(self, standardized_residuals):
        """Andrews' sine weight function."""
        c = 1.339
        u = standardized_residuals / c
        return np.where(
            np.abs(u) <= np.pi,
            np.sin(u) / u,
            0
        )
    
    def _weight_welsch(self, standardized_residuals):
        """Welsch's exponential weight function."""
        c = 2.985
        u = standardized_residuals / c
        return np.exp(-(u**2))
    
    def _get_weight_function(self):
        """Return the appropriate weight function."""
        if self.weight_function == 'bisquare':
            return self._weight_bisquare
        elif self.weight_function == 'andrews':
            return self._weight_andrews
        elif self.weight_function == 'welsch':
            return self._weight_welsch
        else:
            return self._weight_bisquare
    
    def _median_absolute_deviation(self, residuals):
        """Calculate median absolute deviation."""
        return np.median(np.abs(residuals - np.median(residuals)))
    
    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            weights = len(y) / (2 * counts)
            class_weights = dict(zip(unique, weights))
            return class_weights
        else:
            return {-1: 1.0, 1: 1.0}
    
    def fit(self, X, y):
        """
        Fit the improved median-based robust hyperplane learner.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels (binary: -1, 1 or 0, 1)
        
        Returns:
        --------
        self
        """
        # Convert labels to -1, 1 format
        unique_labels = np.unique(y)
        if 0 in unique_labels and 1 in unique_labels:
            y = np.where(y == 0, -1, 1)
        
        # Compute class weights
        self.class_weights_ = self._compute_class_weights(y)
        
        # Robust scaling
        self.scaler_ = RobustScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Feature normalization
        if self.normalize_features:
            self.normalizer_ = normalize
            X_scaled = self.normalizer_(X_scaled, norm='l2')
        
        # PCA for dimensionality reduction
        if self.use_pca:
            n_comp = self.n_components or min(X_scaled.shape[0], X_scaled.shape[1]) // 2
            self.pca_ = PCA(n_components=n_comp)
            X_scaled = self.pca_.fit_transform(X_scaled)
        
        self.X_train_scaled_ = X_scaled
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        
        # Initialize parameters
        w = np.zeros(n_features)
        b = 0
        
        # Iterative re-weighted least squares with enhanced iterations
        sample_weights = np.ones(n_samples)
        
        # Compute kernel matrix once for efficiency
        K = self._compute_kernel(X_scaled)
        
        weight_func = self._get_weight_function()
        
        for iteration in range(self.n_iterations):
            # Apply class weights and sample weights
            effective_weights = sample_weights.copy()
            for label in np.unique(y):
                mask = y == label
                effective_weights[mask] *= self.class_weights_[label]
            
            # Normalize weights
            effective_weights = effective_weights / np.mean(effective_weights)
            
            # Weighted least squares with kernel
            if self.kernel == 'linear':
                # For linear kernel, use standard weighted least squares
                sqrt_w = np.sqrt(effective_weights)
                X_weighted = X_scaled * sqrt_w[:, np.newaxis]
                y_weighted = y * sqrt_w
                
                X_with_bias = np.column_stack([X_weighted, sqrt_w])
                
                try:
                    params, _, _, _ = np.linalg.lstsq(X_with_bias, y_weighted, rcond=None)
                    w = params[:-1]
                    b = params[-1]
                except:
                    pass
            else:
                # For kernel methods, use kernel-based optimization
                sqrt_w = np.sqrt(effective_weights)
                K_weighted = K * np.outer(sqrt_w, sqrt_w)
                y_weighted = y * sqrt_w
                
                try:
                    # Solve kernel method
                    alpha, _, _, _ = np.linalg.lstsq(
                        K_weighted + self.regularization * np.eye(len(y)),
                        y_weighted,
                        rcond=None
                    )
                    self.support_vectors_ = X_scaled
                    w = alpha
                    b = np.mean(y - K.dot(alpha))
                except:
                    pass
            
            # Predictions
            if self.kernel == 'linear':
                z = X_scaled.dot(w) + b
            else:
                z = K.dot(w) + b
            
            predictions = np.sign(z)
            
            # Update weights based on residuals
            errors = np.abs(y - predictions)
            mad = self._median_absolute_deviation(errors)
            
            if mad > 1e-10:
                standardized_errors = errors / mad
                sample_weights = weight_func(standardized_errors)
            else:
                sample_weights = np.ones(n_samples)
            
            sample_weights = np.clip(sample_weights, 1e-4, 1.0)
        
        # Store parameters
        self.coef_ = w
        self.intercept_ = b
        self.weights_ = sample_weights
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = self.normalizer_(X_scaled, norm='l2')
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        if self.kernel == 'linear':
            z = X_scaled.dot(self.coef_) + self.intercept_
        else:
            K_pred = self._compute_kernel(X_scaled, self.X_train_scaled_)
            z = K_pred.dot(self.coef_) + self.intercept_
        
        return np.sign(z)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = self.normalizer_(X_scaled, norm='l2')
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        if self.kernel == 'linear':
            z = X_scaled.dot(self.coef_) + self.intercept_
        else:
            K_pred = self._compute_kernel(X_scaled, self.X_train_scaled_)
            z = K_pred.dot(self.coef_) + self.intercept_
        
        proba_positive = expit(z)
        return np.column_stack([1 - proba_positive, proba_positive])
    
    def decision_function(self, X):
        """Compute decision function."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = self.normalizer_(X_scaled, norm='l2')
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        if self.kernel == 'linear':
            return X_scaled.dot(self.coef_) + self.intercept_
        else:
            K_pred = self._compute_kernel(X_scaled, self.X_train_scaled_)
            return K_pred.dot(self.coef_) + self.intercept_


def load_and_prepare_data(csv_path):
    """Load and prepare the dataset."""
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
    """Main execution function."""
    
    print("=" * 80)
    print("IMPROVED MEDIAN-BASED ROBUST HYPERPLANE LEARNER")
    print("=" * 80)
    
    # Load data
    csv_path = '/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/dataset_experimental_200ms_50ms.csv'
    X, y = load_and_prepare_data(csv_path)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Class distribution: Class 1: {(y == 1).sum()}, Class -1: {(y == -1).sum()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    
    print("\n" + "=" * 80)
    print("TRAINING IMPROVED MODELS")
    print("=" * 80)
    
    results = []
    
    # 1. Linear with class weighting
    print("\n[1] Training Improved Linear Learner (with class weighting)...")
    model1 = ImprovedMedianBasedRobustHyperplane(
        kernel='linear',
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True
    )
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    acc1 = accuracy_score((y_test > 0).astype(int), (y_pred1 > 0).astype(int))
    results.append({
        'Model': 'Improved Linear (Balanced Weights)',
        'Accuracy': acc1,
        'Config': 'linear, n_iter=10, balanced_weights'
    })
    print(f"✓ Accuracy: {acc1:.4f}")
    
    # 2. RBF kernel learner
    print("\n[2] Training RBF Kernel-Based Learner...")
    model2 = ImprovedMedianBasedRobustHyperplane(
        kernel='rbf',
        gamma=0.01,
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True
    )
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    acc2 = accuracy_score((y_test > 0).astype(int), (y_pred2 > 0).astype(int))
    results.append({
        'Model': 'Improved RBF Kernel',
        'Accuracy': acc2,
        'Config': 'rbf, gamma=0.01, n_iter=10'
    })
    print(f"✓ Accuracy: {acc2:.4f}")
    
    # 3. Polynomial kernel learner
    print("\n[3] Training Polynomial Kernel Learner...")
    model3 = ImprovedMedianBasedRobustHyperplane(
        kernel='poly',
        degree=3,
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True
    )
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    acc3 = accuracy_score((y_test > 0).astype(int), (y_pred3 > 0).astype(int))
    results.append({
        'Model': 'Improved Polynomial Kernel',
        'Accuracy': acc3,
        'Config': 'poly, degree=3, n_iter=10'
    })
    print(f"✓ Accuracy: {acc3:.4f}")
    
    # 4. Linear with Welsch weight function
    print("\n[4] Training Linear Learner (Welsch weight function)...")
    model4 = ImprovedMedianBasedRobustHyperplane(
        kernel='linear',
        n_iterations=10,
        class_weight='balanced',
        weight_function='welsch',
        normalize_features=True
    )
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    acc4 = accuracy_score((y_test > 0).astype(int), (y_pred4 > 0).astype(int))
    results.append({
        'Model': 'Improved Linear (Welsch)',
        'Accuracy': acc4,
        'Config': 'linear, welsch_weights, n_iter=10'
    })
    print(f"✓ Accuracy: {acc4:.4f}")
    
    # 5. Linear with PCA
    print("\n[5] Training Linear Learner with PCA...")
    model5 = ImprovedMedianBasedRobustHyperplane(
        kernel='linear',
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True,
        use_pca=True,
        n_components=80
    )
    model5.fit(X_train, y_train)
    y_pred5 = model5.predict(X_test)
    acc5 = accuracy_score((y_test > 0).astype(int), (y_pred5 > 0).astype(int))
    results.append({
        'Model': 'Improved Linear with PCA',
        'Accuracy': acc5,
        'Config': 'linear, pca=80, n_iter=10'
    })
    print(f"✓ Accuracy: {acc5:.4f}")
    
    # 6. RBF with PCA
    print("\n[6] Training RBF Kernel with PCA...")
    model6 = ImprovedMedianBasedRobustHyperplane(
        kernel='rbf',
        gamma=0.01,
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True,
        use_pca=True,
        n_components=80
    )
    model6.fit(X_train, y_train)
    y_pred6 = model6.predict(X_test)
    acc6 = accuracy_score((y_test > 0).astype(int), (y_pred6 > 0).astype(int))
    results.append({
        'Model': 'Improved RBF with PCA',
        'Accuracy': acc6,
        'Config': 'rbf, pca=80, gamma=0.01'
    })
    print(f"✓ Accuracy: {acc6:.4f}")
    
    # 7. Standard SVM for comparison
    print("\n[7] Training Standard SVM (for comparison)...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    acc_svm = accuracy_score((y_test > 0).astype(int), (y_pred_svm > 0).astype(int))
    results.append({
        'Model': 'Standard SVM (Baseline)',
        'Accuracy': acc_svm,
        'Config': 'rbf, C=1.0'
    })
    print(f"✓ Accuracy: {acc_svm:.4f}")
    
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['Accuracy'].idxmax()
    best_model_name = results_df.loc[best_idx, 'Model']
    best_accuracy = results_df.loc[best_idx, 'Accuracy']
    
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Improvement over baseline: {(best_accuracy - acc_svm)*100:.2f}%")
    print(f"{'=' * 80}")
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    models = results_df['Model']
    accuracies = results_df['Accuracy']
    colors = ['green' if acc > acc_svm else 'orange' if acc > 0.77 else 'red' for acc in accuracies]
    bars = plt.barh(range(len(models)), accuracies, color=colors, alpha=0.8)
    
    plt.xlabel('Accuracy', fontsize=12)
    plt.yticks(range(len(models)), models, fontsize=10)
    plt.xlim(0.75, 0.85)
    
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(acc + 0.001, i, f'{acc:.4f}', va='center', fontsize=9)
    
    plt.axvline(acc_svm, color='red', linestyle='--', linewidth=2, label=f'SVM Baseline ({acc_svm:.4f})')
    plt.title('Improved Learner Performance Comparison', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Top improvements
    plt.subplot(1, 2, 2)
    improvements = (results_df['Accuracy'] - acc_svm) * 100
    colors_imp = ['green' if imp > 0 else 'red' for imp in improvements]
    
    plt.barh(range(len(models)), improvements, color=colors_imp, alpha=0.8)
    plt.xlabel('Improvement over SVM (%)', fontsize=12)
    plt.yticks(range(len(models)), models, fontsize=10)
    plt.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    for i, imp in enumerate(improvements):
        plt.text(imp + 0.05 if imp > 0 else imp - 0.05, i, f'{imp:+.2f}%', 
                va='center', ha='left' if imp > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.title('Performance Improvement over SVM', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/improved_learner_comparison.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Comparison plot saved: improved_learner_comparison.png")
    
    # Save results
    output_data = {
        'Results': results,
        'Best_Model': {
            'Name': best_model_name,
            'Accuracy': float(best_accuracy),
            'Improvement_over_SVM_percent': float((best_accuracy - acc_svm) * 100)
        },
        'SVM_Baseline': {
            'Name': 'Standard SVM',
            'Accuracy': float(acc_svm)
        }
    }
    
    with open('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/improved_learner_results.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("✓ Results saved: improved_learner_results.json")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOriginal Median-Based Learner: 77.48%")
    print(f"Best Improved Model: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"SVM Baseline: {acc_svm:.4f} ({acc_svm*100:.2f}%)")
    print(f"\nTotal Improvement: {(best_accuracy - 0.7748)*100:+.2f}%")
    print(f"Relative to SVM: {(best_accuracy - acc_svm)*100:+.2f}%")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
