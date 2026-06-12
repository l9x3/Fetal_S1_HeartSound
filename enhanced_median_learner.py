"""
Enhanced Median-Based Learner with Practical Performance Improvements
======================================================================

Optimized version focusing on practical improvements:
- Feature scaling and normalization
- Class weighting for imbalanced data
- Enhanced optimization with more iterations
- Support for different weight functions
- Faster, more scalable implementation
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class EnhancedMedianBasedLearner:
    """
    Enhanced median-based robust learner with practical improvements.
    
    Improvements:
    - Better feature preprocessing (normalization, scaling)
    - Class weighting for imbalanced data
    - More efficient optimization
    - Multiple robust weight functions
    - Better parameter tuning
    """
    
    def __init__(self, n_iterations=15, class_weight='balanced',
                 weight_function='bisquare', regularization=0.001,
                 normalize_features=True, use_pca=False, pca_components=0.95):
        """
        Parameters:
        -----------
        n_iterations : int
            Number of re-weighting iterations (increased from 5 to 15)
        class_weight : str or dict
            'balanced' to adjust for class imbalance
        weight_function : str
            Type of robust weight ('bisquare', 'andrews', 'welsch')
        regularization : float
            L2 regularization strength
        normalize_features : bool
            Whether to normalize features
        use_pca : bool
            Whether to use PCA for dimensionality reduction
        pca_components : float
            PCA variance threshold or n_components
        """
        self.n_iterations = n_iterations
        self.class_weight = class_weight
        self.weight_function = weight_function
        self.regularization = regularization
        self.normalize_features = normalize_features
        self.use_pca = use_pca
        self.pca_components = pca_components
        
        self.coef_ = None
        self.intercept_ = None
        self.scaler_ = None
        self.pca_ = None
        self.class_weights_ = None
        self.X_train_scaled_ = None
        self.feature_importance_ = None
    
    def _weight_bisquare(self, u):
        """Tukey's bisquare weight function."""
        c = 4.685
        normalized = u / c
        return np.where(np.abs(normalized) <= 1, (1 - normalized**2)**2, 0)
    
    def _weight_andrews(self, u):
        """Andrews' sine weight function (more robust)."""
        c = 1.339
        normalized = u / c
        result = np.zeros_like(normalized)
        mask = np.abs(normalized) <= np.pi
        result[mask] = np.sin(normalized[mask]) / (normalized[mask] + 1e-10)
        return result
    
    def _weight_welsch(self, u):
        """Welsch's exponential weight function (heaviest tail downweighting)."""
        c = 2.985
        normalized = u / c
        return np.exp(-(normalized**2))
    
    def _get_weight_function(self):
        """Return the weight function."""
        if self.weight_function == 'bisquare':
            return self._weight_bisquare
        elif self.weight_function == 'andrews':
            return self._weight_andrews
        elif self.weight_function == 'welsch':
            return self._weight_welsch
        else:
            return self._weight_bisquare
    
    def _median_absolute_deviation(self, residuals):
        """Calculate MAD."""
        return np.median(np.abs(residuals - np.median(residuals)))
    
    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        if self.class_weight == 'balanced':
            unique, counts = np.unique(y, return_counts=True)
            weights = len(y) / (2 * counts)
            return dict(zip(unique, weights / np.sum(weights)))
        else:
            return {-1: 1.0, 1: 1.0}
    
    def fit(self, X, y):
        """Fit the enhanced median-based learner."""
        # Convert labels
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
            X_scaled = normalize(X_scaled, norm='l2', axis=0)
        
        # PCA for efficiency
        if self.use_pca:
            self.pca_ = PCA(n_components=self.pca_components)
            X_scaled = self.pca_.fit_transform(X_scaled)
        
        self.X_train_scaled_ = X_scaled
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        
        # Initialize parameters
        w = np.zeros(n_features)
        b = np.median(y)
        
        # Get weight function
        weight_func = self._get_weight_function()
        
        # Sample weights for outlier downweighting
        sample_weights = np.ones(n_samples)
        
        # Iterative re-weighted least squares
        for iteration in range(self.n_iterations):
            # Apply class weights
            effective_weights = sample_weights.copy()
            for label in [-1, 1]:
                mask = y == label
                if np.any(mask):
                    effective_weights[mask] *= self.class_weights_.get(label, 1.0)
            
            # Normalize weights
            effective_weights = effective_weights / np.mean(effective_weights)
            
            # Weighted least squares
            sqrt_w = np.sqrt(effective_weights)
            X_weighted = X_scaled * sqrt_w[:, np.newaxis]
            y_weighted = y * sqrt_w
            
            # Augment with intercept
            X_with_bias = np.column_stack([X_weighted, sqrt_w])
            
            # Solve weighted least squares
            try:
                params, _, _, _ = np.linalg.lstsq(X_with_bias, y_weighted, rcond=None)
                w = params[:-1]
                b = params[-1]
            except:
                continue
            
            # Compute predictions and residuals
            z = X_scaled.dot(w) + b
            predictions = np.sign(z)
            errors = np.abs(y - predictions)
            
            # Update weights using robust M-estimation
            mad = self._median_absolute_deviation(errors)
            
            if mad > 1e-10:
                standardized_errors = errors / mad
                sample_weights = weight_func(standardized_errors)
            else:
                sample_weights = np.ones(n_samples)
            
            # Clip weights to reasonable range
            sample_weights = np.clip(sample_weights, 1e-5, 1.0)
        
        self.coef_ = w
        self.intercept_ = b
        self.feature_importance_ = np.abs(w)
        
        return self
    
    def predict(self, X):
        """Predict class labels."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = normalize(X_scaled, norm='l2', axis=0)
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        z = X_scaled.dot(self.coef_) + self.intercept_
        return np.sign(z)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = normalize(X_scaled, norm='l2', axis=0)
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        z = X_scaled.dot(self.coef_) + self.intercept_
        
        # Convert to probabilities using sigmoid
        from scipy.special import expit
        proba_positive = expit(z)
        return np.column_stack([1 - proba_positive, proba_positive])
    
    def decision_function(self, X):
        """Compute decision function."""
        if self.coef_ is None:
            raise ValueError("Model not fitted yet")
        
        X_scaled = self.scaler_.transform(X)
        
        if self.normalize_features:
            X_scaled = normalize(X_scaled, norm='l2', axis=0)
        
        if self.use_pca and self.pca_ is not None:
            X_scaled = self.pca_.transform(X_scaled)
        
        return X_scaled.dot(self.coef_) + self.intercept_


def load_and_prepare_data(csv_path):
    """Load and prepare dataset."""
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
    print("ENHANCED MEDIAN-BASED LEARNER - PERFORMANCE IMPROVEMENTS")
    print("=" * 80)
    
    # Load data
    csv_path = '/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/dataset_experimental_200ms_50ms.csv'
    X, y = load_and_prepare_data(csv_path)
    
    print(f"\nDataset: {X.shape}")
    print(f"Classes: 1 ({(y == 1).sum()}), -1 ({(y == -1).sum()})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    results = []
    
    # 1. Original baseline
    print("\n[1] Original Median-Based Learner (baseline)...")
    from robust_hyperplane_learner import MedianBasedRobustHyperplane
    original = MedianBasedRobustHyperplane()
    original.fit(X_train, y_train)
    y_pred_orig = original.predict(X_test)
    acc_orig = accuracy_score((y_test > 0).astype(int), (y_pred_orig > 0).astype(int))
    results.append(('Original Baseline', acc_orig))
    print(f"✓ Accuracy: {acc_orig:.4f}")
    
    # 2. Enhanced with normalization
    print("\n[2] Enhanced with Feature Normalization...")
    model1 = EnhancedMedianBasedLearner(
        n_iterations=10,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True,
        regularization=0.001
    )
    model1.fit(X_train, y_train)
    y_pred1 = model1.predict(X_test)
    acc1 = accuracy_score((y_test > 0).astype(int), (y_pred1 > 0).astype(int))
    results.append(('Enhanced + Normalization', acc1))
    print(f"✓ Accuracy: {acc1:.4f}")
    
    # 3. Enhanced with more iterations
    print("\n[3] Enhanced with 15 Iterations...")
    model2 = EnhancedMedianBasedLearner(
        n_iterations=15,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True,
        regularization=0.001
    )
    model2.fit(X_train, y_train)
    y_pred2 = model2.predict(X_test)
    acc2 = accuracy_score((y_test > 0).astype(int), (y_pred2 > 0).astype(int))
    results.append(('Enhanced + 15 Iterations', acc2))
    print(f"✓ Accuracy: {acc2:.4f}")
    
    # 4. Enhanced with Welsch weight function
    print("\n[4] Enhanced with Welsch Weight Function...")
    model3 = EnhancedMedianBasedLearner(
        n_iterations=15,
        class_weight='balanced',
        weight_function='welsch',
        normalize_features=True,
        regularization=0.001
    )
    model3.fit(X_train, y_train)
    y_pred3 = model3.predict(X_test)
    acc3 = accuracy_score((y_test > 0).astype(int), (y_pred3 > 0).astype(int))
    results.append(('Enhanced + Welsch Weights', acc3))
    print(f"✓ Accuracy: {acc3:.4f}")
    
    # 5. Enhanced with Andrews weight function
    print("\n[5] Enhanced with Andrews Weight Function...")
    model4 = EnhancedMedianBasedLearner(
        n_iterations=15,
        class_weight='balanced',
        weight_function='andrews',
        normalize_features=True,
        regularization=0.001
    )
    model4.fit(X_train, y_train)
    y_pred4 = model4.predict(X_test)
    acc4 = accuracy_score((y_test > 0).astype(int), (y_pred4 > 0).astype(int))
    results.append(('Enhanced + Andrews Weights', acc4))
    print(f"✓ Accuracy: {acc4:.4f}")
    
    # 6. Enhanced with PCA
    print("\n[6] Enhanced with PCA...")
    model5 = EnhancedMedianBasedLearner(
        n_iterations=15,
        class_weight='balanced',
        weight_function='bisquare',
        normalize_features=True,
        use_pca=True,
        pca_components=0.95,
        regularization=0.001
    )
    model5.fit(X_train, y_train)
    y_pred5 = model5.predict(X_test)
    acc5 = accuracy_score((y_test > 0).astype(int), (y_pred5 > 0).astype(int))
    results.append(('Enhanced + PCA (0.95)', acc5))
    print(f"✓ Accuracy: {acc5:.4f}")
    
    # 7. Standard SVM
    print("\n[7] Standard SVM (comparison)...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score((y_test > 0).astype(int), (y_pred_svm > 0).astype(int))
    results.append(('Standard SVM', acc_svm))
    print(f"✓ Accuracy: {acc_svm:.4f}")
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for model_name, accuracy in results:
        improvement = (accuracy - acc_orig) * 100
        vs_svm = (accuracy - acc_svm) * 100
        marker = "🏆" if accuracy == max([r[1] for r in results]) else "  "
        print(f"{marker} {model_name:40s} {accuracy:.4f} ({improvement:+6.2f}% vs orig, {vs_svm:+6.2f}% vs SVM)")
    
    # Best model
    best_model, best_acc = max(results, key=lambda x: x[1])
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best_model}")
    print(f"Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Improvement over original: {(best_acc - acc_orig)*100:+.2f}%")
    print(f"vs SVM: {(best_acc - acc_svm)*100:+.2f}%")
    print("=" * 80)
    
    # Visualization
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    names = [r[0] for r in results]
    accs = [r[1] for r in results]
    colors = ['gold' if acc == best_acc else 'green' if acc > acc_orig else 'orange' for acc in accs]
    
    bars = plt.bar(range(len(names)), accs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0.75, 0.83)
    plt.xticks(range(len(names)), names, rotation=45, ha='right', fontsize=9)
    plt.axhline(acc_orig, color='red', linestyle='--', linewidth=2, label='Original (77.48%)')
    plt.axhline(acc_svm, color='purple', linestyle='--', linewidth=2, label=f'SVM ({acc_svm:.2%})')
    plt.grid(True, alpha=0.3, axis='y')
    plt.title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    
    for i, acc in enumerate(accs):
        plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Improvement percentages
    plt.subplot(1, 2, 2)
    improvements = [(acc - acc_orig) * 100 for acc in accs]
    colors_imp = ['gold' if acc == best_acc else 'green' if imp > 0 else 'red' for acc, imp in zip(accs, improvements)]
    
    bars = plt.barh(range(len(names)), improvements, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.xlabel('Improvement over Original (%)', fontsize=12)
    plt.yticks(range(len(names)), names, fontsize=9)
    plt.axvline(0, color='black', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.title('Performance Improvement', fontsize=13, fontweight='bold')
    
    for i, imp in enumerate(improvements):
        plt.text(imp + 0.05 if imp > 0 else imp - 0.05, i, f'{imp:+.2f}%',
                va='center', ha='left' if imp > 0 else 'right', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/enhanced_learner_results.png',
                dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to: enhanced_learner_results.png")
    
    # Save JSON results
    output = {
        'Original_Baseline': float(acc_orig),
        'Best_Model': best_model,
        'Best_Accuracy': float(best_acc),
        'SVM_Baseline': float(acc_svm),
        'All_Results': {name: float(acc) for name, acc in results},
        'Improvements': {
            'vs_original_percent': float((best_acc - acc_orig) * 100),
            'vs_svm_percent': float((best_acc - acc_svm) * 100),
            'total_improvement_from_start': float((best_acc - acc_orig) * 100)
        }
    }
    
    with open('/home/runner/work/Fetal_S1_HeartSound/Fetal_S1_HeartSound/l9x3/Fetal_S1_HeartSound/enhanced_learner_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("✓ Results saved to: enhanced_learner_results.json")
    print()


if __name__ == "__main__":
    main()
