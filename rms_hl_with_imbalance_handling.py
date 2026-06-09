"""
RMS-HL: Robust Median-Based Supervised Hyperplane Learner
Fast Demonstration with Class Imbalance Handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, NuSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix
)
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import time
warnings.filterwarnings('ignore')

print("="*80)
print("RMS-HL: Robust Median-Based Supervised Hyperplane Learner")
print("Fast Demonstration with Class Imbalance Handling")
print("="*80)

# Kernels
def kernel_gaussian(t):
    return np.exp(-t**2 / 2)

KERNELS = {'gaussian': kernel_gaussian}

# Huber Loss
def huber_loss(z, delta=1.0):
    z = np.atleast_1d(z)
    loss = np.where(np.abs(z) <= delta, z**2/2, delta*(np.abs(z) - delta/2))
    return loss if len(loss) > 1 else loss[0]

def huber_deriv(z, delta=1.0):
    z = np.atleast_1d(z)
    deriv = np.where(np.abs(z) <= delta, z, delta*np.sign(z))
    return deriv if len(deriv) > 1 else deriv[0]

# Geometric Median
def geometric_median(X, max_iter=20):
    m = np.mean(X, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - m, axis=1)
        d[d < 1e-10] = 1e-10
        m_new = np.sum(X / d[:, None], axis=0) / np.sum(1.0/d)
        if np.linalg.norm(m_new - m) < 1e-5:
            break
        m = m_new
    return m

# RMS-HL Classifier with Class Weights
class RMSHL(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=0.1, delta=1.0, max_iter=20, random_state=None, class_weight='balanced'):
        self.lam = lam
        self.delta = delta
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.w = None
        self.b = None
        self.scaler = None
        self.class_weights = None
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Handle class labels
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError(f"Expected 2 classes, got {len(unique_labels)}")
        
        # Map to -1, 1
        label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        y_mapped = np.array([label_map[label] for label in y])
        
        # Compute class weights if balanced
        if self.class_weight == 'balanced':
            n_samples = len(y_mapped)
            n_classes = 2
            class_counts = np.bincount(np.where(y_mapped == -1, 0, 1))
            self.class_weights = n_samples / (n_classes * class_counts)
            self.class_weights[0] = self.class_weights[0] * (class_counts[0] / class_counts[1])
        else:
            self.class_weights = np.ones(2)
        
        n = X.shape[1]
        p = np.zeros(n + 1)
        
        for i in range(self.max_iter):
            w = p[:n]
            # Compute medians for each class
            for label_idx, label in enumerate([-1, 1]):
                mask = y_mapped == label
                if np.any(mask):
                    med = geometric_median(X[mask])
            
            # Optimize with weighted loss
            def objective(p):
                w = p[:n]
                b = p[n]
                margin = 1 - y_mapped * (X @ w + b)
                loss = huber_loss(margin, self.delta)
                
                # Apply class weights
                weighted_loss = np.zeros_like(loss)
                for idx, label in enumerate(y_mapped):
                    class_idx = 0 if label == -1 else 1
                    weighted_loss[idx] = loss[idx] * self.class_weights[class_idx]
                
                return self.lam * np.sum(w**2) + np.sum(weighted_loss)
            
            res = minimize(
                objective,
                p, method='L-BFGS-B', options={'maxiter': 15}
            )
            p = res.x
        
        self.w = p[:n]
        self.b = p[n]
        return self
    
    def predict(self, X):
        X = self.scaler.transform(X)
        return np.where(X @ self.w + self.b >= 0, 1, -1)
    
    def predict_proba(self, X):
        X = self.scaler.transform(X)
        s = 1.0 / (1.0 + np.exp(-(X @ self.w + self.b)))
        return np.column_stack([1-s, s])

print("\nLoading dataset...")
df = pd.read_csv('dataset_contaminated.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

print(f"Samples: {len(X)}, Features: {X.shape[1]}")
class_counts = np.bincount(np.where(y == 1, 1, 0))
print(f"Classes: {np.unique(y)} with counts {np.bincount(np.where(y == 1, 1, 0))}")
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

print("\n" + "="*80)
print("CLASS IMBALANCE HANDLING STRATEGIES")
print("="*80)
print("\n1. SMOTE (Synthetic Minority Over-sampling)")
print("   - Generates synthetic samples for minority class")
print("   - Applied during training only")
print("\n2. Class Weights")
print("   - Inverse frequency weighting")
print("   - Applied directly to loss function")
print("\n3. Evaluation Metrics")
print("   - F1-Score for balanced evaluation")
print("   - ROC-AUC for probability threshold independence")
print("   - Balanced Accuracy for class-independent measure")

print("\nRunning 2-Fold Cross-Validation...")
print("-" * 80)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
results = {
    'RMS-HL': [],
    'RMS-HL+SMOTE': [],
    'SVM': [],
    'SVM+SMOTE': [],
    'SVM (class_weight)': [],
    'Nu-SVM': [],
}

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/2 ...", end=" ", flush=True)
    t0 = time.time()
    
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    y_te_b = (y_te == 1).astype(int)
    
    models = {}
    
    # RMS-HL (with class weights)
    try:
        m = RMSHL(lam=0.1, delta=1.0, max_iter=15, random_state=42, class_weight='balanced')
        m.fit(X_tr, y_tr)
        pred_rmshl = m.predict(X_te)
        pred_rmshl_b = (pred_rmshl == 1).astype(int)
        proba_rmshl = m.predict_proba(X_te)[:, 1]
        models['RMS-HL'] = (pred_rmshl_b, proba_rmshl)
    except Exception as e:
        print(f"RMS-HL Error: {e}")
        models['RMS-HL'] = None
    
    # RMS-HL + SMOTE
    try:
        X_tr_smote = X_tr.copy()
        y_tr_smote = y_tr.copy()
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        try:
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_smote, y_tr_smote)
        except:
            pass  # Fallback if SMOTE fails due to few samples
        
        m = RMSHL(lam=0.1, delta=1.0, max_iter=15, random_state=42, class_weight='balanced')
        m.fit(X_tr_smote, y_tr_smote)
        pred_rmshl_smote = m.predict(X_te)
        pred_rmshl_smote_b = (pred_rmshl_smote == 1).astype(int)
        proba_rmshl_smote = m.predict_proba(X_te)[:, 1]
        models['RMS-HL+SMOTE'] = (pred_rmshl_smote_b, proba_rmshl_smote)
    except Exception as e:
        print(f"RMS-HL+SMOTE Error: {e}")
        models['RMS-HL+SMOTE'] = None
    
    # SVM
    try:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr, y_tr)
        pred_svm = (m.predict(X_te) == 1).astype(int)
        proba_svm = m.predict_proba(X_te)[:, 1]
        models['SVM'] = (pred_svm, proba_svm)
    except Exception as e:
        print(f"SVM Error: {e}")
        models['SVM'] = None
    
    # SVM + SMOTE
    try:
        X_tr_smote = X_tr.copy()
        y_tr_smote = y_tr.copy()
        
        smote = SMOTE(random_state=42, k_neighbors=5)
        try:
            X_tr_smote, y_tr_smote = smote.fit_resample(X_tr_smote, y_tr_smote)
        except:
            pass
        
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr_smote, y_tr_smote)
        pred_svm_smote = (m.predict(X_te) == 1).astype(int)
        proba_svm_smote = m.predict_proba(X_te)[:, 1]
        models['SVM+SMOTE'] = (pred_svm_smote, proba_svm_smote)
    except Exception as e:
        print(f"SVM+SMOTE Error: {e}")
        models['SVM+SMOTE'] = None
    
    # SVM with class weights
    try:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, 
                probability=True, class_weight='balanced')
        m.fit(X_tr, y_tr)
        pred_svm_cw = (m.predict(X_te) == 1).astype(int)
        proba_svm_cw = m.predict_proba(X_te)[:, 1]
        models['SVM (class_weight)'] = (pred_svm_cw, proba_svm_cw)
    except Exception as e:
        print(f"SVM (class_weight) Error: {e}")
        models['SVM (class_weight)'] = None
    
    # Nu-SVM
    try:
        m = NuSVC(kernel='rbf', nu=0.3, gamma='scale', random_state=42, 
                  probability=True, class_weight='balanced')
        m.fit(X_tr, y_tr)
        pred_nusvm = (m.predict(X_te) == 1).astype(int)
        proba_nusvm = m.predict_proba(X_te)[:, 1]
        models['Nu-SVM'] = (pred_nusvm, proba_nusvm)
    except Exception as e:
        try:
            m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, 
                    probability=True, class_weight='balanced')
            m.fit(X_tr, y_tr)
            pred_nusvm = (m.predict(X_te) == 1).astype(int)
            proba_nusvm = m.predict_proba(X_te)[:, 1]
            models['Nu-SVM'] = (pred_nusvm, proba_nusvm)
        except:
            print(f"Nu-SVM Error: {e}")
    
    # Evaluate all models
    for name, pred_data in models.items():
        if pred_data is None:
            continue
        pred_b, proba = pred_data
        
        try:
            results[name].append({
                'fold': fold,
                'accuracy': accuracy_score(y_te_b, pred_b),
                'precision': precision_score(y_te_b, pred_b, zero_division=0),
                'recall': recall_score(y_te_b, pred_b, zero_division=0),
                'f1': f1_score(y_te_b, pred_b, zero_division=0),
                'roc_auc': roc_auc_score(y_te_b, proba),
                'balanced_acc': balanced_accuracy_score(y_te_b, pred_b)
            })
        except Exception as e:
            print(f"Evaluation error for {name}: {e}")
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    for name in results:
        if results[name] and len(results[name]) > fold - 1:
            print(f"  {name:20s}: F1={results[name][-1]['f1']:.4f}, AUC={results[name][-1]['roc_auc']:.4f}")

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY (Mean ± Std across folds)")
print("="*80)

summary = {}
for model in results:
    if not results[model]:
        continue
    df_r = pd.DataFrame(results[model])
    summary[model] = {
        'Accuracy': f"{df_r['accuracy'].mean():.4f}±{df_r['accuracy'].std():.4f}",
        'Precision': f"{df_r['precision'].mean():.4f}±{df_r['precision'].std():.4f}",
        'Recall': f"{df_r['recall'].mean():.4f}±{df_r['recall'].std():.4f}",
        'F1 Score': f"{df_r['f1'].mean():.4f}±{df_r['f1'].std():.4f}",
        'ROC-AUC': f"{df_r['roc_auc'].mean():.4f}±{df_r['roc_auc'].std():.4f}",
        'Balanced': f"{df_r['balanced_acc'].mean():.4f}±{df_r['balanced_acc'].std():.4f}"
    }

results_df = pd.DataFrame(summary).T
print("\n" + results_df.to_string())

# Save summary
results_df.to_csv('results_summary.csv')
print("\n✓ Saved: results_summary.csv")

# Visualizations
print("\nCreating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Model Comparison: Class Imbalance Handling', fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_acc']
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Balanced Acc']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECB56', '#DDA15E']

model_names = list(results.keys())
for idx, (metric, label) in enumerate(zip(metrics, labels)):
    ax = axes[idx//3, idx%3]
    data = []
    names = []
    for model in model_names:
        if results[model]:
            data.append(pd.DataFrame(results[model])[metric].values)
            names.append(model)
    
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax.set_ylabel(label, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('01_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_comparison.png")
plt.close()

# Bar plot with error bars
fig, ax = plt.subplots(figsize=(14, 7))

model_names = [m for m in results.keys() if results[m]]
x = np.arange(len(model_names))
width = 0.25

for offset, (metric, label, color) in enumerate(zip(
    ['f1', 'roc_auc', 'balanced_acc'],
    ['F1 Score', 'ROC-AUC', 'Balanced Accuracy'],
    ['#FF6B6B', '#4ECDC4', '#45B7D1'])):
    
    means = []
    stds = []
    for model in model_names:
        if results[model]:
            df_r = pd.DataFrame(results[model])
            means.append(df_r[metric].mean())
            stds.append(df_r[metric].std())
        else:
            means.append(0)
            stds.append(0)
    
    ax.bar(x + offset*width, means, width, label=label, color=color, alpha=0.9,
          yerr=stds, capsize=5, error_kw={'elinewidth': 2})

ax.set_xlabel('Model', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Performance Metrics Comparison (Key Metrics for Imbalanced Data)', fontweight='bold', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('02_metrics_bar.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_metrics_bar.png")
plt.close()

# Per-fold performance
fig, ax = plt.subplots(figsize=(14, 7))

model_names = [m for m in results.keys() if results[m]]
fold_nums = []
f1_scores = {model: [] for model in model_names}

for model in model_names:
    if results[model]:
        for result in results[model]:
            f1_scores[model].append(result['f1'])
            if model == model_names[0]:
                fold_nums.append(result['fold'])

for i, model in enumerate(model_names):
    ax.plot(fold_nums, f1_scores[model], marker='o', label=model, linewidth=2, markersize=8)

ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
ax.set_ylabel('F1 Score', fontweight='bold', fontsize=12)
ax.set_title('F1 Score Per Fold', fontweight='bold', fontsize=14)
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_xticks(fold_nums)

plt.tight_layout()
plt.savefig('03_per_fold_f1.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_per_fold_f1.png")
plt.close()

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nOutput files saved:")
print("  - results_summary.csv")
print("  - 01_comparison.png")
print("  - 02_metrics_bar.png")
print("  - 03_per_fold_f1.png")

# Print detailed summary
print("\n" + "="*80)
print("DETAILED SUMMARY BY METRIC")
print("="*80)

for metric in ['f1', 'roc_auc', 'balanced_acc', 'recall', 'precision']:
    print(f"\n{metric.upper().replace('_', ' ')}:")
    print("-" * 80)
    for model in model_names:
        if results[model]:
            df_r = pd.DataFrame(results[model])
            mean_val = df_r[metric].mean()
            std_val = df_r[metric].std()
            print(f"  {model:20s}: {mean_val:.4f} ± {std_val:.4f}")
