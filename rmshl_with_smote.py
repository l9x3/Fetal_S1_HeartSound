"""
RMS-HL: Fast Demonstration with SMOTE-Based Resampling
Enhanced with Synthetic Minority Over-sampling Technique (SMOTE)
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
    roc_auc_score, balanced_accuracy_score
)
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
import warnings
import time
warnings.filterwarnings('ignore')

print("="*80)
print("RMS-HL: Robust Median-Based Supervised Hyperplane Learner")
print("Enhanced with SMOTE-Based Resampling")
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

# RMS-HL Classifier
class RMSHL(BaseEstimator, ClassifierMixin):
    def __init__(self, lam=0.1, delta=1.0, max_iter=20, random_state=None):
        self.lam = lam
        self.delta = delta
        self.max_iter = max_iter
        self.random_state = random_state
        self.w = None
        self.b = None
        self.scaler = None
        
    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        y = np.where(y == np.unique(y)[0], -1, 1)
        
        n = X.shape[1]
        p = np.zeros(n + 1)
        
        for i in range(self.max_iter):
            w = p[:n]
            # Compute medians
            for label in [-1, 1]:
                mask = y == label
                if np.any(mask):
                    med = geometric_median(X[mask])
            # Optimize
            res = minimize(
                lambda p: self.lam*np.sum(p[:n]**2) + np.sum(huber_loss(1-y*(X@p[:n]+p[n]), self.delta)),
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
df = pd.read_csv('/tmp/workspace/l9x3/Fetal_S1_HeartSound/dataset_contaminated.csv')
X = df.drop('label', axis=1).values
y = df['label'].values

print(f"Original Data - Samples: {len(X)}, Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)} with counts {np.bincount(np.where(y==1, 1, 0))}")
print(f"Class -1: {np.sum(y==-1)}, Class 1: {np.sum(y==1)}")
print(f"Class Imbalance Ratio: {np.sum(y==-1) / np.sum(y==1):.2f}:1")

# Global scaling (applied once)
scaler_global = StandardScaler()
X = scaler_global.fit_transform(X)

print("\nRunning 2-Fold Cross-Validation with SMOTE...")
print("-" * 80)

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
results = {
    'RMS-HL': [], 
    'RMS-HL+SMOTE': [], 
    'SVM': [], 
    'SVM+SMOTE': [],
    'Nu-SVM': [],
    'Nu-SVM+SMOTE': [],
    'Ramp-SVM': [],
    'Ramp-SVM+SMOTE': []
}

fold_data = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\nFold {fold}/2 ...", end=" ", flush=True)
    t0 = time.time()
    
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    y_te_b = (y_te == 1).astype(int)
    
    # Apply SMOTE to training data
    try:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_tr_smote, y_tr_smote = smote.fit_resample(X_tr, y_tr)
        smote_applied = True
        print(f"\n  SMOTE: {X_tr.shape[0]} -> {X_tr_smote.shape[0]} samples")
        print(f"  Original: class -1: {np.sum(y_tr==-1)}, class 1: {np.sum(y_tr==1)}")
        print(f"  After SMOTE: class -1: {np.sum(y_tr_smote==-1)}, class 1: {np.sum(y_tr_smote==1)}")
    except Exception as e:
        print(f"\n  SMOTE Error: {e}. Skipping SMOTE for this fold.")
        X_tr_smote, y_tr_smote = X_tr, y_tr
        smote_applied = False
    
    models = {}
    
    # RMS-HL (without SMOTE)
    try:
        m = RMSHL(lam=0.1, delta=1.0, max_iter=15, random_state=42)
        m.fit(X_tr, y_tr)
        models['RMS-HL'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  RMS-HL Error: {e}")
        models['RMS-HL'] = None
    
    # RMS-HL (with SMOTE)
    try:
        m = RMSHL(lam=0.1, delta=1.0, max_iter=15, random_state=42)
        m.fit(X_tr_smote, y_tr_smote)
        models['RMS-HL+SMOTE'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  RMS-HL+SMOTE Error: {e}")
        models['RMS-HL+SMOTE'] = None
    
    # SVM (without SMOTE)
    try:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr, y_tr)
        models['SVM'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  SVM Error: {e}")
        models['SVM'] = None
    
    # SVM (with SMOTE)
    try:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr_smote, y_tr_smote)
        models['SVM+SMOTE'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  SVM+SMOTE Error: {e}")
        models['SVM+SMOTE'] = None
    
    # Nu-SVM (without SMOTE)
    try:
        m = NuSVC(kernel='rbf', nu=0.3, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr, y_tr)
        models['Nu-SVM'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr, y_tr)
        models['Nu-SVM'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    
    # Nu-SVM (with SMOTE)
    try:
        m = NuSVC(kernel='rbf', nu=0.3, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr_smote, y_tr_smote)
        models['Nu-SVM+SMOTE'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except:
        m = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr_smote, y_tr_smote)
        models['Nu-SVM+SMOTE'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    
    # Ramp-SVM (without SMOTE)
    try:
        m = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr, y_tr)
        models['Ramp-SVM'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  Ramp-SVM Error: {e}")
        models['Ramp-SVM'] = None
    
    # Ramp-SVM (with SMOTE)
    try:
        m = SVC(kernel='rbf', C=1, gamma='scale', random_state=42, probability=True)
        m.fit(X_tr_smote, y_tr_smote)
        models['Ramp-SVM+SMOTE'] = (m.predict(X_te), m.predict_proba(X_te)[:, 1])
    except Exception as e:
        print(f"  Ramp-SVM+SMOTE Error: {e}")
        models['Ramp-SVM+SMOTE'] = None
    
    # Evaluate
    for name, pred_data in models.items():
        if pred_data is None:
            continue
        pred, proba = pred_data
        pred_b = (pred == 1).astype(int)
        
        metrics_dict = {
            'fold': fold,
            'accuracy': accuracy_score(y_te_b, pred_b),
            'precision': precision_score(y_te_b, pred_b, zero_division=0),
            'recall': recall_score(y_te_b, pred_b, zero_division=0),
            'f1': f1_score(y_te_b, pred_b, zero_division=0),
            'roc_auc': roc_auc_score(y_te_b, proba),
            'balanced_acc': balanced_accuracy_score(y_te_b, pred_b)
        }
        results[name].append(metrics_dict)
        fold_data.append({'fold': fold, 'model': name, **metrics_dict})
    
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    for name in results:
        if results[name] and results[name][-1]['fold'] == fold:
            print(f"  {name:15s}: F1={results[name][-1]['f1']:.4f}, AUC={results[name][-1]['roc_auc']:.4f}")

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY (Mean ± Std across folds)")
print("="*80)

summary = {}
for model in sorted(results.keys()):
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

summary_df = pd.DataFrame(summary).T
print("\n" + summary_df.to_string())

# Save detailed results
print("\n" + "="*80)
print("Saving results...")
print("="*80)

summary_df.to_csv('/tmp/workspace/l9x3/Fetal_S1_HeartSound/results_summary_smote.csv')
print("\n✓ Saved: results_summary_smote.csv")

fold_results_df = pd.DataFrame(fold_data)
fold_results_df.to_csv('/tmp/workspace/l9x3/Fetal_S1_HeartSound/results_detailed_smote.csv', index=False)
print("✓ Saved: results_detailed_smote.csv")

# Visualizations
print("\nCreating visualizations...")

# 1. Comparison boxplot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('RMS-HL with/without SMOTE vs Baseline Models: Performance Comparison', 
             fontsize=16, fontweight='bold')

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'balanced_acc']
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Balanced Acc']
model_names = sorted([m for m in results.keys() if results[m]])
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF', '#48DBFB']

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
plt.savefig('/tmp/workspace/l9x3/Fetal_S1_HeartSound/01_comparison_smote.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_comparison_smote.png")
plt.close()

# 2. SMOTE Impact Comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Impact of SMOTE on Model Performance', fontsize=14, fontweight='bold')

base_models = ['RMS-HL', 'SVM', 'Nu-SVM', 'Ramp-SVM']
metrics_to_plot = ['f1', 'roc_auc', 'balanced_acc']
metric_labels = ['F1 Score', 'ROC-AUC', 'Balanced Accuracy']

for ax_idx, (metric, metric_label) in enumerate(zip(metrics_to_plot, metric_labels)):
    ax = axes[ax_idx]
    x_pos = np.arange(len(base_models))
    width = 0.35
    
    without_smote = []
    with_smote = []
    
    for base_model in base_models:
        if results[base_model]:
            df_without = pd.DataFrame(results[base_model])
            without_smote.append(df_without[metric].mean())
        else:
            without_smote.append(0)
        
        smote_model = f"{base_model}+SMOTE"
        if smote_model in results and results[smote_model]:
            df_with = pd.DataFrame(results[smote_model])
            with_smote.append(df_with[metric].mean())
        else:
            with_smote.append(0)
    
    ax.bar(x_pos - width/2, without_smote, width, label='Without SMOTE', color='#FF6B6B', alpha=0.9)
    ax.bar(x_pos + width/2, with_smote, width, label='With SMOTE', color='#4ECDC4', alpha=0.9)
    
    ax.set_ylabel(metric_label, fontweight='bold')
    ax.set_title(metric_label, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(base_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('/tmp/workspace/l9x3/Fetal_S1_HeartSound/02_smote_impact.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_smote_impact.png")
plt.close()

# 3. Bar plot with error bars
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(model_names))
width = 0.2

for offset, (metric, label, color) in enumerate(zip(
    ['accuracy', 'f1', 'roc_auc'],
    ['Accuracy', 'F1 Score', 'ROC-AUC'],
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
ax.set_title('Performance Metrics Comparison (With SMOTE Integration)', fontweight='bold', fontsize=14)
ax.set_xticks(x + width)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('/tmp/workspace/l9x3/Fetal_S1_HeartSound/03_metrics_bar_smote.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_metrics_bar_smote.png")
plt.close()

# 4. Detailed comparison table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = []
table_data.append(['Model'] + list(summary[list(summary.keys())[0]].keys()))
for model in sorted(summary.keys()):
    row = [model] + list(summary[model].values())
    table_data.append(row)

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15] + [0.14]*6)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Color header
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(len(table_data[0])):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('#ffffff')

plt.title('Detailed Performance Results Summary', fontweight='bold', fontsize=14, pad=20)
plt.savefig('/tmp/workspace/l9x3/Fetal_S1_HeartSound/04_results_table.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_results_table.png")
plt.close()

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nOutput files saved to /tmp/workspace/l9x3/Fetal_S1_HeartSound/:")
print("  - results_summary_smote.csv")
print("  - results_detailed_smote.csv")
print("  - 01_comparison_smote.png")
print("  - 02_smote_impact.png")
print("  - 03_metrics_bar_smote.png")
print("  - 04_results_table.png")

# Summary statistics
print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)
print("\nBest Models by Metric (Mean across folds):")
for metric in ['f1', 'roc_auc', 'balanced_acc']:
    best_model = None
    best_score = -1
    for model in results:
        if results[model]:
            df_r = pd.DataFrame(results[model])
            score = df_r[metric].mean()
            if score > best_score:
                best_score = score
                best_model = model
    print(f"  {metric.upper():15s}: {best_model:15s} = {best_score:.4f}")

print("\n" + "="*80)
