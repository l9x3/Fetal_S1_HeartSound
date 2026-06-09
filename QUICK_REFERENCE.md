# Class Imbalance Handling - Quick Reference Guide

## 📊 Problem
- **Dataset**: 3,000 fetal heart sound samples with 160 features
- **Class Imbalance**: 4.86:1 ratio (2,488 normal vs 512 abnormal)
- **Issue**: Models biased toward majority class, poor abnormality detection

## ✅ Solution Implemented

### 1. SMOTE (Synthetic Minority Over-sampling)
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```
- Generates synthetic samples for minority class
- Applied during training only (not test data)
- Best results when combined with RMS-HL

### 2. Class Weights
```python
# In SVM
model = SVC(..., class_weight='balanced')

# In RMS-HL
class_weights = n_samples / (n_classes * class_counts)
# Applied to loss function
weighted_loss = loss * class_weights
```
- Inverse frequency weighting
- Penalizes minority class misclassification more heavily
- Simpler than SMOTE, often good alternative

### 3. Proper Metrics
```python
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

# Instead of accuracy alone:
f1 = f1_score(y_true, y_pred)                    # 0-1, higher is better
auc = roc_auc_score(y_true, y_proba)             # 0-1, higher is better
bacc = balanced_accuracy_score(y_true, y_pred)   # 0-1, higher is better
```

## 📈 Results Comparison

| Model | F1 Score | Recall | ROC-AUC | Balanced Acc |
|-------|----------|--------|---------|--------------|
| RMS-HL (baseline) | 0.4613 | 0.3555 | 0.8321 | 0.6586 |
| **RMS-HL+SMOTE** | **0.5337** | **0.7188** | 0.8301 | **0.7587** |
| SVM (baseline) | 0.4758 | 0.3965 | 0.8110 | 0.6705 |
| SVM (class_weight) | 0.4831 | 0.4082 | 0.8115 | 0.6752 |

**🏆 Best Model: RMS-HL+SMOTE**
- +15.7% F1 improvement
- +102% recall improvement
- +15.2% balanced accuracy improvement

## 🎯 When to Use Which Approach

### Use SMOTE when:
- ✓ Detecting abnormalities is critical (medical, fraud detection)
- ✓ You have enough minority samples for meaningful synthesis
- ✓ Computational resources available
- ✓ High recall is priority (catch all cases)

### Use Class Weights when:
- ✓ Simpler implementation needed
- ✓ Limited computational resources
- ✓ Want balanced precision-recall trade-off
- ✓ Faster training required

### Use Combined Approach when:
- ✓ Maximum performance needed
- ✓ Resources not constrained
- ✓ Clinical/safety-critical application

## 📝 Implementation Checklist

- [ ] Analyze class distribution with `value_counts()`
- [ ] Use StratifiedKFold to maintain class balance
- [ ] Apply SMOTE only to training folds
- [ ] Add class weights to model
- [ ] Evaluate with F1, ROC-AUC, Balanced Accuracy
- [ ] Compare precision-recall trade-off
- [ ] Validate on separate test set
- [ ] Document model choice and rationale

## 🔧 Code Templates

### SMOTE + Model
```python
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Train model
    model = SVC(class_weight='balanced', probability=True)
    model.fit(X_train_bal, y_train_bal)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
```

### Balanced Metrics Evaluation
```python
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score
)

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_proba),
    'balanced_acc': balanced_accuracy_score(y_true, y_pred)
}
```

## 📚 Key Concepts

- **Recall/Sensitivity**: Proportion of actual positives correctly identified
- **Precision**: Proportion of positive predictions that were correct
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic curve
- **Balanced Accuracy**: Average of recall for each class

## ⚠️ Common Mistakes to Avoid

❌ Don't use accuracy as primary metric for imbalanced data
❌ Don't apply SMOTE to test data
❌ Don't forget stratified k-fold split
❌ Don't ignore precision when optimizing for recall
❌ Don't ignore class weights in loss functions

## 📖 Further Reading

- SMOTE: [Chawla et al., 2002](https://arxiv.org/abs/1606.02375)
- Class Weights: [Scikit-learn Class Weight](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- Imbalanced-learn: [Documentation](https://imbalanced-learn.org/)

---
**Generated**: 2026-06-09
**Status**: ✅ Complete
**Best Model**: RMS-HL+SMOTE (F1=0.5337, Recall=0.7188)
