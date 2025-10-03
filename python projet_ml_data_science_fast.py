
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import pickle
import os

# 1) Charger données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Prétraitement
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Modèles
lr = LogisticRegression(max_iter=5000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)  # hyperparams raisonnables

# 5) CV rapide
print("CV F1 LogisticRegression:", cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='f1'))
print("CV F1 RandomForest:", cross_val_score(rf, X_train, y_train, cv=5, scoring='f1'))

# 6) Entraînement final
lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# 7) Prédictions et métriques
y_pred_lr = lr.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)
y_prob_lr = lr.predict_proba(X_test_scaled)[:,1]
y_prob_rf = rf.predict_proba(X_test)[:,1]

def evaluate(y_true, y_pred, y_prob, name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    print(f"--- {name} ---")
    print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, ROC_AUC={roc:.3f}")
    print("Confusion matrix:\n", cm)
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'roc_auc':roc,'cm':cm}

res_lr = evaluate(y_test, y_pred_lr, y_prob_lr, "LogisticRegression")
res_rf = evaluate(y_test, y_pred_rf, y_prob_rf, "RandomForest")

# 8) ROC plots (séparés)
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob_lr, name="LogisticRegression", ax=ax)
ax.set_title("ROC LogisticRegression")
plt.show()

fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="RandomForest", ax=ax)
ax.set_title("ROC RandomForest")
plt.show()

# 9) Sauvegarder le meilleur modèle (ici RandomForest)
os.makedirs('models', exist_ok=True)
path = 'models/rf_breast_cancer.pkl'
with open(path, 'wb') as f:
    pickle.dump({'model': rf, 'scaler': scaler, 'features': list(X.columns)}, f)
print("Modèle sauvegardé:", path)

