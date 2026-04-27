# Final Project - Modeling
# Sachin Patel
# U01124876

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, roc_curve

import warnings
warnings.filterwarnings('ignore')

seed = 42
np.random.seed(seed)

# load cleaned data
df = pd.read_csv('data/dataco_cleaned.csv')

print("Data loaded")
print("Shape:", df.shape)


# drop repeated columns
drop_cols = [
    'Benefit per order',
    'Sales per customer',
    'Order Item Product Price'
]
df = df.drop(columns=drop_cols)
print("New shape:", df.shape)
# split target and features
y = df['Late_delivery_risk']
X = df.drop(columns=['Late_delivery_risk'])
print("X shape:", X.shape)
print("y shape:", y.shape)

# one hot encoding
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

print("Categorical cols:", cat_cols)
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
print("Shape after encoding:", X_encoded.shape)

# train, validation, test split Data
X_temp, X_test, y_temp, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.15,
    random_state=seed,
    stratify=y
)

# split train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.1765,
    random_state=seed,
    stratify=y_temp
)

print("Train Size :", X_train.shape[0])
print("Validation Size:", X_val.shape[0])
print("Test Size:", X_test.shape[0])

print("\nLate % in splits ")
print("Train Size:", round(y_train.mean()*100,1))
print("Validation Size:", round(y_val.mean()*100,1))
print("Test Size:", round(y_test.mean()*100,1))


# scaling for logistic regression
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("Numeric cols:", len(num_cols))

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val_scaled[num_cols] = scaler.transform(X_val[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

print("Scaling done")

# logistic regression

print("\nLogistic Regression")

lr_param_grid = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'max_iter': [500]
}

print("Tuning model...")
print("C values tried:", len(lr_param_grid['C']))

lr_grid = GridSearchCV(
    LogisticRegression(random_state=seed),
    lr_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

lr_grid.fit(X_train_scaled, y_train)

lr_best = lr_grid.best_estimator_

print("\nBest parameters:", lr_grid.best_params_)
print("CV F1:", round(lr_grid.best_score_,4))

lr_val_pred = lr_best.predict(X_val_scaled)

print("\nValidation results")
print("Accuracy Score:- ", round(accuracy_score(y_val, lr_val_pred),4))
print("Precision Score:- ", round(precision_score(y_val, lr_val_pred),4))
print("Recall Score:- ", round(recall_score(y_val, lr_val_pred),4))
print("F1-score:- ", round(f1_score(y_val, lr_val_pred),4))

# decision tree

print("\nDecision Tree")

dt_param_grid = {
    'max_depth': [5,10,15,20,None],
    'min_samples_split': [2,10,20],
    'min_samples_leaf': [1,5,10],
    'criterion': ['gini','entropy']
}

print("Tuning model...")

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=seed),
    dt_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

dt_best = dt_grid.best_estimator_

print("\nBest parameters:", dt_grid.best_params_)
print("CV F1:", round(dt_grid.best_score_,4))

dt_val_pred = dt_best.predict(X_val)

print("\nValidation results")
print("Accuracy Score :", round(accuracy_score(y_val, dt_val_pred),4))
print("Precision Score:", round(precision_score(y_val, dt_val_pred),4))
print("Recall Score   :", round(recall_score(y_val, dt_val_pred),4))
print("F1 Score       :", round(f1_score(y_val, dt_val_pred),4))

print()

# random forest

print("\nRandom Forest")

rf_param_grid = {
    'n_estimators': [100,200],
    'max_depth': [10,20,None],
    'min_samples_split': [2,10],
    'min_samples_leaf': [1,5]
}

print("Tuning model...")
print("Please wait, this one takes longer...")

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=seed, n_jobs=1),
    rf_param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

rf_best = rf_grid.best_estimator_

print("\nBest parameters:", rf_grid.best_params_)
print("CV F1:", round(rf_grid.best_score_,4))

rf_val_pred = rf_best.predict(X_val)

print("\nValidation results")
print("Accuracy Score :", round(accuracy_score(y_val, rf_val_pred),4))
print("Precision Score:", round(precision_score(y_val, rf_val_pred),4))
print("Recall Score   :", round(recall_score(y_val, rf_val_pred),4))
print("F1 Score       :", round(f1_score(y_val, rf_val_pred),4))

print()

# final test results

print("\nFinal Test Results")

# logistic regression on test set
print("\nLogistic Regression")

lr_pred = lr_best.predict(X_test_scaled)
lr_prob = lr_best.predict_proba(X_test_scaled)[:,1]

lr_acc = accuracy_score(y_test, lr_pred)
lr_prec = precision_score(y_test, lr_pred)
lr_rec = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_auc = roc_auc_score(y_test, lr_prob)
lr_cm = confusion_matrix(y_test, lr_pred)

print("Accuracy Score :", round(lr_acc,4))
print("Precision Score:", round(lr_prec,4))
print("Recall Score   :", round(lr_rec,4))
print("F1 Score       :", round(lr_f1,4))
print("ROC-AUC        :", round(lr_auc,4))
print("Confusion Matrix")
print(lr_cm)


# decision tree on test set
print("\nDecision Tree")

dt_pred = dt_best.predict(X_test)
dt_prob = dt_best.predict_proba(X_test)[:,1]

dt_acc = accuracy_score(y_test, dt_pred)
dt_prec = precision_score(y_test, dt_pred)
dt_rec = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)
dt_auc = roc_auc_score(y_test, dt_prob)
dt_cm = confusion_matrix(y_test, dt_pred)

print("Accuracy Score :", round(dt_acc,4))
print("Precision Score:", round(dt_prec,4))
print("Recall Score   :", round(dt_rec,4))
print("F1 Score       :", round(dt_f1,4))
print("ROC-AUC        :", round(dt_auc,4))
print("Confusion Matrix")
print(dt_cm)


# random forest on test set
print("\nRandom Forest")

rf_pred = rf_best.predict(X_test)
rf_prob = rf_best.predict_proba(X_test)[:,1]

rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
rf_cm = confusion_matrix(y_test, rf_pred)

print("Accuracy Score :", round(rf_acc,4))
print("Precision Score:", round(rf_prec,4))
print("Recall Score   :", round(rf_rec,4))
print("F1 Score       :", round(rf_f1,4))
print("ROC-AUC        :", round(rf_auc,4))
print("Confusion Matrix")
print(rf_cm)


# plots

print("\nCreating plots")

# confusion matrices for all 3 models
fig, axes = plt.subplots(1,3, figsize=(18,5))

# logistic regression cm
axes[0].imshow(lr_cm, cmap='Blues')
axes[0].set_title('Logistic Regression\nAcc: ' + str(round(lr_acc,4)) + ' | F1: ' + str(round(lr_f1,4)))
axes[0].set_xticks([0,1])
axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['On-Time','Late'])
axes[0].set_yticklabels(['On-Time','Late'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(lr_cm[i,j]), ha='center', va='center')

# decision tree cm
axes[1].imshow(dt_cm, cmap='Blues')
axes[1].set_title('Decision Tree\nAcc: ' + str(round(dt_acc,4)) + ' | F1: ' + str(round(dt_f1,4)))
axes[1].set_xticks([0,1])
axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['On-Time','Late'])
axes[1].set_yticklabels(['On-Time','Late'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(dt_cm[i,j]), ha='center', va='center')

# random forest cm
axes[2].imshow(rf_cm, cmap='Blues')
axes[2].set_title('Random Forest\nAcc: ' + str(round(rf_acc,4)) + ' | F1: ' + str(round(rf_f1,4)))
axes[2].set_xticks([0,1])
axes[2].set_yticks([0,1])
axes[2].set_xticklabels(['On-Time','Late'])
axes[2].set_yticklabels(['On-Time','Late'])
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(rf_cm[i,j]), ha='center', va='center')

plt.suptitle('Confusion Matrices')
plt.tight_layout()

plt.savefig('results/plots/05_confusion_matrices.png',
            dpi=100,
            bbox_inches='tight')

plt.close()

fig, ax = plt.subplots(figsize=(11,6))

metrics_names = ['Accuracy','Precision','Recall','F1','AUC']
x = np.arange(len(metrics_names))
width = 0.25

lr_vals = [lr_acc, lr_prec, lr_rec, lr_f1, lr_auc]
dt_vals = [dt_acc, dt_prec, dt_rec, dt_f1, dt_auc]
rf_vals = [rf_acc, rf_prec, rf_rec, rf_f1, rf_auc]

ax.bar(x - width, lr_vals, width, label='Logistic Regression', color='#3498db')
ax.bar(x, dt_vals, width, label='Decision Tree', color='#2ecc71')
ax.bar(x + width, rf_vals, width, label='Random Forest', color='#e74c3c')

ax.set_title('Model Comparison')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0.93,1.01)

plt.tight_layout()

plt.savefig('results/plots/06_model_comparison.png',
            dpi=100,
            bbox_inches='tight')

plt.close()


fig, ax = plt.subplots(figsize=(10,8))

feature_imp = pd.Series(rf_best.feature_importances_, index=X_train.columns)
top_15 = feature_imp.sort_values(ascending=False).head(15)

top_15.sort_values().plot(kind='barh', ax=ax, color='#16a085')

ax.set_title('Top 15 Random Forest Features')
ax.set_xlabel('Importance')

plt.tight_layout()

plt.savefig('results/plots/07_feature_importance.png',
            dpi=100,
            bbox_inches='tight')

plt.close()

print("Plots saved")

# roc curves
fig, ax = plt.subplots(figsize=(9,7))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
ax.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = ' + str(round(lr_auc,4)) + ')', linewidth=2)

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)
ax.plot(fpr_dt, tpr_dt, label='Decision Tree (AUC = ' + str(round(dt_auc,4)) + ')', linewidth=2)

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
ax.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = ' + str(round(rf_auc,4)) + ')', linewidth=2)

ax.plot([0,1], [0,1], 'k--', label='Random (AUC = 0.5)')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - All Models')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig('results/plots/08_roc_curves.png',
            dpi=100,
            bbox_inches='tight')

plt.close()


# save models

print("\nSaving models")

with open('results/lr_model.pkl', 'wb') as f:
    pickle.dump(lr_best, f)
print("Saved: results/lr_model.pkl")

with open('results/dt_model.pkl', 'wb') as f:
    pickle.dump(dt_best, f)
print("Saved: results/dt_model.pkl")

with open('results/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_best, f)
print("Saved: results/rf_model.pkl")

print("\nBaseline late rate: 0.5483")
print("Models and plots saved")