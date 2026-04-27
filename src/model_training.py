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

print("Dropped repeated cols:", len(drop_cols))
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

print("Late % in splits ")
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

models = {
    'Logistic Regression': (lr_best, X_test_scaled),
    'Decision Tree': (dt_best, X_test),
    'Random Forest': (rf_best, X_test)
}

results = {}

for name, (model, xdata) in models.items():

    pred = model.predict(xdata)
    prob = model.predict_proba(xdata)[:,1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, prob)
    cm = confusion_matrix(y_test, pred)

    results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'auc': auc,
        'cm': cm,
        'predictions': pred
    }

    print("\n" + name)
    print("Accuracy Score :", round(acc,4))
    print("Precision Score:", round(prec,4))
    print("Recall Score   :", round(rec,4))
    print("F1 Score       :", round(f1,4))
    print("ROC-AUC        :", round(auc,4))

    print("Confusion Matrix")
    print(cm)
# plots

print("\nCreating plots")

# confusion matrices
fig, axes = plt.subplots(1,3, figsize=(18,5))

for idx, (name, res) in enumerate(results.items()):
    cm = res['cm']
    ax = axes[idx]

    ax.imshow(cm, cmap='Blues')
    ax.set_title(name + "\nAcc: " + str(round(res['accuracy'],4)) +
                 " | F1: " + str(round(res['f1'],4)))

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(['On-Time','Late'])
    ax.set_yticklabels(['On-Time','Late'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center')

plt.suptitle('Confusion Matrices')
plt.tight_layout()

plt.savefig('results/plots/05_confusion_matrices.png',
            dpi=100,
            bbox_inches='tight')

plt.close()


# model comparison
fig, ax = plt.subplots(figsize=(11,6))

metrics_names = ['Accuracy','Precision','Recall','F1','AUC']
metrics_keys = ['accuracy','precision','recall','f1','auc']

x = np.arange(len(metrics_names))
width = 0.25
colors_list = ['#3498db','#2ecc71','#e74c3c']

for i, (name, res) in enumerate(results.items()):
    vals = [res[k] for k in metrics_keys]
    ax.bar(x + i*width, vals, width, label=name, color=colors_list[i])

ax.set_title('Model Comparison')
ax.set_ylabel('Score')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim(0.93,1.01)

plt.tight_layout()

plt.savefig('results/plots/06_model_comparison.png',
            dpi=100,
            bbox_inches='tight')

plt.close()


# random forest feature importance
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

for name, (model, xdata) in models.items():
    prob = model.predict_proba(xdata)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, label=name + ' (AUC = ' + str(round(auc_val,4)) + ')', linewidth=2)

# diagonal line for random model
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


# save trained models

print("\nSaving models")

with open('results/trained_models.pkl', 'wb') as f:
    pickle.dump({
        'logistic_regression': lr_best,
        'decision_tree': dt_best,
        'random_forest': rf_best,
        'lr_best_params': lr_grid.best_params_,
        'dt_best_params': dt_grid.best_params_,
        'rf_best_params': rf_grid.best_params_,
        'results': results
    }, f)

print("Saved: results/trained_models.pkl")


# final summary

print("\nFinal test summary")

print("Model               Acc      Prec     Recall   F1       AUC")
print("----------------------------------------------------------")

for name, res in results.items():
    print(
        f"{name:20s} "
        f"{res['accuracy']:.4f}   "
        f"{res['precision']:.4f}   "
        f"{res['recall']:.4f}   "
        f"{res['f1']:.4f}   "
        f"{res['auc']:.4f}"
    )

print("\nBaseline late rate:", 0.5483)
print("Models and plots saved")