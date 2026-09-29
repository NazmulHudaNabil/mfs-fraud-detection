import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ---

df = pd.read_csv("/Users/mdnazmulhudanabil/Documents/mfs-fraud-detection/Data/frud_data.csv")
df.head()

# ---

df.shape

# ---

df.info()

# ---

df.isnull().sum()

# ---

df.describe()

# ---

df.dtypes

# ---

df.duplicated().sum()

# ---

df["isFraud"].value_counts()

# ---

sns.countplot(x="isFraud", data=df)
plt.title("Count of Fraudulent vs Non-Fraudulent Transactions")
plt.xlabel("Is Fraud")
plt.ylabel("Count")
plt.show()

# ---

df["isFlaggedFraud"].value_counts()

# ---

# Type bazında

sns.countplot(data=df, x="type", hue="isFraud")

# ---

# Oranlar çok uçuk olduğundan sadece isFraud = 1 için bakalım

sns.countplot(
    data=df[df["isFraud"] == 1],
    x="type"
)

plt.title("Fraud Transactions by Transaction Type")
plt.show()


# ---

df["amount"].hist(bins=50)

plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.title("Transaction Amount Distribution")
plt.show()

# ---

print(df[df["amount"]>0])

# ---

pd.crosstab(df["isFlaggedFraud"], df["isFraud"], normalize="index") *100

# ---

df[(df["isFraud"] == 1) & (df["isFlaggedFraud"] == 1)].shape

# ---

matrix = df.corr(numeric_only=True)
plt.figure(figsize=(8,6))
sns.heatmap(matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# ---

pd.crosstab(
    df["type"],
    df["isFraud"],
    normalize="index"
) * 100

# ---

fraud_types = df[df["type"].isin(["TRANSFER", "CASH_OUT"])]
print(fraud_types.shape)

# ---

sns.histplot(
    data=fraud_types,
    x="amount",
    hue="isFraud",
    bins=50
)

plt.title("Transaction Amount: Fraud vs Normal")
plt.show()

# ---

sns.histplot(
    data=fraud_types[fraud_types["isFraud"] == 1],
    x="amount",
    bins=50
)

plt.title("Fraud Transaction Amount Distribution")
plt.show()

# ---

sns.histplot(
    data=fraud_types[fraud_types["isFraud"] == 0],
    x="amount",
    bins=50
)

plt.title("Normal Transaction Amount Distribution")
plt.show()

# ---

df["balance_change_org"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# ---

print(df["balance_change_org"])

# ---

transfer_df = df[df["type"] == "TRANSFER"].copy()

# ---

transfer_df.groupby("isFraud")["balance_change_org"].agg(
    ["count", "mean", "median"]
)

# ---

sns.boxplot(
    data=transfer_df,
    x="isFraud",
    y="balance_change_org"
)

plt.title("Balance Change by Fraud Status")
plt.show()

# ---

df["balance_error_org"] = (
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

# ---

df["balance_change_dest"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# ---

df.head()

# ---

df["hour"] = df["step"] % 24

# ---

X = df[[
    "step",
    "type",
    "amount"
]]

y = df["isFraud"]

# ---

X = pd.get_dummies(X, columns=["type"], drop_first=True)
X.head()

# ---

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y # using stratify=y because the proportion of fraud is very low and we want to maintain that ratio
)

# ---

print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())

# ---

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---

X_train_scaled[:5]

# ---

# Baseline Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# ---

y_pred = model.predict(X_test_scaled)

# ---

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ---

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

# ---

tn, fp, fn, tp = cm.ravel()

print("TN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

# ---

y_proba = model.predict_proba(X_test_scaled)[:, 1]
y_proba[:5]

# ---

print(y_proba.max())

# ---

print((y_proba >= 0.5).sum())

# ---

from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))

print(
    "ROC-AUC:",
    roc_auc_score(y_test, y_proba)
)

# ---

threshold = 0.10

y_pred_01 = (y_proba >= threshold).astype(int)

# ---

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_01))
print(classification_report(y_test, y_pred_01))

# ---

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_proba
)

# ---

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))

plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Baseline Logistic Regression")

plt.show()

# ---

df.columns

# ---

X2 = df[[
    "step",
    "type",
    "amount",
    "balance_change_org",
    "balance_error_org",
    "balance_change_dest",
    "hour"
]]

y = df["isFraud"]

# ---

X2 = pd.get_dummies(
    X2,
    columns=["type"],
    drop_first=True
)

# ---

X2[:5]

# ---

df["balance_error_org"].describe()

# ---

(df["balance_error_org"] != 0).sum()

# ---

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---

scaler2 = StandardScaler()

X2_train_scaled = scaler2.fit_transform(X2_train)
X2_test_scaled = scaler2.transform(X2_test)

# ---

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model2.fit(X2_train_scaled, y2_train)

y2_pred = model2.predict(X2_test_scaled)
y2_proba = model2.predict_proba(X2_test_scaled)[:, 1]

# ---

print("Confusion Matrix:")
print(confusion_matrix(y2_test, y2_pred))

print("\nClassification Report:")
print(classification_report(y2_test, y2_pred))

print(
    "\nROC-AUC:",
    roc_auc_score(y2_test, y2_proba)
)


# ---

import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

TUNING_ROWS = min(200_000, len(X2_train))
X_tune, _, y_tune, _ = train_test_split(
    X2_train,
    y2_train,
    train_size=TUNING_ROWS,
    random_state=42,
    stratify=y2_train
)
X_rf_train, X_rf_valid, y_rf_train, y_rf_valid = train_test_split(
    X_tune,
    y_tune,
    test_size=0.2,
    random_state=42,
    stratify=y_tune
)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
        "max_depth": trial.suggest_int("max_depth", 8, 24, step=4),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        "n_jobs": -1,
        "random_state": 42,
    }
    classifier = RandomForestClassifier(**params)
    classifier.fit(X_rf_train, y_rf_train)
    validation_probability = classifier.predict_proba(X_rf_valid)[:, 1]
    return roc_auc_score(y_rf_valid, validation_probability)

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=12, timeout=300, show_progress_bar=False)

print(f"Completed {len(study.trials)} trials")
print(f"Best validation ROC-AUC: {study.best_value:.4f}")
print("Best parameters:")
print(study.best_params)

# ---

best_rf = RandomForestClassifier(
    **study.best_params,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X2_train, y2_train)

best_rf_prediction = best_rf.predict(X2_test)
best_rf_probability = best_rf.predict_proba(X2_test)[:, 1]

print("Tuned Random Forest")
print("Confusion Matrix:")
print(confusion_matrix(y2_test, best_rf_prediction))
print("\nClassification Report:")
print(classification_report(y2_test, best_rf_prediction))
print("\nTest ROC-AUC:", roc_auc_score(y2_test, best_rf_probability))

# ---

