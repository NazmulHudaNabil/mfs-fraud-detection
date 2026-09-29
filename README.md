# 🛡️ Mobile Financial Service (MFS) Fraud Detection

A robust, production-ready machine learning pipeline for detecting fraudulent transactions in Mobile Financial Services (MFS). This project emphasizes rigorous data science methodologies—from Exploratory Data Analysis (EDA) and custom mathematical Feature Engineering to Bayesian Hyperparameter Tuning—to accurately identify fraudulent behavior in a massive, highly imbalanced dataset.

---

## 💾 The Dataset

The project utilizes a massive dataset containing over **6.3 million financial transactions**. Like most real-world financial data, it suffers from extreme class imbalance, where less than 1% of the transactions are actual fraud. Handling this massive scale while preventing the model from simply predicting "Legitimate" for every transaction was the core technical challenge of this project.

---

## 🧠 Machine Learning Methodology

### 1. Exploratory Data Analysis (EDA)
Extensive EDA revealed that fraudulent transactions do not just have larger amounts—they have distinct mathematical discrepancies. Fraudulent transfers often resulted in logical errors between the stated `amount` and the actual change in the sender's (`oldbalanceOrg` vs `newbalanceOrig`) and recipient's balances. This insight directly drove our feature engineering strategy.

### 2. Custom Feature Engineering
Raw data was insufficient for the model to capture the nuances of fraud. We engineered explicit signals to highlight the discrepancies discovered during EDA:
- `balance_change_org`: Tracked the exact movement of the sender's balance (`oldbalanceOrg - newbalanceOrig`).
- `balance_error_org`: Detected hidden manipulations or unrecorded fees (`oldbalanceOrg - amount - newbalanceOrig`).
- `balance_change_dest`: Tracked the recipient's balance changes.
- `hour`: Extracted daily temporal patterns by converting the raw `step` feature (`step % 24`).

### 3. Data Transformation (Preprocessing)
To prevent data leakage and ensure seamless production deployments, all transformations were strictly encapsulated in a `ColumnTransformer` pipeline:
- **Categorical Variables (`type`):** Processed using `OneHotEncoder` (with `drop="first"` to prevent multicollinearity).
- **Numerical Variables:** Standardized using `StandardScaler` to ensure all features contributed equally to the model's decision-making process.

---

## ⚙️ Model Selection & Evaluation

Given the massive 6.3 million row dataset, model selection required balancing raw predictive power with training time efficiency.

1. **Gradient Boosting (XGBoost / CatBoost):** While these models are traditionally powerful, they were ultimately discarded. The extreme training times on a dataset of this scale made iterative hyperparameter tuning and deployment highly impractical.
2. **Logistic Regression:** This was tested as a fast baseline model. While it achieved an impressive 90% Precision and an ROC-AUC of 0.988, it only achieved **47% Recall**. In the context of financial fraud, missing 53% of fraudulent transactions is unacceptable.
3. **Random Forest Classifier (Chosen Model):** Random Forest offered the perfect balance. It was significantly faster to train than XGBoost on this dataset and easily captured complex non-linear patterns.

### Hyperparameter Tuning (Optuna)
To squeeze out maximum performance from the Random Forest, we utilized **Optuna** for Bayesian hyperparameter optimization. The most critical tuning choice was enforcing `class_weight='balanced'`, which heavily penalized the model for missing the minority fraud class. Tuned parameters included `n_estimators` (125), `max_depth` (24), `min_samples_split` (10), and `min_samples_leaf` (2).

---

## 📊 Model Performance

In fraud detection, standard accuracy is highly misleading (predicting everything as "Legitimate" yields a 99% accuracy but is completely useless). We prioritized **Recall** to ensure fraudulent transactions were successfully caught.

| Metric | Before Feature Engineering | Logistic Regression | Random Forest (Tuned) |
|--------|----------------------------|---------------------|-----------------------|
| **Accuracy** | 99.87% | ~99.90% | **~99.70%** |
| **Recall (Fraud Detected)**| 0.00% | 47.00% | **88.00%** |
| **Precision (Fraud)**| 0.00% | 90.00% | **26.00%** |
| **F1 Score** | 0.00 | 0.62 | **0.40** |
| **ROC-AUC** | 0.896 | 0.988 | **0.996** |

**💡 Conclusion:** By leveraging custom feature engineering and a tuned Random Forest model, we successfully shifted the model from catching 47% of fraud (Logistic Regression) to catching **88% of all fraudulent transactions**!

---

## 🚀 Key Engineering Features

- **End-to-End Modular Pipeline:** Clean architectural separation of Data Ingestion, Preprocessing, Feature Engineering, Transformation, and Model Training.
- **MLflow Tracking:** Automated tracking of metrics, tuned parameters, and serialized model artifacts.
- **Interactive Web UI:** A beautiful, responsive Gradio interface mounted directly onto a FastAPI backend for real-time inference.
- **Dockerized & Deployment Ready:** A highly optimized `Dockerfile` (Render & Heroku compatible) leveraging `requirements.txt` and multi-stage layer caching.
- **Robust Testing:** Integrated unit tests using `pytest`.

---

## 💻 Getting Started

### 1. Installation
Ensure you have Python 3.12+ installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Application
Run the full API and the Web UI simultaneously with a single command:
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once the server starts, open your browser and navigate to:
👉 **http://127.0.0.1:8000/**

### 3. Retraining the Pipeline
To re-run the entire machine learning pipeline on your data (generates new models in `artifacts/`):
```bash
python -m src.pipline.training_pipline
```

### 4. Running Tests
To execute the unit tests for data preprocessing and feature engineering mathematical logic:
```bash
PYTHONPATH=. pytest scripts/
```

---

## 🐳 Docker Deployment

This project includes a clean, production-ready Dockerfile that automatically handles dynamic `$PORT` assignments for platforms like **Render** and **Heroku**.

```bash
# Build the image locally
docker build -t mfs-fraud-detection .

# Run the container
docker run -p 8000:8000 mfs-fraud-detection
```

---

## 📁 Project Structure

```text
├── artifacts/             # Serialized models and ColumnTransformers (.pkl)
├── Data/                  # Raw dataset (Ignored in Docker)
├── mlruns/                # MLflow tracking data
├── notebook/              # Exploratory Data Analysis & Optuna Tuning
├── scripts/               # Pytest unit tests verifying ML logic
├── src/                   # Core application code
│   ├── app/               # FastAPI backend and Gradio UI integration
│   ├── component/         # ML pipeline steps (Ingestion, FE, Transformation, Training)
│   └── pipline/           # Orchestrator script for the pipeline
├── Dockerfile             # Optimized Docker build
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
