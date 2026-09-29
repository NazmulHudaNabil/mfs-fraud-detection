# 🛡️ Mobile Financial Service (MFS) Fraud Detection

A robust, production-ready machine learning pipeline for detecting fraudulent transactions in Mobile Financial Services (MFS). This project leverages custom feature engineering and a tuned Random Forest model to accurately identify fraudulent behavior in a highly imbalanced dataset.

---

## 📊 Model Performance

In fraud detection, standard accuracy is highly misleading due to class imbalance (predicting everything as "Legitimate" yields a high accuracy but is useless). The true impact of our custom feature engineering is demonstrated by the massive improvement in the model's ability to actually detect fraud (**Recall**).

| Metric | Before Feature Engineering | After Feature Engineering (Tuned) |
|--------|----------------------------|-----------------------------------|
| **Accuracy** | 99.87% | **~99.70%** |
| **Recall (Fraud Detection)** | 0.00% | **88.00%** |
| **Precision (Fraud)**| 0.00% | **26.00%** |
| **F1 Score** | 0.00 | **0.40** |
| **ROC-AUC** | 0.896 | **0.996** |

**💡 Conclusion:** Before feature engineering, the model failed to catch *any* fraud (0% Recall). After applying mathematical feature engineering (e.g., calculating `balance_error_org`, `balance_change_dest`) and hyperparameter tuning with `class_weight='balanced'`, the model now successfully catches **88% of all fraudulent transactions**!

---

## 🚀 Key Features

- **End-to-End Modular Pipeline:** Clean separation of Data Ingestion, Preprocessing, Feature Engineering, Transformation, and Model Training.
- **Interactive Web UI:** A beautiful, responsive Gradio interface mounted directly onto a FastAPI backend for real-time predictions.
- **MLflow Tracking:** Automated tracking of metrics, parameters, and model artifacts via MLflow.
- **Dockerized & Deployment Ready:** A highly optimized `Dockerfile` (Render & Heroku compatible) with a strict `.dockerignore` for minimal image size.
- **Robust Testing:** Integrated unit tests using `pytest`.

---

## 💻 Getting Started

### 1. Installation
Ensure you have Python 3.12+ installed. Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Application
You can run the full API and the Web UI simultaneously with a single command:
```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Once the server starts, open your browser and navigate to:
👉 **http://127.0.0.1:8000/**

### 3. Retraining the Model
To re-run the entire machine learning pipeline on your data (generates new models in `artifacts/`):
```bash
python -m src.pipline.training_pipline
```

### 4. Running Tests
To execute the unit tests for data preprocessing and feature engineering:
```bash
PYTHONPATH=. pytest scripts/
```

---

## 🐳 Docker Deployment

This project includes a clean, production-ready Dockerfile that is fully compatible with PaaS providers like **Render** and **Heroku**. It automatically handles dynamic `$PORT` assignments.

```bash
# Build the image locally
docker build -t mfs-fraud-detection .

# Run the container
docker run -p 8000:8000 mfs-fraud-detection
```

---

## 📁 Project Structure

```text
├── artifacts/             # Trained models and preprocessor objects
├── Data/                  # Raw dataset (Ignored in Docker)
├── mlruns/                # MLflow tracking data
├── notebook/              # Exploratory Data Analysis & Optuna Tuning
├── scripts/               # Pytest unit tests for the pipeline
├── src/                   # Core application code
│   ├── app/               # FastAPI backend and Gradio UI integration
│   ├── component/         # ML pipeline steps (Ingestion, FE, Training)
│   └── pipline/           # Orchestrator script for the pipeline
├── Dockerfile             # Optimized multi-stage Docker build
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
