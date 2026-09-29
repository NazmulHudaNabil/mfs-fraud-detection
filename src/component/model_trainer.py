import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred, pred_proba):
        # Calculate evaluation metrics
        acc = accuracy_score(actual, pred)
        prec = precision_score(actual, pred, zero_division=0)
        rec = recall_score(actual, pred, zero_division=0)
        f1 = f1_score(actual, pred, zero_division=0)
        roc_auc = roc_auc_score(actual, pred_proba)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            print("Configuring MLflow...")
            mlflow.set_experiment("MFS_Fraud_Detection_Experiment")

            with mlflow.start_run():
                print("Training RandomForestClassifier...")
                rf_params = {
                    "n_estimators": 125, 
                    "max_depth": 24, 
                    "min_samples_split": 10, 
                    "min_samples_leaf": 2, 
                    "max_features": "log2", 
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1
                }
                
                model = RandomForestClassifier(**rf_params)
                model.fit(X_train, y_train)

                print("Evaluating model...")
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                metrics = self.eval_metrics(y_test, y_pred, y_pred_proba)
                
                print("Logging metrics and params to MLflow...")
                mlflow.log_params(rf_params)
                mlflow.log_metrics(metrics)
                
                print("Logging model to MLflow...")
                mlflow.sklearn.log_model(
                    model, 
                    "random_forest_model", 
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE
                )

                print(f"Saving model locally to {self.trainer_config.trained_model_file_path}...")
                os.makedirs(os.path.dirname(self.trainer_config.trained_model_file_path), exist_ok=True)
                joblib.dump(model, self.trainer_config.trained_model_file_path)

                print("\n=== Model Metrics ===")
                for name, value in metrics.items():
                    print(f"{name.capitalize()}: {value:.4f}")
                print("=====================\n")

                return metrics
        except Exception as e:
            raise e
