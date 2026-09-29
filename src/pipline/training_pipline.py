import sys
from src.component.data_ingestion import DataIngestion
from src.component.data_preprocessing import DataPreprocessing
from src.component.feature_engineering import FeatureEngineering
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

def run_training_pipeline(data_path: str = "Data/frud_data.csv"):
    try:
        print(">>> Starting Training Pipeline <<<")
        
        # 1. Data Ingestion
        ingestion = DataIngestion()
        train_df, test_df = ingestion.initiate_data_ingestion(data_path)
        print("--- Data Ingestion Completed ---\n")
        
        # 1.5. Data Preprocessing
        preprocessor = DataPreprocessing()
        train_df, test_df = preprocessor.initiate_data_preprocessing(train_df, test_df)
        print("--- Data Preprocessing Completed ---\n")
        
        # 2. Feature Engineering
        fe = FeatureEngineering()
        train_df_fe, test_df_fe = fe.initiate_feature_engineering(train_df, test_df)
        print("--- Feature Engineering Completed ---\n")
        
        # 3. Data Transformation
        transformer = DataTransformation()
        X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(train_df_fe, test_df_fe)
        print("--- Data Transformation Completed ---\n")
        
        # 4. Model Training
        trainer = ModelTrainer()
        metrics = trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        print("--- Model Training Completed ---\n")
        
        print(">>> Training Pipeline Finished Successfully <<<")
        return metrics

    except Exception as e:
        print(f"Error in training pipeline: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_training_pipeline()
