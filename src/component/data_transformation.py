import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        # Define categorical and numerical columns based on feature engineering output
        categorical_columns = ["type"]
        numerical_columns = [
            "step", "amount", "balance_change_org", 
            "balance_error_org", "balance_change_dest", "hour"
        ]

        # Use OneHotEncoder for categorical and StandardScaler for numerical
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_columns),
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_columns)
            ]
        )
        return preprocessor

    def initiate_data_transformation(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            print("Separating features and target...")
            target_column_name = "isFraud"
            
            X_train = train_df.drop(columns=[target_column_name])
            y_train = train_df[target_column_name].values
            
            X_test = test_df.drop(columns=[target_column_name])
            y_test = test_df[target_column_name].values

            print("Obtaining preprocessor object...")
            preprocessor = self.get_data_transformer_object()

            print("Applying preprocessing object on training and testing data...")
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            print(f"Saving preprocessor object to {self.transformation_config.preprocessor_obj_file_path}")
            os.makedirs(os.path.dirname(self.transformation_config.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessor, self.transformation_config.preprocessor_obj_file_path)

            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            raise e
