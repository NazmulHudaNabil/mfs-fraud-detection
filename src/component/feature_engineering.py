import pandas as pd

class FeatureEngineering:
    def __init__(self):
        pass

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Calculate new features from the notebook
        df["balance_change_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["balance_error_org"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
        df["balance_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["hour"] = df["step"] % 24
        
        # Select required columns
        columns_to_keep = [
            "step", "type", "amount", "balance_change_org", 
            "balance_error_org", "balance_change_dest", "hour", "isFraud"
        ]
        return df[columns_to_keep]

    def initiate_feature_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            print("Applying feature engineering on train data...")
            train_df_fe = self._apply_transformations(train_df)
            
            print("Applying feature engineering on test data...")
            test_df_fe = self._apply_transformations(test_df)
            
            return train_df_fe, test_df_fe
        except Exception as e:
            raise e