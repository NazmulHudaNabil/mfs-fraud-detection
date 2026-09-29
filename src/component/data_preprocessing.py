import pandas as pd

class DataPreprocessing:
    def __init__(self):
        pass

    def _preprocess_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        print(f"Preprocessing {dataset_name} data...")
        df = df.copy()
        
        # 1. Remove duplicates
        initial_shape = df.shape
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape[0]:
            print(f"  -> Removed {initial_shape[0] - df.shape[0]} duplicate rows.")
            
        # 2. Handle missing values (simple approach)
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"  -> Found {missing_count} missing values. Filling them...")
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].fillna("Unknown")
                else:
                    df[col] = df[col].fillna(0)
        else:
            print("  -> No missing values found.")
            
        return df

    def initiate_data_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            train_df_processed = self._preprocess_data(train_df, "training")
            test_df_processed = self._preprocess_data(test_df, "testing")
            
            return train_df_processed, test_df_processed
        except Exception as e:
            raise e
