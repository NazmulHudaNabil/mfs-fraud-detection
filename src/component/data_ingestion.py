import os
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train_data.csv")
    test_data_path: str = os.path.join("artifacts", "test_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            print("Reading raw dataset...")
            df = pd.read_csv(data_path)
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            print("Splitting dataset into train and test...")
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["isFraud"]
            )
            
            # Saving splits for record-keeping
            print(f"Saving train data to {self.ingestion_config.train_data_path}")
            train_df.to_csv(self.ingestion_config.train_data_path, index=False)
            
            print(f"Saving test data to {self.ingestion_config.test_data_path}")
            test_df.to_csv(self.ingestion_config.test_data_path, index=False)

            return train_df, test_df
        except Exception as e:
            raise e