import pandas as pd
import numpy as np
from src.component.data_preprocessing import DataPreprocessing

def test_remove_duplicates_and_fill_na():
    preprocessor = DataPreprocessing()
    
    # Create a mock dataframe with duplicates and missing values
    data = {
        "amount": [100.0, 100.0, np.nan, 200.0],
        "type": ["CASH_OUT", "CASH_OUT", "TRANSFER", np.nan],
        "isFraud": [0, 0, 1, 0]
    }
    df = pd.DataFrame(data)

    processed_df = preprocessor._preprocess_data(df, "test")
    
    # 1. Check if duplicates are removed
    assert len(processed_df) == 3
    
    # 2. Check if NaN in numerical is filled with 0
    assert processed_df.isnull().sum().sum() == 0
    
    # Check specific fill values
    transfer_row = processed_df[processed_df["type"] == "TRANSFER"].iloc[0]
    assert np.isnan(df.iloc[2]["amount"]) # Confirm it was NaN originally
    assert transfer_row["amount"] == 0.0
    
    # The row with amount 200.0 had NaN type -> should be "Unknown"
    nan_type_row = processed_df[processed_df["amount"] == 200.0].iloc[0]
    assert nan_type_row["type"] == "Unknown"
