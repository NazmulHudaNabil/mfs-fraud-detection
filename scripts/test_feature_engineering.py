import pandas as pd
from src.component.feature_engineering import FeatureEngineering

def test_feature_calculations():
    fe = FeatureEngineering()
    
    # Mock dataframe with necessary columns for calculation
    data = {
        "step": [1, 25], 
        "type": ["TRANSFER", "CASH_OUT"],
        "amount": [100.0, 50.0],
        "oldbalanceOrg": [1000.0, 500.0],
        "newbalanceOrig": [900.0, 450.0],
        "oldbalanceDest": [200.0, 100.0],
        "newbalanceDest": [300.0, 150.0],
        "isFraud": [1, 0]
    }
    df = pd.DataFrame(data)

    processed_df = fe._apply_transformations(df)
    
    # 1. Check balance_change_org (oldbalanceOrg - newbalanceOrig)
    assert processed_df["balance_change_org"].iloc[0] == 100.0
    assert processed_df["balance_change_org"].iloc[1] == 50.0
    
    # 2. Check balance_error_org (oldbalanceOrg - amount - newbalanceOrig)
    assert processed_df["balance_error_org"].iloc[0] == 0.0
    
    # 3. Check balance_change_dest (newbalanceDest - oldbalanceDest)
    assert processed_df["balance_change_dest"].iloc[0] == 100.0
    
    # 4. Check hour calculation (step % 24)
    assert processed_df["hour"].iloc[0] == 1
    assert processed_df["hour"].iloc[1] == 1
    
    # 5. Check if correct columns are selected
    expected_columns = [
        "step", "type", "amount", "balance_change_org", 
        "balance_error_org", "balance_change_dest", "hour", "isFraud"
    ]
    assert list(processed_df.columns) == expected_columns
