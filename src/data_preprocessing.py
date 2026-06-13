import pandas as pd

def preprocess_data(df):
    """Clean and preprocess the dataset."""
    df = df.drop_duplicates()
    df = pd.get_dummies(df, drop_first=True)
    return df