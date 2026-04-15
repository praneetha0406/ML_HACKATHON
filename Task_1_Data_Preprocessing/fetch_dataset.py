import os
import pandas as pd
from sklearn.datasets import fetch_openml

def download_electricity_dataset(output_path):
    print("Fetching the genuine 'Australian New South Wales (NSW) Electricity Market' dataset...")
    # OpenML ID 151 corresponds to the electricity dataset
    # This dataset contains 45,312 instances of half-hourly electricity demand and prices
    dataset = fetch_openml(data_id=151, as_frame=True, parser='auto')
    
    df = dataset.frame
    
    print(f"Dataset securely fetched! Shape: {df.shape}")
    print("First few rows:")
    print(df.head())
    
    # Save to the output path
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    output_csv = "raw_nsw_electricity.csv"
    download_electricity_dataset(output_csv)
