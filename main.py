import pandas as pd
import os

def main():
    # Set the file path (adjust if needed)
    file_path = "./data/cleaned_noshow_data.csv"
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found. Please ensure the file is in the correct directory.")
        return
    
    # Load the data
    data = pd.read_csv(file_path)
    
    # Display basic information about the dataset
    print("✅ Data loaded successfully.")
    print("Data Shape:", data.shape)
    print("First 5 rows:")
    print(data.head())
    
    # Display basic statistics
    print("\nBasic Statistics:")
    print(data.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(data.isna().sum())

if __name__ == "__main__":
    main()
