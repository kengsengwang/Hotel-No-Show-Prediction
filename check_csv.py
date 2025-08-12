import os
import pandas as pd

# Path to your CSV
csv_path = r"C:\Users\DELL\Documents\Hotel-No-Show-Prediction\data\hotel_no_show_cleaned.csv"

# Check if file exists
if os.path.isfile(csv_path):
    print(f"✅ Found CSV at: {csv_path}")
    # Load and preview the first 5 rows
    df = pd.read_csv(csv_path)
    print("\n--- First 5 rows ---")
    print(df.head())
else:
    print(f"❌ CSV not found at: {csv_path}")
