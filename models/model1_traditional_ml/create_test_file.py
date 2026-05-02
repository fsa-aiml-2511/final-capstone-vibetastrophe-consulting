# Create a test file from the `/data/processed/processed_city_traffic_accidents.csv` file that will be saved in the `/test_data/` directory. This will be chosen from random rows in the original processed file, and will be a small subset of records.

import os
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEST_DATA_FILE = TEST_DATA_DIR / "City_traffic_Test.csv"

from pipelines.data_pipeline import load_raw_data, clean_data, engineer_features, save_dataframe

# city_df = load_raw_data("smart_city_csvs/city_traffic_accidents.csv")


# Load the processed data
processed_file_path = PROJECT_ROOT / 'data' / 'raw' / 'smart_city_csvs' / 'city_traffic_accidents.csv'
df = pd.read_csv(processed_file_path)
# Set the number of random rows to select for the test file
num_rows = 100  # You can adjust this number based on your needs
# Randomly select rows from the DataFrame
test_df = df.sample(n=num_rows, random_state=42)  # Setting random_state for reproducibility
# Save the test DataFrame to a new CSV file in the /test_data/ directory
cleansed_city_df = clean_data(test_df, keep_id=True)
engineered_city_df = engineer_features(cleansed_city_df)

engineered_city_df.to_csv(TEST_DATA_FILE, index=False)
print(f'Test file created at: {TEST_DATA_FILE} with {num_rows} random rows from the original processed file.')
