import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.data_pipeline import load_raw_data, urban_data_clean, engineer_urban_features, save_dataframe

city_df = load_raw_data("smart_city_csvs/urbanpulse_311_complaints.csv")
cleansed_urban_df = urban_data_clean(city_df)
engineered_urban_df = engineer_urban_features(cleansed_urban_df)
save_dataframe(engineered_urban_df, "processed_urbanpulse_311_complaints.csv")
