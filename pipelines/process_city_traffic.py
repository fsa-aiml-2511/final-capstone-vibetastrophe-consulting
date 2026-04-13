import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipelines.data_pipeline import load_raw_data, clean_data, engineer_features, save_dataframe

city_df = load_raw_data("smart_city_csvs/city_traffic_accidents.csv")
cleansed_city_df = clean_data(city_df)
engineered_city_df = engineer_features(cleansed_city_df)
save_dataframe(engineered_city_df, "processed_city_traffic_accidents.csv")
