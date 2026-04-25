"""
Shared Data Pipeline
====================
Shared data loading and preprocessing functions used across all models.
Put your common data cleaning, feature engineering, and splitting logic here.

Usage from any model:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipelines.data_pipeline import load_raw_data, clean_data, engineer_features, split_data
"""
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_raw_data(filename):
    """Load a raw CSV file from data/raw/.

    Args:
        filename: Name of the CSV file (e.g., "city_traffic_accidents.csv")

    Returns:
        pandas DataFrame

    Example:
        df = load_raw_data("city_traffic_accidents.csv")
        df = load_raw_data("urbanpulse_311_complaints.csv")
    """
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Make sure you've downloaded the data to data/raw/"
        )
    return pd.read_csv(filepath)


def clean_data(df):
    """Apply common data cleaning steps.

    Things to handle:
    - Missing value encoding (e.g., '?' -> NaN)
    - Data type conversions
    - Remove duplicates
    - Drop irrelevant columns

    Returns:
        Cleaned DataFrame
    """
    # Encode common missing-value sentinels as NaN
    df = df.replace({'?': pd.NA, 'N/A': pd.NA, 'NA': pd.NA, '': pd.NA})

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Drop columns where every value is null
    df = df.dropna(axis=1, how='all')

    # Strip leading/trailing whitespace from string columns
    str_cols = df.select_dtypes(include='object').columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # Drop the column called 'Description'
    # if 'Description' in df.columns:
    #     df = df.drop(columns=['Description'])

    # Drop the following columns: Street,City,County,State,Zipcode,Country
    for col in ['ID', 'Source', 'Description', 'Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Airport_Code']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # If precipitation value is empty or blank, fill with 0.0
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0.0)

    df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(0.0)

    df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(0.0)
    humidity_mean = df['Humidity(%)'].mean()
    df['Humidity(%)'] = df['Humidity(%)'].fillna(humidity_mean)

    df['End_Lat'] = df['End_Lat'].fillna(df['Start_Lat'])
    df['End_Lng'] = df['End_Lng'].fillna(df['Start_Lng'])

    # Convert the following TRUE /FALSE columns to binary 1/0: Amenity,Bump,Crossing,Give_Way,Junction,No_Exit,Railway,Roundabout,Station,Stop,Traffic_Calming,Traffic_Signal,Turning_Loop
    binary_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'TRUE': 1, 'FALSE': 0}).fillna(0).astype(int)

    # Convert the following columns that have values either set to DAY / NIGHT to binary 1/0: Sunrise_Sunset, Civil_Twilight, Nautical_Twilight, Astronomical_Twilight
    day_night_cols = ['Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
    for col in day_night_cols:
        if col in df.columns:
            df[col] = df[col].map({'DAY': 1, 'NIGHT': 0}).fillna(0).astype(int)
    
    # Convert the timezone from 'US/Eastern' 'US/Central', 'US/Mountain', 'US/Pacific' to 0,1,2,3 respectively 
    timezone_map = {'US/Eastern': 0, 'US/Central': 1, 'US/Mountain': 2, 'US/Pacific': 3}
    if 'Timezone' in df.columns:
        df['Timezone'] = df['Timezone'].map(timezone_map).fillna(0).astype(int)

    return df


def engineer_features(df):
    """Create new features from existing columns.

    Examples:
    - Parse datetime columns -> hour, day_of_week, month
    - Create binary flags from categorical data
    - Bin continuous variables into categories
    - Interaction features

    Returns:
        DataFrame with new feature columns
    """
    # Parse any object columns that look like datetimes
    for col in df.select_dtypes(include='object').columns:
        if any(kw in col.lower() for kw in ('time', 'date', 'dt')):
            try:
                df[col] = pd.to_datetime(df[col])
            except (ValueError, TypeError):
                df[col] = df[col]  # If parsing fails, set to NaN

    # Extract temporal features from all datetime columns
    for col in df.select_dtypes(include='datetime').columns:
        prefix = col.lower().replace(' ', '_')
        df[f'{prefix}_hour'] = df[col].dt.hour
        df[f'{prefix}_day_of_week'] = df[col].dt.dayofweek
        df[f'{prefix}_month'] = df[col].dt.month
        df[f'{prefix}_is_weekend'] = (df[col].dt.dayofweek >= 5).astype(int)

    df['wind_dir_deg'] = df['Wind_Direction'].map({
        'N': 0,
        'North': 0,
        'NE': 45,
        'NNE': 22.5,
        'NNW': 337.5,
        'E': 90,
        'ENE': 67.5,
        'East': 90,
        'ESE': 112.5,
        'SE': 135,
        'S': 180,
        'South': 180,
        'SW': 225,
        'SSW': 202.5,
        'SSE': 157.5,
        'W': 270,
        'West': 270,
        'WNW': 292.5,
        'WSW': 247.5,
        'NW': 315
    })

    mean_degrees = df['wind_dir_deg'].mean()
    df['wind_dir_deg'].fillna(mean_degrees, inplace=True)

    df['weather_cond_num'] = df['Weather_Condition'].map({
        'Mostly Cloudy': 0,
        'Light Rain with Thunder': 3,
        'Fair': 0,
        'Clear': 0,
        'Overcast': 0,
        'Cloudy': 0,
        'Scattered Clouds': 0,
        'Light Snow': 1,
        'Partly Cloudy': 0,
        'T-Storm': 3,
        'Heavy Snow': 2,
        'Light Rain': 1,
        'Smoke': 1,
        'Fog': 2,
        'Fair / Windy': 0,
        'Light Drizzle': 1,
        'Drizzle': 1,
        'Haze': 1,
        'Rain': 2,
        'Heavy Rain': 3,
        'T-Storm / Windy': 3,
        'Thunder in the Vicinity': 1,
        'Snow': 2,
        'Heavy Drizzle': 2,
        'Light Thunderstorms and Rain': 3,
        'Cloudy / Windy': 0,
        'Wintry Mix': 2,
        'N/A Precipitation': 0,
        'Thunder': 1,
        'Light Snow / Windy': 1,
        'Heavy Snow / Windy': 2,
        'Light Rain / Windy': 1,
        'Patches of Fog': 2,
        'Heavy Rain / Windy': 3,
        'Heavy Thunderstorms and Rain': 3,
        'Mostly Cloudy / Windy': 0,
        'Heavy T-Storm': 3,
        'Mist': 2,
        'Partly Cloudy / Windy': 0,
        'Shallow Fog': 2,
        'Snow / Windy': 2,
        'Blowing Snow': 2,
        'Thunderstorm': 3,
        'Haze / Windy': 1,
        'Freezing Rain': 2,
        'Blowing Snow / Windy': 2,
        'Rain / Windy': 2,
        'Showers in the Vicinity': 0,
        'Light Freezing Rain': 1,
        'Snow and Sleet': 2,
        'Widespread Dust': 1,
        'Snow and Sleet / Windy': 2,
        'Thunderstorms and Rain': 3,
        'Light Freezing Drizzle': 1,
        'Fog / Windy': 2,
        'Heavy T-Storm / Windy': 3,
        'Light Freezing Fog': 1,
        'Ice Pellets': 2,
        'Tornado': 3,
        'Light Thunderstorms and Snow': 3,
        'Light Rain Shower': 1,
        'Drizzle and Fog': 2,
        'Blowing Dust / Windy': 1,
        'Heavy Sleet': 2,
        'Blowing Dust': 1,
        'Light Drizzle / Windy': 1,
        'Light Freezing Rain / Windy': 1,
        'Wintry Mix / Windy': 2,
        'Thunder / Windy': 1,
        'Hail': 2,
        'Light Snow and Sleet': 2,
        'Light Rain Showers': 1,
        'Smoke / Windy': 1,
        'Light Ice Pellets': 1,
        'Snow and Thunder': 3,
        'Sleet': 2,
        'Small Hail': 2,
        'Sleet / Windy': 2,
        'Light Snow with Thunder': 3,
        'Widespread Dust / Windy': 1,
        'Duststorm': 1,
        'Sand / Dust Whirlwinds': 1,
        'Light Rain Shower / Windy': 1,
        'Light Snow and Sleet / Windy': 2,
        'Rain Showers': 2,
        'Light Snow Shower': 1,
        'Freezing Drizzle': 2,
        'Light Snow Showers': 1,
        'Thunder / Wintry Mix': 2,
        'Rain Shower': 2,
        'Squalls': 2,
        'Drifting Snow / Windy': 2,
        'Mist / Windy': 2,
        'Sand / Dust Whirls Nearby': 1,
        'Light Snow Grains': 1,
        'Sand': 1,
        'Light Sleet': 1,
        'Partial Fog': 2,
        'Drizzle / Windy': 1,
        'Patches of Fog / Windy': 2,
        'Thunder / Wintry Mix / Windy': 2,
        'Light Sleet / Windy': 1,
        'Squalls / Windy': 2,
        'Heavy Freezing Drizzle': 2,
        'Snow Grains': 2,
        'Shallow Fog / Windy': 1})
    
    df = df.drop(columns=['Weather_Condition', 'Wind_Direction'])

    # Create a new feature called 'accident_dir' that is the degrees from the start point to the end point using the lat and lng values. Use the arctangent of the difference in latitudes and longitudes to calculate the angle in degrees. Handle cases where the start and end points are the same to avoid division by zero.
    # The accident direction should be in degrees where 0 degrees is north, 90 degrees is east, 180 degrees is south, and 270 degrees is west. You can use the numpy arctan2 function to calculate the angle in radians and then convert it to degrees. 
    df['accident_dir'] = np.degrees(np.arctan2(df['End_Lng'] - df['Start_Lng'], df['End_Lat'] - df['Start_Lat']))
    df['accident_dir'] = (df['accident_dir'] + 360) % 360

    return df


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and validation sets.

    IMPORTANT: Use stratify=y for imbalanced classification tasks.

    Args:
        X: Feature matrix
        y: Target variable
        test_size: Proportion for validation (default 0.2)
        random_state: For reproducibility

    Returns:
        X_train, X_val, y_train, y_val
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def save_processed_data(df, filename):
    """Save processed data to data/processed/.

    Args:
        df: Processed DataFrame
        filename: Output filename (e.g., "encounters_processed.csv")
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")


def load_processed_data(filename):
    """Load previously processed data from data/processed/.

    Args:
        filename: Name of the processed CSV file

    Returns:
        pandas DataFrame
    """
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run the data pipeline first to generate processed data."
        )
    return pd.read_csv(filepath)

def save_dataframe(df, filename):
    """Save a DataFrame to the processed data directory."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved DataFrame to {output_path}")
