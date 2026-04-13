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
