# utils.py
# Utility functions for the Student Success Analytics Dashboard

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from config import *


def normalize_column_name(col_name: str) -> str:
    """
    Normalize column name to database format.
    Example: "Marital status" -> "marital_status"
    """
    return col_name.strip().lower().replace("'", "").replace(" ", "_").replace("(", "").replace(")", "")


def get_risk_level(probability: float) -> str:
    """
    Determine risk level based on dropout probability.
    
    Args:
        probability: Dropout probability (0.0 to 1.0)
        
    Returns:
        Risk level: 'None', 'Mild', 'Moderate', or 'Severe'
    """
    if probability < RISK_THRESHOLDS['mild'][0]:
        return 'None'
    elif RISK_THRESHOLDS['mild'][0] <= probability < RISK_THRESHOLDS['mild'][1]:
        return 'Mild'
    elif RISK_THRESHOLDS['moderate'][0] <= probability < RISK_THRESHOLDS['moderate'][1]:
        return 'Moderate'
    elif probability >= RISK_THRESHOLDS['severe'][0]:
        return 'Severe'
    else:
        return 'None'


def get_risk_color(risk_level: str) -> str:
    """Get color code for risk level."""
    return RISK_COLORS.get(risk_level, RISK_COLORS['None'])


def validate_csv_features(df: pd.DataFrame, required_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate if CSV contains all required features.
    Case-insensitive and whitespace-tolerant matching.
    
    Args:
        df: DataFrame to validate
        required_features: List of required feature names
        
    Returns:
        Tuple of (is_valid, missing_features)
    """
    # Normalize column names for comparison (strip and lowercase)
    df_columns_normalized = {col.strip().lower(): col for col in df.columns}
    required_normalized = {feat.strip().lower(): feat for feat in required_features}
    
    # Find missing features
    missing = []
    for req_norm, req_orig in required_normalized.items():
        if req_norm not in df_columns_normalized:
            missing.append(req_orig)
    
    return len(missing) == 0, missing


def map_csv_columns_to_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CSV column names to database column names.
    Handles flexible column ordering and name variations.
    
    Args:
        df: Input DataFrame with CSV column names
        
    Returns:
        DataFrame with database column names
    """
    # Create a mapping of normalized names
    rename_dict = {}
    
    for col in df.columns:
        # Try exact match first
        if col in FEATURE_NAME_MAPPING:
            rename_dict[col] = FEATURE_NAME_MAPPING[col]
        else:
            # Try normalized match
            normalized = normalize_column_name(col)
            # Check if this normalized name exists in our mapping values
            for display_name, db_name in FEATURE_NAME_MAPPING.items():
                if normalize_column_name(display_name) == normalized:
                    rename_dict[col] = db_name
                    break
    
    # Rename columns
    df_renamed = df.rename(columns=rename_dict)
    
    return df_renamed


def load_meta_json() -> Optional[Dict[str, Any]]:
    """
    Load model metadata from meta.json.
    
    Returns:
        Dictionary with metadata or None if file doesn't exist
    """
    if os.path.exists(META_JSON_PATH):
        try:
            with open(META_JSON_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading meta.json: {e}")
            return None
    return None


def save_meta_json(metadata: Dict[str, Any]) -> bool:
    """
    Save model metadata to meta.json.
    
    Args:
        metadata: Dictionary with metadata to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(META_JSON_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving meta.json: {e}")
        return False


def generate_unique_username(base_username: str, existing_usernames: List[str]) -> str:
    """
    Generate a unique username by appending numbers if needed.
    
    Args:
        base_username: Base username (e.g., 'john')
        existing_usernames: List of existing usernames
        
    Returns:
        Unique username (e.g., 'john', 'john1', 'john2')
    """
    username = base_username.lower()
    
    if username not in existing_usernames:
        return username
    
    counter = 1
    while f"{username}{counter}" in existing_usernames:
        counter += 1
    
    return f"{username}{counter}"


def extract_first_name(full_name: str) -> str:
    """
    Extract first word from full name for username generation.
    
    Args:
        full_name: Full name (e.g., "John Smith")
        
    Returns:
        First word in lowercase (e.g., "john")
    """
    return full_name.strip().split()[0].lower()


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.2f}%"


def is_categorical(series: pd.Series, threshold: int = 10) -> bool:
    """
    Determine if a series is categorical based on number of unique values.
    
    Args:
        series: Pandas Series to check
        threshold: Maximum unique values to consider categorical
        
    Returns:
        True if categorical, False if numerical
    """
    return series.nunique() <= threshold


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to maximum length with ellipsis."""
    return text if len(text) <= max_length else text[:max_length-3] + "..."


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Check if file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def parse_csv_with_separator(file_path: str, separator: str = ';') -> pd.DataFrame:
    """
    Parse CSV file with specified separator, fallback to comma if fails.
    Automatically strips whitespace from column names.
    
    Args:
        file_path: Path to CSV file
        separator: Primary separator to try (default: semicolon)
        
    Returns:
        Parsed DataFrame with cleaned column names
    """
    try:
        df = pd.read_csv(file_path, sep=separator)
        # Check if parsing was successful (more than 1 column)
        if len(df.columns) > 1:
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            return df
    except:
        pass
    
    # Fallback to comma
    try:
        df = pd.read_csv(file_path, sep=',')
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {e}")


def convert_to_numeric_safe(value: Any) -> float:
    """Safely convert value to numeric, return NaN if fails."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by handling missing values and infinite values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Replace infinite values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median for numerical columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
    
    return df_clean


def get_feature_type(series: pd.Series, feature_name: str = None) -> str:
    """
    Determine feature type for visualization.
    
    Args:
        series: Data series to analyze
        feature_name: Optional feature name to check for label mappings
    
    Returns:
        'categorical', 'discrete', or 'continuous'
    """
    # Check if feature has text labels defined - if so, treat as categorical
    if feature_name and feature_name in FEATURE_VALUE_LABELS:
        return 'categorical'
    
    unique_count = series.nunique()
    
    if unique_count <= 10:
        return 'categorical'
    elif unique_count <= 30:
        return 'discrete'
    else:
        return 'continuous'


def create_student_dict_from_row(row: pd.Series) -> Dict[str, Any]:
    """
    Convert DataFrame row to student dictionary.
    
    Args:
        row: Pandas Series representing a student record
        
    Returns:
        Dictionary with student data
    """
    student_dict = {}
    
    for db_col, display_name in DB_TO_DISPLAY_MAPPING.items():
        if db_col in row.index:
            student_dict[display_name] = row[db_col]
    
    return student_dict


def format_datetime(dt_str: str) -> str:
    """Format datetime string for display."""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def check_model_exists() -> bool:
    """Check if trained model exists."""
    return (
        os.path.exists(MODEL_PATH) and
        os.path.exists(SCALER_PATH) and
        os.path.exists(FEATURE_COLUMNS_PATH) and
        os.path.exists(META_JSON_PATH)
    )


def check_training_data_exists() -> bool:
    """Check if training data file exists."""
    return os.path.exists(TRAINING_DATA_PATH)


def get_display_name(db_column_name: str) -> str:
    """Convert database column name to display name."""
    return DB_TO_DISPLAY_MAPPING.get(db_column_name, db_column_name)


def get_db_column_name(display_name: str) -> str:
    """Convert display name to database column name."""
    return FEATURE_NAME_MAPPING.get(display_name, normalize_column_name(display_name))
