# feature_ranking.py
# Random Forest Classifier for Feature Importance Ranking

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from config import *
from utils import parse_csv_with_separator, load_meta_json


def calculate_feature_importance(training_data_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Calculate feature importance using Random Forest Classifier.
    Uses only the features that were selected for training.
    
    Args:
        training_data_path: Path to training CSV file
        
    Returns:
        Tuple of (success, message, importance_df)
        importance_df has columns: ['feature', 'importance']
    """
    try:
        # Load training data
        print(f"Loading training data from: {training_data_path}")
        df = parse_csv_with_separator(training_data_path, CSV_SEPARATOR)
        
        # Find target column
        target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
        if not target_col:
            return False, "Target column not found in training data", None
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)
        
        # Get trained features from meta.json
        meta = load_meta_json()
        if meta is None or 'trained_features' not in meta:
            print("⚠️ Warning: meta.json not found. Using all available features.")
            trained_features = None
        else:
            trained_features = meta['trained_features']
            print(f"Using {len(trained_features)} trained features from meta.json")
        
        # Apply encoding (same as training)
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median
        for col in X_encoded.columns:
            if X_encoded[col].isna().any():
                median_val = X_encoded[col].median()
                X_encoded[col].fillna(median_val, inplace=True)
        
        # If we have trained features, filter to use only those
        if trained_features is not None:
            # Ensure all trained features exist
            available_features = []
            for feat in trained_features:
                if feat in X_encoded.columns:
                    available_features.append(feat)
            
            if len(available_features) == 0:
                return False, "No matching features found", None
            
            X_encoded = X_encoded[available_features]
            print(f"Filtered to {len(available_features)} features")
        
        # Train Random Forest
        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=MODEL_CONFIG['random_state'],
            max_depth=10,
            n_jobs=-1
        )
        
        rf.fit(X_encoded, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': X_encoded.columns,
            'importance': importances
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print(f"✅ Feature importance calculated for {len(importance_df)} features")
        print(f"   Top 5 features:")
        for idx, row in importance_df.head(5).iterrows():
            print(f"     {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        return True, "Feature importance calculated successfully", importance_df
        
    except Exception as e:
        return False, f"Error calculating feature importance: {e}", None


def get_top_features(n: int = 10) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Get top N most important features.
    
    Args:
        n: Number of top features to return
        
    Returns:
        Tuple of (success, message, top_features_df)
    """
    success, message, importance_df = calculate_feature_importance(TRAINING_DATA_PATH)
    
    if not success:
        return False, message, None
    
    top_features = importance_df.head(n)
    
    return True, f"Top {n} features retrieved", top_features


def get_feature_importance_percentage(training_data_path: str = TRAINING_DATA_PATH) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """
    Calculate feature importance as percentages.
    
    Returns:
        Tuple of (success, message, importance_df)
        importance_df has columns: ['feature', 'importance', 'percentage']
    """
    success, message, importance_df = calculate_feature_importance(training_data_path)
    
    if not success:
        return False, message, None
    
    # Calculate percentages
    total_importance = importance_df['importance'].sum()
    importance_df['percentage'] = (importance_df['importance'] / total_importance * 100).round(2)
    
    return True, "Feature importance with percentages calculated", importance_df


def explain_top_feature(feature_name: str) -> str:
    """
    Provide a human-readable explanation for a feature name.
    
    Args:
        feature_name: Encoded feature name (e.g., 'age_at_enrollment', 'course_171')
        
    Returns:
        Human-readable explanation
    """
    # Map common feature patterns to explanations
    explanations = {
        'age_at_enrollment': "Student's age when enrolling",
        'admission_grade': "Grade obtained in admission exam",
        'curricular_units_1st_sem_grade': "Average grade in first semester",
        'curricular_units_2nd_sem_grade': "Average grade in second semester",
        'curricular_units_1st_sem_approved': "Number of units approved in first semester",
        'curricular_units_2nd_sem_approved': "Number of units approved in second semester",
        'curricular_units_1st_sem_enrolled': "Number of units enrolled in first semester",
        'curricular_units_2nd_sem_enrolled': "Number of units enrolled in second semester",
        'scholarship_holder': "Whether student has a scholarship",
        'tuition_fees_up_to_date': "Tuition fee payment status",
        'debtor': "Whether student has debt",
        'unemployment_rate': "Unemployment rate (economic indicator)",
        'inflation_rate': "Inflation rate (economic indicator)",
        'gdp': "GDP (economic indicator)",
    }
    
    # Check for exact match
    if feature_name in explanations:
        return explanations[feature_name]
    
    # Check for partial matches
    for key, explanation in explanations.items():
        if key in feature_name:
            return f"{explanation} (category: {feature_name})"
    
    # Default: return cleaned feature name
    return feature_name.replace('_', ' ').title()


if __name__ == '__main__':
    # Test mode
    print("=== Feature Importance Ranking Test ===")
    
    if not os.path.exists(TRAINING_DATA_PATH):
        print(f"❌ Training data not found: {TRAINING_DATA_PATH}")
    else:
        success, message, df = calculate_feature_importance(TRAINING_DATA_PATH)
        
        if success:
            print(f"\n✅ {message}")
            print(f"\nTop 10 Features:")
            print(df.head(10).to_string(index=False))
        else:
            print(f"\n❌ {message}")
