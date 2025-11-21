# prediction.py
# Prediction logic with feature alignment

import pandas as pd
import numpy as np
import os
import sys
import joblib
from typing import Tuple, Optional, Dict, Any

try:
    import tensorflow as tf
except ImportError:
    print("❌ TensorFlow is not installed")
    tf = None

from config import *
from utils import load_meta_json, get_risk_level
import database as db


def load_model_artifacts() -> Tuple[Optional[Any], Optional[Any], Optional[list]]:
    """
    Load trained model, scaler, and feature columns.
    
    Returns:
        Tuple of (model, scaler, feature_names) or (None, None, None) if error
    """
    try:
        # Check if all files exist
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            return None, None, None
        
        if not os.path.exists(SCALER_PATH):
            print(f"❌ Scaler not found at: {SCALER_PATH}")
            return None, None, None
        
        if not os.path.exists(FEATURE_COLUMNS_PATH):
            print(f"❌ Feature columns not found at: {FEATURE_COLUMNS_PATH}")
            return None, None, None
        
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load scaler
        scaler = joblib.load(SCALER_PATH)
        
        # Load feature columns
        feature_names = joblib.load(FEATURE_COLUMNS_PATH)
        
        print(f"✅ Model artifacts loaded successfully")
        print(f"   Features: {len(feature_names)}")
        
        return model, scaler, feature_names
        
    except Exception as e:
        print(f"❌ Error loading model artifacts: {e}")
        return None, None, None


def align_features(student_features: Dict[str, Any], trained_features: list) -> pd.DataFrame:
    """
    Align student features with the model's expected feature vector.
    """
    # Sets for detection
    db_cols_set = set(FEATURE_NAME_MAPPING.values())
    display_cols_set = set(FEATURE_NAME_MAPPING.keys())
    trained_set = set(trained_features)

    # Build a single-row DataFrame from the student's raw features (DB column names)
    student_df = pd.DataFrame([student_features])

    # Case 1: numeric model trained on DB column names
    if (len(trained_features) == len(db_cols_set)) and (trained_set == db_cols_set):
        aligned_df = student_df.reindex(columns=trained_features)
        aligned_df = aligned_df.apply(pd.to_numeric, errors='coerce')
        aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        return aligned_df

    # Case 2: numeric model trained on display names
    if (len(trained_features) == len(display_cols_set)) and (trained_set == display_cols_set):
        student_df_disp = student_df.rename(columns=DB_TO_DISPLAY_MAPPING)
        aligned_df = student_df_disp.reindex(columns=trained_features)
        aligned_df = aligned_df.apply(pd.to_numeric, errors='coerce')
        aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        return aligned_df

    # Case 3: one-hot or mixed naming - choose the encoding with max overlap
    student_encoded_db = pd.get_dummies(student_df, drop_first=True)
    student_encoded_disp = pd.get_dummies(student_df.rename(columns=DB_TO_DISPLAY_MAPPING), drop_first=True)

    overlap_db = sum(1 for c in trained_features if c in student_encoded_db.columns)
    overlap_disp = sum(1 for c in trained_features if c in student_encoded_disp.columns)

    student_encoded = student_encoded_disp if overlap_disp > overlap_db else student_encoded_db

    aligned = {}
    for col in trained_features:
        aligned[col] = student_encoded[col].values[0] if col in student_encoded.columns else 0

    aligned_df = pd.DataFrame([aligned], columns=trained_features)
    aligned_df = aligned_df.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    return aligned_df


def predict_single_student(student_id: int) -> Tuple[bool, str, Optional[float], Optional[str]]:
    """
    Predict dropout probability for a single student.
    
    Args:
        student_id: Student ID
        
    Returns:
        Tuple of (success, message, probability, risk_level)
    """
    try:
        # Load model artifacts
        model, scaler, trained_features = load_model_artifacts()
        
        if model is None or scaler is None or trained_features is None:
            return False, "Model artifacts not loaded", None, None
        
        # Get student features from database
        student_features = db.get_student_features_for_prediction(student_id)
        
        if student_features is None:
            return False, f"Student {student_id} not found", None, None
        
        # Align features with trained model
        aligned_features = align_features(student_features, trained_features)
        
        # Scale features
        scaled_features = scaler.transform(aligned_features)
        
        # Predict
        raw_prob = float(model.predict(scaled_features, verbose=0)[0][0])
        
        dropout_probability = 1.0 - raw_prob
        
        # Determine risk level
        risk_level = get_risk_level(dropout_probability)
        
        # Update database
        db.update_student_prediction(student_id, dropout_probability, risk_level)
        
        return True, "Prediction successful", dropout_probability, risk_level
        
    except Exception as e:
        return False, f"Error during prediction: {e}", None, None


def predict_all_students() -> Tuple[bool, str, int]:
    """
    Predict dropout probability for all students in database.
    
    Returns:
        Tuple of (success, message, count_predicted)
    """
    try:
        # Load model artifacts
        model, scaler, trained_features = load_model_artifacts()
        
        if model is None or scaler is None or trained_features is None:
            return False, "Model not trained yet. Please train the model first.", 0
        
        # Get all students
        students_df = db.get_all_students()
        
        if len(students_df) == 0:
            return False, "No students in database. Please add students first.", 0
        
        print(f"🎯 Predicting for {len(students_df)} students...")
        print(f"📊 Using model with {len(trained_features)} features")
        sys.stdout.flush()
        
        count_success = 0
        count_failed = 0
        
        for idx, row in students_df.iterrows():
            student_id = row['studentID']
            
            # Extract features (36 columns)
            student_features = {}
            for db_col in FEATURE_NAME_MAPPING.values():
                if db_col in row.index:
                    student_features[db_col] = row[db_col]
            
            try:
                # Align features
                aligned_features = align_features(student_features, trained_features)
                
                # Scale features
                scaled_features = scaler.transform(aligned_features)
                
                # Predict
                raw_prob = float(model.predict(scaled_features, verbose=0)[0][0])
                
                dropout_probability = 1.0 - raw_prob
                
                # Determine risk level
                risk_level = get_risk_level(dropout_probability)
                
                # Update database
                success, msg = db.update_student_prediction(student_id, dropout_probability, risk_level)
                
                if success:
                    count_success += 1
                    if count_success % 10 == 0:  # Progress every 10 students
                        print(f"  ✅ Predicted {count_success}/{len(students_df)} students...")
                        sys.stdout.flush()
                else:
                    count_failed += 1
                    print(f"  ❌ Failed to update student {student_id}: {msg}")
                    sys.stdout.flush()
                    
            except Exception as e:
                count_failed += 1
                print(f"  ❌ Error predicting student {student_id}: {e}")
                import traceback
                traceback.print_exc()
                sys.stdout.flush()
        
        print(f"✅ Prediction complete: {count_success} successful, {count_failed} failed")
        sys.stdout.flush()
        
        if count_success == 0:
            return False, f"Failed to predict any students. {count_failed} errors occurred. Check terminal for details.", 0
        elif count_failed > 0:
            return True, f"Predicted {count_success} students successfully ({count_failed} failed)", count_success
        else:
            return True, f"Predicted all {count_success} students successfully", count_success
        
    except Exception as e:
        return False, f"Error during batch prediction: {e}", 0


def get_prediction_summary() -> Dict[str, Any]:
    """
    Get summary of prediction results.
    
    Returns:
        Dictionary with prediction statistics
    """
    try:
        students_df = db.get_all_students()
        
        if len(students_df) == 0:
            return {
                'total_students': 0,
                'predicted_students': 0,
                'unpredicted_students': 0,
                'risk_counts': {'None': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0},
                'avg_probability': 0.0
            }
        
        predicted_df = students_df[students_df['prediction_probability'].notna()]
        
        risk_counts = db.get_at_risk_students_count()
        
        avg_prob = predicted_df['prediction_probability'].mean() if len(predicted_df) > 0 else 0.0
        
        return {
            'total_students': len(students_df),
            'predicted_students': len(predicted_df),
            'unpredicted_students': len(students_df) - len(predicted_df),
            'risk_counts': risk_counts,
            'avg_probability': float(avg_prob)
        }
        
    except Exception as e:
        print(f"Error getting prediction summary: {e}")
        return {
            'total_students': 0,
            'predicted_students': 0,
            'unpredicted_students': 0,
            'risk_counts': {'None': 0, 'Mild': 0, 'Moderate': 0, 'Severe': 0},
            'avg_probability': 0.0
        }


def test_prediction_pipeline():
    """Test the prediction pipeline with dummy data."""
    print("Testing prediction pipeline...")
    
    # Check if model exists
    model, scaler, features = load_model_artifacts()
    
    if model is None:
        print("❌ Model not loaded. Train model first.")
        return False
    
    print(f"✅ Model loaded with {len(features)} features")
    print(f"   Feature sample: {features[:5]}...")
    
    # Create dummy student features (all 36)
    dummy_features = {}
    for db_col in FEATURE_NAME_MAPPING.values():
        dummy_features[db_col] = 1.0  # Dummy value
    
    try:
        # Test feature alignment
        aligned = align_features(dummy_features, features)
        print(f"✅ Feature alignment successful. Shape: {aligned.shape}")
        
        # Test scaling
        scaled = scaler.transform(aligned)
        print(f"✅ Scaling successful. Shape: {scaled.shape}")
        
        # Test prediction
        prob = model.predict(scaled, verbose=0)
        print(f"✅ Prediction successful. Probability: {prob[0][0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == '__main__':
    # Test mode
    print("=== Prediction Module Test ===")
    test_prediction_pipeline()
