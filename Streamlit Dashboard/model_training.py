# model_training.py
# MLP Model Training with Feature Tracking

import pandas as pd
import numpy as np
import os
import random
import sys
import warnings
import joblib
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    print("❌ TensorFlow is not installed. Please run: pip install tensorflow")
    sys.exit(1)

from config import *
from utils import save_meta_json, parse_csv_with_separator


# Custom progress callback
class ProgressCallback(callbacks.Callback):
    """Custom callback to print training progress."""
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"Epoch {epoch+1}: loss={logs.get('loss', 0):.4f}, "
              f"accuracy={logs.get('accuracy', 0):.4f}, "
              f"val_auc={logs.get('val_auc', 0):.4f}")
        sys.stdout.flush()  # Force output to appear immediately


# Set seeds for reproducibility
SEED = MODEL_CONFIG['random_state']
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path: str) -> Optional[Tuple]:
    """
    Load, preprocess, and split data. Returns scaler and column names.
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names)
    """
    print(f"--- Loading data from '{file_path}' ---")
    
    try:
        df = parse_csv_with_separator(file_path, CSV_SEPARATOR)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    # Find target column (case-insensitive)
    target_col = next((c for c in df.columns if c.strip().lower() == "target"), None)
    if not target_col:
        print("❌ Error: 'Target' column not found.")
        return None
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Preprocessing
    print("Preprocessing features...")
    X_enc = pd.get_dummies(X, drop_first=True)
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    for col in X_enc.columns:
        if X_enc[col].isna().any():
            median_val = X_enc[col].median()
            X_enc[col].fillna(median_val, inplace=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, 
        test_size=MODEL_CONFIG['test_size'], 
        random_state=SEED, 
        stratify=y
    )
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data preparation complete.")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    print(f"Features after encoding: {X_train_scaled.shape[1]}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns.tolist()


def create_mlp_model(input_shape: Tuple[int]) -> models.Sequential:
    """
    Create MLP model architecture.
    
    Args:
        input_shape: Tuple representing input dimensions
        
    Returns:
        Compiled Keras Sequential model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(MODEL_CONFIG['hidden_layer_1'], activation='relu'),
        layers.Dropout(MODEL_CONFIG['dropout_1']),
        layers.Dense(MODEL_CONFIG['hidden_layer_2'], activation='relu'),
        layers.Dropout(MODEL_CONFIG['dropout_2']),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(file_path: Optional[str] = None) -> Tuple[bool, str]:
    """
    Train MLP model and save artifacts.
    
    Args:
        file_path: Path to training CSV file
        
    Returns:
        Tuple of (success, message)
    """
    # Load and prepare data
    file_to_use = file_path if file_path is not None else TRAINING_DATA_PATH
    prepared_data = load_and_prepare_data(file_to_use)
    if prepared_data is None:
        return False, "Failed to load or prepare data"
    
    X_train, X_test, y_train, y_test, scaler, feature_names = prepared_data
    
    print(f"--- Building and Training MLP Model ---")
    
    # Create model
    model = create_mlp_model(input_shape=(X_train.shape[1],))
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Callbacks (matches reference code for reproducibility)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=10,  # Matches reference code for 0.9215 accuracy
        restore_best_weights=True,
        verbose=1
    )
    
    progress_callback = ProgressCallback()
    
    # Train model
    print(f"\n🎯 Training model for up to {MODEL_CONFIG['epochs']} epochs...")
    print("Progress will be shown below:\n")
    sys.stdout.flush()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MODEL_CONFIG['epochs'],
        batch_size=MODEL_CONFIG['batch_size'],
        callbacks=[early_stopping, progress_callback],
        verbose=0  # Changed to 0, using custom callback instead
    )
    
    print("\n✅ Model training finished.")
    
    # Evaluate model
    print("\n--- Evaluating Model Performance ---")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions for classification report
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Graduate', 'Dropout']))
    
    # Save model and artifacts
    print("\n--- Saving Model and Artifacts ---")
    
    # Create directory
    os.makedirs(MODEL_ASSETS_DIR, exist_ok=True)
    
    # 1. Save TensorFlow model
    model.save(MODEL_PATH)
    print(f"✅ Model saved to: {MODEL_PATH}")
    
    # 2. Save StandardScaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"✅ Scaler saved to: {SCALER_PATH}")
    
    # 3. Save feature columns
    joblib.dump(feature_names, FEATURE_COLUMNS_PATH)
    print(f"✅ Feature columns saved to: {FEATURE_COLUMNS_PATH}")
    
    # 4. Save metadata to meta.json
    metadata = {
        'trained_features': feature_names,
        'num_features': len(feature_names),
        'training_date': pd.Timestamp.now().isoformat(),
        'model_path': MODEL_PATH,
        'scaler_path': SCALER_PATH,
        'feature_columns_path': FEATURE_COLUMNS_PATH,
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'test_loss': float(test_loss)
    }
    
    save_meta_json(metadata)
    print(f"✅ Metadata saved to: {META_JSON_PATH}")
    
    # 5. Save to database
    try:
        import database as db
        db.save_model_metadata(
            trained_features=feature_names,
            model_path=MODEL_PATH,
            scaler_path=SCALER_PATH,
            feature_columns_path=FEATURE_COLUMNS_PATH
        )
        print(f"✅ Metadata saved to database")
    except Exception as e:
        print(f"⚠️ Warning: Could not save to database: {e}")
    
    return True, f"Model trained successfully! Test AUC: {test_auc:.4f}"


def retrain_model_with_new_data(new_data_path: str, target_training_path: str) -> Tuple[bool, str]:
    """
    Retrain model with new training data.
    
    Args:
        new_data_path: Path to new training data CSV
        target_training_path: Path where to save the new training data
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Validate new data file exists
        if not os.path.exists(new_data_path):
            return False, "Training data file not found"
        
        # Replace old training data with new one
        import shutil
        shutil.copy(new_data_path, target_training_path)
        print(f"✅ Training data replaced: {target_training_path}")
        
        # Train model with new data
        success, message = train_model(target_training_path)
        
        if success:
            return True, f"Model retrained successfully! {message}"
        else:
            return False, f"Retraining failed: {message}"
            
    except Exception as e:
        return False, f"Error during retraining: {e}"


if __name__ == '__main__':
    # Command-line interface for standalone training
    if len(sys.argv) < 2:
        csv_file_path = TRAINING_DATA_PATH
        print(f"No CSV path provided. Using default from config: {csv_file_path}")
    else:
        csv_file_path = sys.argv[1]
    
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: File '{csv_file_path}' not found.")
        sys.exit(1)
    
    success, message = train_model(csv_file_path)
    
    if success:
        print(f"\n🎉 {message}")
    else:
        print(f"\n❌ {message}")
        sys.exit(1)
