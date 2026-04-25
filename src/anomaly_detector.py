"""
AI Energy Anomaly Explainer
===========================
Production pipeline for detecting anomalies in industrial sensor data.

Author: Mahima Rajesh
Project: Imperial College MSc BA Portfolio
Dataset: SKAB (Skoltech Anomaly Benchmark)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Set up logging so we can track what the pipeline is doing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# All settings in one place — easy to change without touching code
# ============================================================

SENSORS = ['Pressure', 'Volume Flow RateRMS', 'Current', 'Temperature']

WINDOW_SIZE = 30  # seconds — optimal from Day 3 experiments

FAULT_FILES = [
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/1.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/2.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve1/3.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/1.csv",
    "https://raw.githubusercontent.com/waico/SKAB/master/data/valve2/2.csv",
]

NORMAL_DATA_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data/anomaly-free/anomaly-free.csv"


# ============================================================
# FUNCTION 1: LOAD DATA
# ============================================================

def load_data():
    """
    Loads and combines normal operation data with multiple fault experiments.
    
    Returns:
        df: Combined DataFrame with all sensor readings and anomaly labels
    """
    logger.info("Loading data...")
    dfs = []
    
    # Load normal operation data
    df_normal = pd.read_csv(
        NORMAL_DATA_URL, 
        sep=';', 
        index_col='datetime', 
        parse_dates=True
    )
    df_normal['anomaly'] = 0.0
    df_normal['changepoint'] = 0.0
    dfs.append(df_normal)
    logger.info(f"Normal data loaded: {len(df_normal)} rows")
    
    # Load each fault experiment
    for url in FAULT_FILES:
        try:
            df_fault = pd.read_csv(
                url, 
                sep=';', 
                index_col='datetime', 
                parse_dates=True
            )
            dfs.append(df_fault)
            logger.info(f"Fault data loaded: {url.split('/')[-2]}/{url.split('/')[-1]} "
                       f"— {len(df_fault)} rows, {int(df_fault['anomaly'].sum())} anomalies")
        except Exception as e:
            logger.error(f"Could not load {url}: {e}")
    
    # Combine all data
    df_combined = pd.concat(dfs)
    logger.info(f"Total data: {len(df_combined)} rows, "
               f"{int(df_combined['anomaly'].sum())} anomalies "
               f"({df_combined['anomaly'].mean():.1%} anomaly rate)")
    
    return df_combined


# ============================================================
# FUNCTION 2: ENGINEER FEATURES
# ============================================================

def engineer_features(df, sensors=SENSORS, window=WINDOW_SIZE):
    """
    Creates rolling statistical features from raw sensor readings.
    
    For each sensor adds:
    - Rolling mean: average over last N seconds (captures trends)
    - Rolling std: variation over last N seconds (captures stability)
    - Rate of change: change from last second (captures sudden shifts)
    
    Args:
        df: Raw sensor DataFrame
        sensors: List of sensor column names to use
        window: Rolling window size in seconds
        
    Returns:
        df_features: DataFrame with engineered features, NaN rows dropped
    """
    logger.info(f"Engineering features: {len(sensors)} sensors, "
               f"window={window}s")
    
    df_features = pd.DataFrame(index=df.index)
    
    for sensor in sensors:
        # Raw value
        df_features[sensor] = df[sensor]
        
        # Rolling mean — what is the trend?
        df_features[f'{sensor}_mean_{window}s'] = (
            df[sensor].rolling(window=window).mean()
        )
        
        # Rolling std — how stable is it?
        df_features[f'{sensor}_std_{window}s'] = (
            df[sensor].rolling(window=window).std()
        )
        
        # Rate of change — how fast is it moving?
        df_features[f'{sensor}_roc'] = df[sensor].diff()
    
    # Drop rows with NaN (first N rows due to rolling window)
    df_features = df_features.dropna()
    
    logger.info(f"Features engineered: {df_features.shape[1]} features, "
               f"{df_features.shape[0]} rows")
    
    return df_features


# ============================================================
# FUNCTION 3: TRAIN MODEL
# ============================================================

def train_model(X, y):
    """
    Trains a Random Forest classifier on engineered features.
    Uses stratified split to ensure balanced train/test sets.
    
    Args:
        X: Feature DataFrame
        y: Label Series (0=normal, 1=anomaly)
        
    Returns:
        model: Trained RandomForestClassifier
        X_test: Test features
        y_test: Test labels
    """
    logger.info("Training Random Forest model...")
    
    # Stratified split — ensures same anomaly rate in train and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    
    for train_idx, test_idx in sss.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
    
    logger.info(f"Training: {len(X_train)} rows "
               f"({y_train.mean():.1%} anomaly rate)")
    logger.info(f"Testing: {len(X_test)} rows "
               f"({y_test.mean():.1%} anomaly rate)")
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    return model, X_test, y_test


# ============================================================
# FUNCTION 4: DETECT ANOMALIES
# ============================================================

def detect_anomalies(model, X):
    """
    Uses trained model to detect anomalies in sensor data.
    Returns both binary predictions and probability scores.
    
    Args:
        model: Trained RandomForestClassifier
        X: Feature DataFrame
        
    Returns:
        predictions: Binary array (0=normal, 1=anomaly)
        probabilities: Anomaly probability for each row (0 to 1)
    """
    logger.info(f"Detecting anomalies in {len(X)} rows...")
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    anomaly_count = predictions.sum()
    logger.info(f"Detected {anomaly_count} anomalies "
               f"({anomaly_count/len(X):.1%} of rows)")
    
    return predictions, probabilities


# ============================================================
# FUNCTION 5: EVALUATE MODEL
# ============================================================

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Prints comprehensive model performance report.
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_proba: Anomaly probabilities (optional)
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE REPORT")
    print("="*60)
    
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Anomaly']
    ))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives  (correctly said normal):    {cm[0][0]}")
    print(f"  False Positives (wrongly said anomaly):     {cm[0][1]}")
    print(f"  False Negatives (missed real anomaly):      {cm[1][0]}")
    print(f"  True Positives  (correctly caught anomaly): {cm[1][1]}")
    
    precision = cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0
    recall = cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Precision: {precision:.3f} — when alarm fires, it is real {precision:.0%} of the time")
    print(f"  Recall:    {recall:.3f} — catches {recall:.0%} of all real faults")
    print(f"  F1 Score:  {f1:.3f}")
    print("="*60)


# ============================================================
# MAIN — runs the full pipeline end to end
# ============================================================

if __name__ == "__main__":
    print("AI Energy Anomaly Explainer — Full Pipeline")
    print("="*60)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Engineer features
    X = engineer_features(df)
    y = df['anomaly'][X.index]
    
    # Step 3: Train model
    model, X_test, y_test = train_model(X, y)
    
    # Step 4: Detect anomalies on test set
    predictions, probabilities = detect_anomalies(model, X_test)
    
    # Step 5: Evaluate
    evaluate_model(y_test, predictions, probabilities)
    
    print("\nPipeline complete.")
    print("Next step: Connect LLM to explain detected anomalies.")