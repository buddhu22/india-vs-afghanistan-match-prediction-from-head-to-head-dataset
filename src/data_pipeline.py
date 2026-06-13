import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def load_data(filepath='data/synthetic_match_data.csv'):
    return pd.read_csv(filepath)

def preprocess_and_engineer_features(df):
    """
    Handle missing values, encode categories, and prepare features.
    """
    # 1. Handling missing values
    df.fillna(method='ffill', inplace=True)
    
    # 2. Encoding Categorical Variables
    label_encoders = {}
    categorical_cols = ['venue', 'venue_type', 'toss_winner', 'toss_decision', 'batting_first']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + '_encoded'] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    # Target Encoding for Winner
    # India: 1, Afghanistan: 0
    df['winner_encoded'] = df['winner'].map({'India': 1, 'Afghanistan': 0})
    
    # Encode Top Scorer & Bowler
    le_scorer = LabelEncoder()
    df['top_scorer_encoded'] = le_scorer.fit_transform(df['top_scorer'])
    label_encoders['top_scorer'] = le_scorer
    
    le_bowler = LabelEncoder()
    df['top_bowler_encoded'] = le_bowler.fit_transform(df['top_bowler'])
    label_encoders['top_bowler'] = le_bowler
    
    # Feature Engineering (already generated in synthetic data, but preparing them here)
    features = [
        'venue_encoded', 'venue_type_encoded', 'toss_winner_encoded', 
        'toss_decision_encoded', 'batting_first_encoded',
        'ind_batting_strength', 'ind_bowling_strength',
        'afg_batting_strength', 'afg_bowling_strength',
        'ind_recent_form', 'afg_recent_form',
        'ind_player_form_idx', 'afg_player_form_idx',
        'ind_h2h_win_pct', 'ind_avg_score', 'afg_avg_score'
    ]
    
    X = df[features]
    y_winner = df['winner_encoded']
    y_scorer = df['top_scorer_encoded']
    y_bowler = df['top_bowler_encoded']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # Save encoders and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(features, 'models/features_list.pkl')
    
    return X_scaled, y_winner, y_scorer, y_bowler

def get_train_test_splits():
    df = load_data()
    X, y_winner, y_scorer, y_bowler = preprocess_and_engineer_features(df)
    
    # Split for winner prediction
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(X, y_winner, test_size=0.2, random_state=42, stratify=y_winner)
    
    # Split for scorer
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_scorer, test_size=0.2, random_state=42)
    
    # Split for bowler
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X, y_bowler, test_size=0.2, random_state=42)
    
    return {
        'winner': (X_train_w, X_test_w, y_train_w, y_test_w),
        'scorer': (X_train_s, X_test_s, y_train_s, y_test_s),
        'bowler': (X_train_b, X_test_b, y_train_b, y_test_b),
        'X': X # for SHAP
    }

if __name__ == "__main__":
    splits = get_train_test_splits()
    print("Data Pipeline Executed Successfully. Encoders and Scaler saved.")
