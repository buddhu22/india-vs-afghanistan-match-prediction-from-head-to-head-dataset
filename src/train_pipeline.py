import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import warnings
from data_pipeline import get_train_test_splits

warnings.filterwarnings('ignore')

def evaluate_model(model, X_test, y_test, is_multiclass=False):
    y_pred = model.predict(X_test)
    if is_multiclass:
        acc = accuracy_score(y_test, y_pred)
        return acc
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }
        return metrics

def train_winner_models(X_train, X_test, y_train, y_test):
    print("Training Winner Prediction Models...")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=0)
    }
    
    # Voting Ensemble
    voting = VotingClassifier(
        estimators=[('xgb', models['XGBoost']), ('rf', models['Random Forest']), ('cb', models['CatBoost'])],
        voting='soft'
    )
    models['Voting Ensemble'] = voting
    
    results = {}
    best_model = None
    best_auc = 0
    best_model_name = ""
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"{name} -> Accuracy: {metrics['Accuracy']:.4f}, AUC: {metrics['ROC-AUC']:.4f}")
        
        if metrics['ROC-AUC'] > best_auc:
            best_auc = metrics['ROC-AUC']
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model Selected: {best_model_name} with AUC {best_auc:.4f}")
    
    # Save the best model and results
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_winner_model.pkl')
    joblib.dump(results, 'models/winner_model_metrics.pkl')
    
    return best_model

def train_player_models(X_train_s, X_test_s, y_train_s, y_test_s, X_train_b, X_test_b, y_train_b, y_test_b):
    print("\nTraining Player Prediction Models...")
    
    # Using Random Forest for multi-class player predictions
    scorer_model = RandomForestClassifier(random_state=42, n_estimators=100)
    scorer_model.fit(X_train_s, y_train_s)
    acc_scorer = evaluate_model(scorer_model, X_test_s, y_test_s, is_multiclass=True)
    print(f"Top Scorer Model Accuracy: {acc_scorer:.4f}")
    
    bowler_model = RandomForestClassifier(random_state=42, n_estimators=100)
    bowler_model.fit(X_train_b, y_train_b)
    acc_bowler = evaluate_model(bowler_model, X_test_b, y_test_b, is_multiclass=True)
    print(f"Top Bowler Model Accuracy: {acc_bowler:.4f}")
    
    joblib.dump(scorer_model, 'models/top_scorer_model.pkl')
    joblib.dump(bowler_model, 'models/top_bowler_model.pkl')

def generate_shap_explainer(best_model, X_train):
    print("\nGenerating SHAP Explainer...")
    # For tree-based models
    try:
        explainer = shap.TreeExplainer(best_model)
    except:
        # Fallback for logistic regression or voting ensemble
        explainer = shap.Explainer(best_model.predict, X_train)
        
    joblib.dump(explainer, 'models/shap_explainer.pkl')
    
    # Save a sample of X_train for baseline references
    joblib.dump(X_train.sample(100, random_state=42), 'models/shap_baseline_X.pkl')
    print("SHAP Explainer saved.")

if __name__ == "__main__":
    splits = get_train_test_splits()
    
    X_train_w, X_test_w, y_train_w, y_test_w = splits['winner']
    X_train_s, X_test_s, y_train_s, y_test_s = splits['scorer']
    X_train_b, X_test_b, y_train_b, y_test_b = splits['bowler']
    X_full = splits['X']
    
    best_winner_model = train_winner_models(X_train_w, X_test_w, y_train_w, y_test_w)
    train_player_models(X_train_s, X_test_s, y_train_s, y_test_s, X_train_b, X_test_b, y_train_b, y_test_b)
    
    # Only generate SHAP if the best model is tree-based for simplicity, or voting ensemble
    # In VotingClassifier, shap.TreeExplainer won't work directly on VotingClassifier.
    # To handle Explainability smoothly in the app, we can train a proxy XGBoost just for SHAP if best model is Voting.
    if isinstance(best_winner_model, VotingClassifier) or isinstance(best_winner_model, LogisticRegression):
        print("Training proxy XGBoost for SHAP Explainability...")
        proxy_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        proxy_xgb.fit(X_train_w, y_train_w)
        generate_shap_explainer(proxy_xgb, X_train_w)
    else:
        generate_shap_explainer(best_winner_model, X_train_w)
        
    print("\nTraining Pipeline Completed Successfully.")
