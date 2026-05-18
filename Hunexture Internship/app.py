from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, roc_auc_score, 
                             precision_recall_curve, auc, brier_score_loss, classification_report)
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
import json
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variable to hold all precomputed dashboard data
dashboard_data = {
    'models': {}
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.json_encoder = NumpyEncoder

def compute_cumulative_gains(y_true, y_prob):
    # Sort instances by descending predicted probability
    sorted_indices = np.argsort(y_prob)[::-1]
    y_true_sorted = y_true.iloc[sorted_indices].values
    
    # Cumulative number of positive cases
    cum_positives = np.cumsum(y_true_sorted)
    
    # Total number of positive cases
    total_positives = np.sum(y_true_sorted)
    
    # Percentage of total positive cases found
    percent_positives = cum_positives / total_positives
    
    # Percentage of sample targeted
    percent_sample = np.arange(1, len(y_true_sorted) + 1) / len(y_true_sorted)
    
    # Downsample points for Chart.js performance (100 points max)
    step = max(1, len(percent_sample) // 100)
    
    return percent_sample[::step].tolist(), percent_positives[::step].tolist()

def compute_lift_curve(percent_sample, percent_positives):
    # Lift = percentage of positives found / percentage of sample targeted
    # Add small epsilon to avoid division by zero
    lift = [p_pos / max(p_samp, 1e-10) for p_samp, p_pos in zip(percent_sample, percent_positives)]
    return lift

def train_and_compute_metrics():
    global dashboard_data
    
    print("Loading and Preprocessing Data...")
    df = pd.read_csv('diversified_ecommerce_dataset.csv')
    df = df.drop('Product ID', axis=1)
    
    le = LabelEncoder()
    categorical_cols = ['Product Name', 'Category', 'Supplier ID', 'Customer Age Group', 'Customer Location', 'Customer Gender', 'Shipping Method']
    for col in categorical_cols:
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])

    df['Seasonality'] = df['Seasonality'].map({'Yes': 1, 'No': 0}).fillna(0)

    X = df.drop('Seasonality', axis=1)
    y = df['Seasonality']
    feature_names = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    train_sample = 5000
    test_sample = 5000

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if len(X_train_full) > train_sample:
        X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=train_sample, stratify=y_train_full, random_state=42)
        X_test, _, y_test, _ = train_test_split(X_test_full, y_test_full, train_size=test_sample, stratify=y_test_full, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X_train_full, X_test_full, y_train_full, y_test_full

    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'k-Nearest Neighbors': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    print("Training models and computing 15 diverse metrics...")
    
    trained_models = {}
    base_accuracies = {}

    for name, model in models.items():
        print(f" -> Processing {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test)
        
        # Determine Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            y_prob_both = model.predict_proba(X_test)
        else:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min()) # Normalize to 0-1
            y_prob_both = np.vstack([1-y_prob, y_prob]).T
            
        acc = accuracy_score(y_test, y_pred)
        base_accuracies[name] = acc
        
        compute_model_dashboard(name, model, X_test, y_test, y_pred, y_prob, y_prob_both, acc, feature_names)

    # Hybrid Model
    sorted_models = sorted(base_accuracies.items(), key=lambda item: item[1], reverse=True)
    best_1, best_2 = sorted_models[0][0], sorted_models[1][0]
    
    print(f" -> Training Hybrid Model ({best_1} + {best_2})...")
    hybrid_model = VotingClassifier(
        estimators=[
            (best_1, trained_models[best_1]),
            (best_2, trained_models[best_2])
        ],
        voting='soft'
    )
    hybrid_model.fit(X_train, y_train)
    y_pred_h = hybrid_model.predict(X_test)
    y_prob_h = hybrid_model.predict_proba(X_test)[:, 1]
    y_prob_both_h = hybrid_model.predict_proba(X_test)
    acc_h = accuracy_score(y_test, y_pred_h)
    
    compute_model_dashboard('Hybrid Model', hybrid_model, X_test, y_test, y_pred_h, y_prob_h, y_prob_both_h, acc_h, feature_names, is_hybrid=True)
    
    print("All preprocessing complete. Ready to serve dashboard!")

def compute_model_dashboard(name, model, X_test, y_test, y_pred, y_prob, y_prob_both, acc, feature_names, is_hybrid=False):
    global dashboard_data
    
    # Basic Metrics
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    error_rate = 1.0 - acc

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Downsample ROC points for frontend
    step_roc = max(1, len(fpr) // 50)
    
    # PR Curve
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(pr_rec, pr_prec)
    step_pr = max(1, len(pr_prec) // 50)
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Cumulative Gains & Lift
    percent_sample, percent_positives = compute_cumulative_gains(y_test, y_prob)
    lift = compute_lift_curve(percent_sample, percent_positives)
    
    # Probability Distribution Histogram
    hist_0, bin_edges = np.histogram(y_prob[y_test == 0], bins=20, range=(0, 1))
    hist_1, _ = np.histogram(y_prob[y_test == 1], bins=20, range=(0, 1))
    bins = [float((bin_edges[i] + bin_edges[i+1])/2) for i in range(len(bin_edges)-1)]
    
    # Classification Report
    cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Class-wise metrics
    dist_actual = [int(np.sum(y_test == 0)), int(np.sum(y_test == 1))]
    dist_pred = [int(np.sum(y_pred == 0)), int(np.sum(y_pred == 1))]
    
    # Feature Importance (Permutation based to work across all types, fast config)
    # Skip hybrid model for time, or assign sum of top models
    imp_features = feature_names
    imp_scores = [0] * len(feature_names)
    if not is_hybrid:
        try:
            r = permutation_importance(model, X_test, y_test, n_repeats=1, random_state=42, n_jobs=-1)
            imp_scores = r.importances_mean.tolist()
        except Exception:
            pass # Fallback generic
            
    # Sort importances
    imp_tuples = sorted(zip(imp_features, imp_scores), key=lambda x: x[1], reverse=True)
    imp_features = [x[0] for x in imp_tuples][:10]
    imp_scores = [x[1] for x in imp_tuples][:10]

    # Structure into the 15 charts requirements
    dashboard_data['models'][name] = {
        'metrics': {
            'accuracy': float(acc), 'precision': float(prec), 
            'recall': float(rec), 'f1': float(f1), 'error_rate': float(error_rate)
        },
        'cm': cm.tolist(),
        'tp_tn_fp_fn': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        'roc': {
            'fpr': fpr[::step_roc].tolist(), 'tpr': tpr[::step_roc].tolist(), 'auc': float(roc_auc)
        },
        'pr': {
            'precision': pr_prec[::step_pr].tolist(), 'recall': pr_rec[::step_pr].tolist(), 'auc': float(pr_auc)
        },
        'calibration': {
            'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()
        },
        'gains': {
            'percent_sample': percent_sample, 'percent_positives': percent_positives
        },
        'lift': {
            'percent_sample': percent_sample, 'lift': lift
        },
        'prob_hist': {
            'bins': bins, 'class_0': hist_0.tolist(), 'class_1': hist_1.tolist()
        },
        'class_report': {
            '0': {'precision': cr['0']['precision'], 'recall': cr['0']['recall'], 'f1': cr['0']['f1-score']},
            '1': {'precision': cr['1']['precision'], 'recall': cr['1']['recall'], 'f1': cr['1']['f1-score']}
        },
        'distributions': {
            'actual': dist_actual, 'predicted': dist_pred
        },
        'importance': {
            'features': imp_features, 'scores': imp_scores
        }
    }

# Run training
try:
    train_and_compute_metrics()
except Exception as e:
    print(f"Error during training computation: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dashboard_data')
def get_dashboard_data():
    return jsonify(dashboard_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
