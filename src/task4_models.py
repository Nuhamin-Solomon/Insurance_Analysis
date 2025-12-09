"""
Task 4 models script
- Data preparation and defensive checks
- Train 3 models: RandomForest, XGBoost (if installed), LightGBM (if installed) or fallback to sklearn GradientBoosting
- Evaluate classification (ClaimFrequency) and regression (ClaimSeverity)
- Compare metrics and save results
- Run SHAP explainability on best-performing regression model for ClaimSeverity
- Save key outputs to results/
"""
import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'insurance_data.csv')
OUT_DIR = os.path.join(ROOT, 'results')
MODEL_DIR = os.path.join(ROOT, 'models')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Try to import XGBoost / LightGBM if available
try:
    import xgboost as xgb
    has_xgb = True
except Exception:
    has_xgb = False

try:
    import lightgbm as lgb
    has_lgb = True
except Exception:
    has_lgb = False

def load_prepare():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found: " + DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    # compute KPIs
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
    df['ClaimFrequency'] = (df['TotalClaims']>0).astype(int)
    if 'NumberOfClaims' in df.columns:
        df['ClaimSeverity'] = df['TotalClaims'] / df['NumberOfClaims'].replace(0, np.nan)
    else:
        df['ClaimSeverity'] = df['TotalClaims'].where(df['TotalClaims']>0, 0)

    # Example features set - adapt to your dataset
    # dynamically detect columns commonly present
    candidate_cat = [c for c in df.columns if c.lower() in ('gender','sex','province','vehicletype','make','model')]
    candidate_num = [c for c in df.columns if c.lower() in ('age','totalpremium','customvalueestimate','custom_value_estimate')]
    # fallback if not found
    if not candidate_cat:
        candidate_cat = [c for c in df.columns if df[c].dtype=='object'][:3]
    if not candidate_num:
        candidate_num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)][:2]

    features = candidate_num + candidate_cat
    print("Features used:", features)
    df = df.dropna(subset=features + ['ClaimFrequency', 'ClaimSeverity'])
    X = df[features]
    y_class = df['ClaimFrequency']
    y_reg = df['ClaimSeverity']
    return X, y_class, y_reg, candidate_num, candidate_cat

def build_preprocessor(numerical, categorical):
    num_pipe = ('num', StandardScaler(), numerical)
    cat_pipe = ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical)
    return ColumnTransformer([num_pipe, cat_pipe])

def fit_and_eval_classification(X_train, X_test, y_train, y_test, preprocessor):
    results = {}
    models = {
        'rf': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    # add xgb if present
    if has_xgb:
        models['xgb'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    if has_lgb:
        models['lgbm'] = lgb.LGBMClassifier(random_state=42)

    for name, model in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:,1] if hasattr(pipe, 'predict_proba') else None
        metrics = {
            'accuracy': accuracy_score(y_test, pred),
            'precision': precision_score(y_test, pred, zero_division=0),
            'recall': recall_score(y_test, pred, zero_division=0),
            'f1': f1_score(y_test, pred, zero_division=0),
            'auc': roc_auc_score(y_test, proba) if proba is not None else np.nan
        }
        results[name] = {'pipeline': pipe, 'metrics': metrics}
        # save model
        joblib.dump(pipe, os.path.join(MODEL_DIR, f'class_{name}.joblib'))
    return results

def fit_and_eval_regression(X_train, X_test, y_train, y_test, preprocessor):
    results = {}
    models = {
        'rf': RandomForestRegressor(n_estimators=200, random_state=42)
    }
    if has_xgb:
        models['xgb'] = xgb.XGBRegressor(random_state=42)
    if has_lgb:
        models['lgbm'] = lgb.LGBMRegressor(random_state=42)
    # fallback
    models['gbr'] = GradientBoostingRegressor(random_state=42)

    for name, model in models.items():
        pipe = Pipeline([('pre', preprocessor), ('reg', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'mae': mean_absolute_error(y_test, pred),
            'r2': r2_score(y_test, pred)
        }
        results[name] = {'pipeline': pipe, 'metrics': metrics}
        joblib.dump(pipe, os.path.join(MODEL_DIR, f'reg_{name}.joblib'))
    return results

def compare_and_save(metrics_cls, metrics_reg):
    # flatten metrics and save CSV
    rows = []
    for name, info in metrics_cls.items():
        m = info['metrics']
        rows.append({'task':'classification','model':name, **m})
    for name, info in metrics_reg.items():
        m = info['metrics']
        rows.append({'task':'regression','model':name, **m})
    dfm = pd.DataFrame(rows)
    out = os.path.join(OUT_DIR,'task4_model_metrics.csv')
    dfm.to_csv(out, index=False)
    print("Saved metrics to", out)
    return dfm

def run_shap_on_best_reg(best_pipeline, X_train):
    # SHAP on tree models - try TreeExplainer
    try:
        import shap
    except Exception:
        print("shap not installed; skipping SHAP step.")
        return

    # build transformed (preprocessed) features and feature names
    pre = best_pipeline.named_steps['pre']
    X_trans = pre.transform(X_train)
    # feature names
    num = pre.transformers_[0][2]
    cat = pre.transformers_[1][2]
    cat_names = pre.named_transformers_['cat'].get_feature_names_out(cat)
    feature_names = list(num) + list(cat_names)
    model = best_pipeline.named_steps[list(best_pipeline.named_steps.keys())[-1]]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)
        # summary plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        shap.summary_plot(shap_values, features=X_trans, feature_names=feature_names, show=False)
        plt.tight_layout()
        fn = os.path.join(OUT_DIR, 'task4_shap_summary.png')
        plt.savefig(fn, dpi=150)
        print("Saved SHAP plot to", fn)
    except Exception as e:
        print("SHAP TreeExplainer failed:", e)

def main():
    X, y_class, y_reg, num_cols, cat_cols = load_prepare()
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    print("Training classification models...")
    cls_results = fit_and_eval_classification(X_train_c, X_test_c, y_train_c, y_test_c, preprocessor)
    print("Training regression models...")
    reg_results = fit_and_eval_regression(X_train_r, X_test_r, y_train_r, y_test_r, preprocessor)

    metrics_df = compare_and_save(cls_results, reg_results)
    print(metrics_df)

    # pick best reg model by RMSE
    reg_best_name = min((name for name in reg_results.keys()), key=lambda n: reg_results[n]['metrics']['rmse'])
    print("Best regression model:", reg_best_name)
    best_pipeline = reg_results[reg_best_name]['pipeline']
    # run SHAP explanation on a sample of training data
    run_shap_on_best_reg(best_pipeline, X_train_r.sample(min(500, len(X_train_r)), random_state=42))

    print("Done.")

if __name__ == "__main__":
    main()
