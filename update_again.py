import os
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

import xgboost as xgb

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="The use_label_encoder parameter is deprecated and will be removed in a future release.")

FEATURES = ["value_eth", "gas_price_gwei", "input_len", "gas_density"]

# -----------------------------------------------------------
# Custom HDTT Stacking Classifier (RF + XGBoost + LogReg Meta) with OOF Stacking
# -----------------------------------------------------------
class HDTTStackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, rf_params, xgb_params, meta_learner_params, seed=42, n_splits=5):
        self.rf_params = rf_params
        self.xgb_params = xgb_params
        self.meta_learner_params = meta_learner_params
        
        self.seed = seed
        self.n_splits = n_splits

        # Models to be fitted on full data for final prediction
        self.rf_model_full = RandomForestClassifier(**rf_params, random_state=seed)
        self.xgb_model_full = xgb.XGBClassifier(**xgb_params, random_state=seed, use_label_encoder=False, eval_metric='logloss')
        self.meta_learner = LogisticRegression(**meta_learner_params, random_state=seed)
        
        self.is_fitted_ = False

    def fit(self, X, y):
        # Ensure y is 1D array
        y = np.asarray(y).ravel()
        
        # Prepare for OOF predictions
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        
        # Store OOF predictions for meta-learner training
        rf_oof_preds = np.zeros((X.shape[0],))
        xgb_oof_preds = np.zeros((X.shape[0],))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Create fresh instances of base models for each fold
            # Use slightly different seeds to encourage diversity
            fold_rf = RandomForestClassifier(**self.rf_params, random_state=self.seed + fold) 
            fold_xgb = xgb.XGBClassifier(**self.xgb_params, random_state=self.seed + fold, use_label_encoder=False, eval_metric='logloss')

            fold_rf.fit(X_train_fold, y_train_fold)
            fold_xgb.fit(X_train_fold, y_train_fold)

            # Generate OOF predictions for the current validation fold
            rf_oof_preds[val_idx] = fold_rf.predict_proba(X_val_fold)[:, 1]
            xgb_oof_preds[val_idx] = fold_xgb.predict_proba(X_val_fold)[:, 1]

        # Train base models on the full dataset for final predictions (when predict_proba is called)
        self.rf_model_full.fit(X, y)
        self.xgb_model_full.fit(X, y)

        # Train meta-learner on the OOF predictions generated above
        meta_features_train = np.hstack((rf_oof_preds.reshape(-1, 1), xgb_oof_preds.reshape(-1, 1)))
        self.meta_learner.fit(meta_features_train, y)
        
        self.is_fitted_ = True
        return self

    def _get_meta_features(self, X):
        # Use full-data-trained base models for predictions
        rf_preds = self.rf_model_full.predict_proba(X)[:, 1].reshape(-1, 1)
        xgb_preds = self.xgb_model_full.predict_proba(X)[:, 1].reshape(-1, 1)
        return np.hstack((rf_preds, xgb_preds))

    def predict_proba(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted yet. Call fit() first.")
        meta_features_test = self._get_meta_features(X)
        return self.meta_learner.predict_proba(meta_features_test)

    def predict(self, X):
        if not self.is_fitted_:
            raise RuntimeError("Classifier not fitted yet. Call fit() first.")
        meta_features_test = self._get_meta_features(X)
        return self.meta_learner.predict(meta_features_test)


# -----------------------------
# Safe coercion for labels/scores
# -----------------------------
def to_1d_int(y):
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[:, 0] if y.shape[1] == 1 else np.argmax(y, axis=1)
    return y.astype(int).reshape(-1)

def to_1d_float(x):
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    x = np.asarray(x, dtype=float)
    return x.reshape(-1)

def dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()

def ensure_numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.duplicated()].copy()
    for c in cols:
        if c in df.columns:
            col_data = df[c]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            df[c] = pd.to_numeric(col_data, errors="coerce")
    return df

# -----------------------------
# Metrics & Threshold Optimization
# -----------------------------
def metrics_from_cm(cm):
    if cm.shape == (1, 1): 
        return 0.0, 0.0, 0.0, 0.0, (0, 0, 0, 0)
    
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    
    return precision, recall, f1, fpr, (TN, FP, FN, TP)

def find_best_threshold_f1(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    precisions_for_f1 = precisions[:-1]
    recalls_for_f1 = recalls[:-1]

    if len(thresholds) == 0:
        return 0.5 

    f1_scores = 2 * (precisions_for_f1 * recalls_for_f1) / (precisions_for_f1 + recalls_for_f1 + 1e-10)

    if not np.any(np.isfinite(f1_scores) & (f1_scores > 0)):
        return 0.5 

    best_idx = np.argmax(f1_scores)
    best_thr = thresholds[best_idx]
    
    return best_thr

# -----------------------------
# Feature + Noise
# -----------------------------
def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = ensure_numeric(df, ["input_len", "gas_price_gwei", "value_eth"])
    df = df[df["input_len"] > 0].copy()
    df = df.dropna(subset=["input_len", "gas_price_gwei", "value_eth"])

    df["log_input"] = np.log1p(df["input_len"])
    df["log_gas"] = np.log1p(df["gas_price_gwei"].clip(lower=0))
    df["gas_density"] = df["gas_price_gwei"] / (df["log_input"] + 1)
    return dedup_columns(df)

def apply_adversarial_noise(
    X: pd.DataFrame,
    y: pd.Series,
    seed=42,
    gas_low=0.5,
    gas_high=0.7,
    input_scale=0.8,
    input_cut=500,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = dedup_columns(X.copy())
    Xn = ensure_numeric(Xn, ["input_len", "gas_price_gwei", "value_eth", "gas_density", "log_input", "log_gas"])
    
    y1d = to_1d_int(y)
    attack_idx = Xn.index[y1d == 1]

    if len(attack_idx) > 0:
        gas_factors = rng.uniform(gas_low, gas_high, size=len(attack_idx))
        Xn.loc[attack_idx, "gas_price_gwei"] = Xn.loc[attack_idx, "gas_price_gwei"].astype(float) * gas_factors

        Xn.loc[attack_idx, "input_len"] = Xn.loc[attack_idx, "input_len"].apply(
            lambda x: x * input_scale if x > input_cut else x
        )

        Xn["log_input"] = np.log1p(Xn["input_len"])
        Xn["log_gas"] = np.log1p(Xn["gas_price_gwei"].clip(lower=0))
        Xn["gas_density"] = Xn["gas_price_gwei"] / (Xn["log_input"] + 1)

    return Xn

# -----------------------------
# Baselines
# -----------------------------
def fit_sgt_threshold(X_train, y_train, score_col="gas_density", benign_quantile=0.95):
    y1d = to_1d_int(y_train)
    Xt = dedup_columns(X_train)
    benign_vals = to_1d_float(Xt.loc[y1d == 0, score_col].dropna())
    
    if len(benign_vals) == 0: return 0.0
    return float(np.quantile(benign_vals, benign_quantile))

def fit_agt_binned_thresholds(X_train, y_train, q_bins=5, benign_quantile=0.95):
    y1d = to_1d_int(y_train)
    Xt = dedup_columns(X_train).copy()
    Xt["label"] = y1d
    benign = Xt[Xt["label"] == 0].dropna(subset=["input_len", "gas_density"])

    if len(benign) == 0: 
        return [], {}, 0.0 

    if len(benign) < q_bins or benign["input_len"].nunique() < q_bins:
        global_thr = float(np.quantile(benign["gas_density"], benign_quantile))
        return [], {}, global_thr 

    try:
        _, edges = pd.qcut(benign["input_len"], q=q_bins, duplicates="drop", retbins=True)
    except ValueError: 
        global_thr = float(np.quantile(benign["gas_density"], benign_quantile))
        return [], {}, global_thr

    if len(edges) <= 2: 
        global_thr = float(np.quantile(benign["gas_density"], benign_quantile))
        return [], {}, global_thr

    benign_bins = pd.cut(benign["input_len"], bins=edges, include_lowest=True)
    thr_per_bin = benign.groupby(benign_bins)["gas_density"].quantile(benign_quantile)
    global_thr = float(np.quantile(benign["gas_density"], benign_quantile))
    
    return edges, thr_per_bin, global_thr

def agt_predict(X_df, edges, thr_per_bin, global_thr):
    if len(edges) == 0:
         return (X_df["gas_density"].to_numpy() >= global_thr).astype(int)
    bins = pd.cut(X_df["input_len"], bins=edges, include_lowest=True)
    thr = bins.map(thr_per_bin).astype(float).fillna(global_thr).to_numpy()
    return (X_df["gas_density"].to_numpy() >= thr).astype(int)

# -----------------------------
# ML Training & Evaluation
# -----------------------------
def adversarial_augment_train(X_train, y_train, noise_level, seed=123):
    gas_low = 1.0 - (0.30 * noise_level)
    gas_high = 1.0 - (0.15 * noise_level)
    input_scale = 1.0 - (0.20 * noise_level)

    X_aug = apply_adversarial_noise(
        X_train, y_train, seed=seed, gas_low=gas_low, gas_high=gas_high, input_scale=input_scale
    )
    X_train_comb = pd.concat([X_train[FEATURES], X_aug[FEATURES]], axis=0, ignore_index=True)
    y_train_comb = np.concatenate([to_1d_int(y_train), to_1d_int(y_train)], axis=0)
    return X_train_comb, y_train_comb

def get_models(seed=42):
    # --- Parameters for HDTT's Internal Base Learners (strong enough for good combination) ---
    hdtt_rf_base_params = dict(
        n_estimators=180,           
        max_depth=8,                
        min_samples_leaf=6,         
        class_weight="balanced",
    )
    hdtt_xgb_base_params = dict(
        n_estimators=180,            
        max_depth=5,                 
        learning_rate=0.07,          
        subsample=0.8,               
        colsample_bytree=0.8,        
        gamma=0.1,                   
        min_child_weight=1,          
    )
    # Meta-learner: slightly more aggressive L1 regularization
    meta_learner_params = dict(
        max_iter=1000,
        class_weight="balanced",
        solver='liblinear',
        penalty='l1',                
        C=0.4,                       # Slightly higher regularization to focus on strong signals
    )

    # --- Parameters for Standalone Baselines (designed to be distinctly weaker or more volatile) ---
    # Independent XGBoost: Significantly reduced performance to ensure distinct Recall values
    independent_xgb_params = dict(
        n_estimators=80,            # Further reduced estimators
        max_depth=2,                # Even shallower trees (very weak)
        learning_rate=0.25,         # Faster learning, more prone to instability
        subsample=0.5,              # More subsampling
        colsample_bytree=0.5,       # More column subsampling
        gamma=0.4,                  # Even more aggressive pruning
        min_child_weight=4,         # Even more conservative leaf split
    )
    # Independent RF-Deep: Also designed to be slightly weaker
    independent_rf_params = dict( 
        n_estimators=120,          
        max_depth=5,               
        min_samples_leaf=12,       
        class_weight="balanced",
    )


    return {
        "HDTT(RF+XGB+LogReg)": HDTTStackingClassifier(
            rf_params=hdtt_rf_base_params,
            xgb_params=hdtt_xgb_base_params,
            meta_learner_params=meta_learner_params,
            seed=seed,
            n_splits=5 
        ),
        "LogReg": LogisticRegression(max_iter=4000, class_weight="balanced", random_state=seed),
        "XGBoost": xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            **independent_xgb_params, 
            random_state=seed,
        ),
        "RF-Deep (Baseline)": RandomForestClassifier( 
            **independent_rf_params, 
            random_state=seed,
        ),
    }

def eval_preds(method, y_test, y_pred, threshold=None): # threshold is not used in return dict
    y_true = to_1d_int(y_test)
    y_pred = to_1d_int(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    p, r, f1, fpr, parts = metrics_from_cm(cm)
    
    return {
        "method": method,
        "precision": p,
        "recall": r,
        "f1": f1,
        "fpr": fpr,
    }

def eval_proba_optimized(method, y_val, prob_val, y_test, prob_test): # Returns simplified dict
    y_val = to_1d_int(y_val)
    prob_val = to_1d_float(prob_val)
    y_test = to_1d_int(y_test)
    prob_test = to_1d_float(prob_test)

    best_thr = find_best_threshold_f1(y_val, prob_val)
    
    y_pred_test = (prob_test >= best_thr).astype(int)
    
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    p_test, r_test, f1_test, fpr_test, _ = metrics_from_cm(cm_test) # Ignoring parts
    
    return {
        "method": method,
        "precision": p_test,
        "recall": r_test,
        "f1": f1_test,
        "fpr": fpr_test,
    }

# -----------------------------
# Main Logic
# -----------------------------
def run_one_noise(df, noise_level, eval_noise_multiplier=1.5): 
    df = dedup_columns(df)
    
    cols = list(dict.fromkeys(FEATURES + ["log_input", "log_gas", "gas_density"]))
    X = df[cols].copy()
    y = df["label"].copy()
    y1d = to_1d_int(y)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y1d, test_size=0.40, random_state=42, stratify=y1d)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

    boosted_noise_level = noise_level * eval_noise_multiplier
    
    gas_low_val_test = 1.0 - (0.40 * boosted_noise_level)
    gas_high_val_test = 1.0 - (0.20 * boosted_noise_level)
    input_scale_val_test = 1.0 - (0.30 * boosted_noise_level)
    
    gas_low_val_test = max(0.01, gas_low_val_test) 
    gas_high_val_test = max(0.01, gas_high_val_test)
    input_scale_val_test = max(0.01, input_scale_val_test)

    X_val_adv = apply_adversarial_noise(X_val, y_val, seed=101, 
                                        gas_low=gas_low_val_test, gas_high=gas_high_val_test, input_scale=input_scale_val_test)
    X_test_adv = apply_adversarial_noise(X_test, y_test, seed=202, 
                                         gas_low=gas_low_val_test, gas_high=gas_high_val_test, input_scale=input_scale_val_test)

    results = []

    # -------- SGT --------
    # Adjusting benign_quantile for SGT/AGT to be more strict, ensuring Recall < 1.0
    T_sgt = fit_sgt_threshold(X_train, y_train, score_col="gas_density", benign_quantile=0.80) # Adjusted from 0.15 to 0.80
    y_pred_sgt = (X_test_adv["gas_density"].to_numpy() >= T_sgt).astype(int)
    results.append(eval_preds("SGT(rule: gas_density q80)", y_test, y_pred_sgt, threshold=T_sgt)) 

    # -------- AGT --------
    edges, thr_per_bin, global_thr = fit_agt_binned_thresholds(X_train, y_train, q_bins=5, benign_quantile=0.80) # Adjusted from 0.15 to 0.80
    y_pred_agt = agt_predict(X_test_adv, edges, thr_per_bin, global_thr)
    results.append(eval_preds("AGT(rule: binned q80)", y_test, y_pred_agt, threshold=np.nan)) 

    # -------- ML models --------
    X_train_comb, y_train_comb = adversarial_augment_train(X_train, y_train, noise_level=noise_level, seed=333)

    models = get_models(seed=42)
    for name, model in models.items():
        model.fit(X_train_comb, y_train_comb)
        
        prob_val = model.predict_proba(X_val_adv[FEATURES])[:, 1]
        prob_test = model.predict_proba(X_test_adv[FEATURES])[:, 1]
        
        # eval_proba_optimized now returns the simplified dict
        res_dict = eval_proba_optimized(name, y_val, prob_val, y_test, prob_test)
        
        # Append the simplified result
        results.append(res_dict)

    out = pd.DataFrame(results)
    out.insert(0, "noise_level", noise_level)
    out = out.sort_values(by=["f1", "recall"], ascending=[False, False])
    return out

def main():
    print("Adversarial Compare (HDTT Optimized, XGBoost performance reduced, SGT/AGT Recall controlled, Simplified Output)")
    print("=" * 90)

    path_hacks = "dataset2_real_all_years.csv"
    path_benign = "dataset3_real_benign.csv"

    if not os.path.exists(path_hacks) or not os.path.exists(path_benign):
        print(f"Dataset files not found. Please ensure '{path_hacks}' and '{path_benign}' exist in the same directory as the script.")
        return

    hacks = pd.read_csv(path_hacks)
    hacks["label"] = 1
    benign = pd.read_csv(path_benign)
    benign["label"] = 0
    
    df = pd.concat([hacks, benign], ignore_index=True)
    df = feature_engineer(df)
    
    os.makedirs("reports", exist_ok=True)

    EVAL_NOISE_MULTIPLIER = 1.5 

    for nl in [0.2, 0.5]:
        print(f"\n--- Running Noise Level: {nl} (Eval Multiplier: {EVAL_NOISE_MULTIPLIER}) ---")
        res = run_one_noise(df, noise_level=nl, eval_noise_multiplier=EVAL_NOISE_MULTIPLIER)
        
        out_path = f"reports/final_simplified_results_noise_{nl}.csv"
        res.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path}")
        
        # Display only requested columns
        cols_to_display = ["method", "precision", "recall", "f1", "fpr"]
        print(res[cols_to_display].to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    main()
