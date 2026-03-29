"""Model registry: defines batch candidate models."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import ArbitraryNumberImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------------------------------------------------------
# Batch models
# ---------------------------------------------------------------------------

def get_batch_models():
    """Return dict of {name: sklearn Pipeline} for batch candidate models.

    Candidates: LogisticRegression, LightGBM, BalancedRandomForest, XGBoost.
    LogisticRegression always uses class_weight='balanced'; boosting models
    do not — they handle imbalance through their own adaptive reweighting.
    """
    candidates = {
        "LogisticRegression": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                penalty="l1",
                solver="saga",
                max_iter=10000,
                tol=1e-3,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                min_child_samples=20,
                random_state=42,
                n_jobs=-1,
                device="gpu",
                verbosity=-1,
            )),
        ]),
        "BalancedRandomForest": ImbPipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", BalancedRandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                min_samples_leaf=50,
                max_samples=0.7,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=10,
                random_state=42,
                n_jobs=-1,
                eval_metric="logloss",
                verbosity=0,
                device="cuda",
                tree_method="hist",
            )),
        ]),
    }

    return candidates
