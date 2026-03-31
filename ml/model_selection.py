"""Model registry: defines batch candidate models."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_engine.imputation import ArbitraryNumberImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------------------------------------------------------
# Batch models
# ---------------------------------------------------------------------------

def get_batch_models(skip_logreg=False, oversampling=False):
    """Return dict of {name: sklearn Pipeline} for batch candidate models.

    Args:
        skip_logreg: Skip LogisticRegression candidate.
        oversampling: Use ADASYN oversampling for boosting models instead of
            scale_pos_weight. Recommended for highly imbalanced targets
            (champion, team) where the minority class is too small.
    """
    candidates = {}

    if not skip_logreg:
        candidates["LogisticRegression"] = Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                solver="saga",
                max_iter=3500,
                tol=1e-3,
                random_state=42,
                class_weight="balanced",
            )),
        ])

    lgbm = LGBMClassifier(
        n_estimators=200,
        num_leaves=10,
        max_depth=3,
        learning_rate=0.01,
        min_child_samples=200,
        reg_alpha=5.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        device="gpu",
        verbosity=-1,
    )
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        min_child_weight=200,
        reg_alpha=5.0,
        reg_lambda=5.0,
        gamma=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
        device="cuda",
        tree_method="hist",
    )

    if oversampling:
        sampler = BorderlineSMOTE(random_state=42, k_neighbors=3)
        candidates["LightGBM"] = ImbPipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("sampler", sampler),
            ("model", lgbm),
        ])
        candidates["XGBoost"] = ImbPipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("sampler", BorderlineSMOTE(random_state=42, k_neighbors=3)),
            ("model", xgb),
        ])
    else:
        candidates["LightGBM"] = Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", lgbm),
        ])
        candidates["XGBoost"] = Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", xgb),
        ])

    candidates["BalancedRandomForest"] = ImbPipeline([
        ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
        ("model", BalancedRandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=50,
            max_samples=0.7,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    return candidates
