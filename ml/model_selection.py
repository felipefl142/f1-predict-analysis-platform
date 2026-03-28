"""Model registry: defines batch and online candidate models."""

import math

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
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
                solver="saga",
                max_iter=1000,
                tol=1e-3,
                random_state=42,
                class_weight="balanced",
            )),
        ]),
        "LightGBM": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                device="gpu",
                verbosity=-1,
            )),
        ]),
        "BalancedRandomForest": ImbPipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", BalancedRandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=50,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", ArbitraryNumberImputer(arbitrary_number=-10000)),
            ("model", XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
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


# ---------------------------------------------------------------------------
# Online models
# ---------------------------------------------------------------------------

def get_online_models():
    """Return dict of {name: model_config} for online learning candidates.

    Candidates:
      - SGDClassifier (hinge loss) — sklearn partial_fit
      - OnlineLogisticRegression (log_loss) — sklearn partial_fit
      - StreamingHoeffdingTree — pure-Python streaming decision tree
      - StreamingAdaptiveForest — ensemble of streaming trees
      - StreamingLogisticRegression — pure-Python SGD logistic regression

    Each entry has 'type' ('sklearn' or 'streaming') and the model/factory.
    SGD-based models always use class_weight='balanced'.
    """
    candidates = {
        "SGDClassifier": {
            "type": "sklearn",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(
                    loss="modified_huber",
                    random_state=42,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-3,
                )),
            ]),
        },
        "OnlineLogisticRegression": {
            "type": "sklearn",
            "model": Pipeline([
                ("scaler", StandardScaler()),
                ("model", SGDClassifier(
                    loss="log_loss",
                    random_state=42,
                    class_weight="balanced",
                    max_iter=5000,
                    tol=1e-3,
                )),
            ]),
        },
        "StreamingLogisticRegression": {
            "type": "streaming",
            "model_factory": lambda: StreamingLogisticRegression(lr=0.01),
        },
        "StreamingHoeffdingTree": {
            "type": "streaming",
            "model_factory": lambda: StreamingHoeffdingTree(max_depth=10, grace_period=50),
        },
        "StreamingAdaptiveForest": {
            "type": "streaming",
            "model_factory": lambda: StreamingAdaptiveForest(n_trees=10, max_depth=8),
        },
    }

    return candidates


# ---------------------------------------------------------------------------
# Pure-Python streaming models (replaces River dependency)
# ---------------------------------------------------------------------------

class StreamingLogisticRegression:
    """Online logistic regression via SGD, one sample at a time."""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.weights = None
        self.bias = 0.0
        self._mean = None
        self._var = None
        self._count = 0

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _update_stats(self, x):
        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._var = np.ones_like(x)
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._var += delta * (x - self._mean)

    def _scale(self, x):
        if self._count < 2:
            return x
        std = np.sqrt(self._var / max(self._count - 1, 1)) + 1e-8
        return (x - self._mean) / std

    def learn_one(self, x_dict, y):
        x = np.array(list(x_dict.values()), dtype=np.float64)
        np.nan_to_num(x, nan=-10000, copy=False)
        self._update_stats(x)
        x_scaled = self._scale(x)

        if self.weights is None:
            self.weights = np.zeros(len(x))

        pred = self._sigmoid(np.dot(self.weights, x_scaled) + self.bias)
        error = pred - y
        self.weights -= self.lr * error * x_scaled
        self.bias -= self.lr * error

    def predict_proba_one(self, x_dict):
        x = np.array(list(x_dict.values()), dtype=np.float64)
        np.nan_to_num(x, nan=-10000, copy=False)
        x_scaled = self._scale(x)

        if self.weights is None:
            return {0: 0.5, 1: 0.5}

        p = self._sigmoid(np.dot(self.weights, x_scaled) + self.bias)
        return {0: 1 - p, 1: p}


class _StreamingTreeNode:
    """A node in a Hoeffding tree."""

    def __init__(self, depth=0, max_depth=10):
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = True
        self.count = 0
        self.class_counts = {0: 0, 1: 0}
        # Split info
        self.split_feature = None
        self.split_value = None
        self.left = None
        self.right = None
        # Stats for finding splits (per-feature running stats)
        self._feature_stats = {}

    def _update_feature_stats(self, x, y):
        for i, val in enumerate(x):
            if i not in self._feature_stats:
                self._feature_stats[i] = {
                    "sum_0": 0.0, "sum_1": 0.0, "sumsq_0": 0.0, "sumsq_1": 0.0,
                    "count_0": 0, "count_1": 0,
                }
            s = self._feature_stats[i]
            key = str(int(y))
            s[f"sum_{key}"] += val
            s[f"sumsq_{key}"] += val * val
            s[f"count_{key}"] += 1

    def predict_proba(self):
        total = self.class_counts[0] + self.class_counts[1]
        if total == 0:
            return {0: 0.5, 1: 0.5}
        return {0: self.class_counts[0] / total, 1: self.class_counts[1] / total}


class StreamingHoeffdingTree:
    """Simplified Hoeffding Tree for binary classification, streaming one sample at a time."""

    def __init__(self, max_depth=10, grace_period=50, delta=1e-7):
        self.max_depth = max_depth
        self.grace_period = grace_period
        self.delta = delta
        self.root = _StreamingTreeNode(depth=0, max_depth=max_depth)
        self._mean = None
        self._var = None
        self._count = 0

    def _update_stats(self, x):
        if self._mean is None:
            self._mean = np.zeros_like(x)
            self._var = np.ones_like(x)
        self._count += 1
        delta = x - self._mean
        self._mean += delta / self._count
        self._var += delta * (x - self._mean)

    def _scale(self, x):
        if self._count < 2:
            return x
        std = np.sqrt(self._var / max(self._count - 1, 1)) + 1e-8
        return (x - self._mean) / std

    def _traverse(self, node, x):
        if node.is_leaf:
            return node
        if x[node.split_feature] <= node.split_value:
            return self._traverse(node.left, x)
        return self._traverse(node.right, x)

    def _try_split(self, node, x):
        if node.depth >= self.max_depth:
            return
        if node.count < self.grace_period:
            return

        best_gain = -1
        best_feature = None
        best_value = None

        total = node.class_counts[0] + node.class_counts[1]
        if total == 0:
            return
        p = node.class_counts[1] / total
        parent_entropy = -p * math.log2(p + 1e-10) - (1 - p) * math.log2(1 - p + 1e-10)

        for feat_idx, stats in node._feature_stats.items():
            c0, c1 = stats["count_0"], stats["count_1"]
            if c0 == 0 or c1 == 0:
                continue
            mean_0 = stats["sum_0"] / max(c0, 1)
            mean_1 = stats["sum_1"] / max(c1, 1)
            split_val = (mean_0 + mean_1) / 2

            left_0 = c0 // 2
            left_1 = c1 // 2
            right_0 = c0 - left_0
            right_1 = c1 - left_1

            left_total = left_0 + left_1
            right_total = right_0 + right_1

            if left_total == 0 or right_total == 0:
                continue

            pl = left_1 / left_total
            pr = right_1 / right_total
            left_ent = -pl * math.log2(pl + 1e-10) - (1 - pl) * math.log2(1 - pl + 1e-10)
            right_ent = -pr * math.log2(pr + 1e-10) - (1 - pr) * math.log2(1 - pr + 1e-10)

            gain = parent_entropy - (left_total / total) * left_ent - (right_total / total) * right_ent

            if gain > best_gain:
                best_gain = gain
                best_feature = feat_idx
                best_value = split_val

        # Hoeffding bound
        epsilon = math.sqrt(math.log(1 / self.delta) / (2 * node.count))

        if best_gain > epsilon and best_feature is not None:
            node.is_leaf = False
            node.split_feature = best_feature
            node.split_value = best_value
            node.left = _StreamingTreeNode(depth=node.depth + 1, max_depth=self.max_depth)
            node.right = _StreamingTreeNode(depth=node.depth + 1, max_depth=self.max_depth)

    def learn_one(self, x_dict, y):
        x = np.array(list(x_dict.values()), dtype=np.float64)
        np.nan_to_num(x, nan=-10000, copy=False)
        self._update_stats(x)
        x = self._scale(x)

        node = self._traverse(self.root, x)
        node.count += 1
        node.class_counts[int(y)] = node.class_counts.get(int(y), 0) + 1
        node._update_feature_stats(x, y)
        self._try_split(node, x)

    def predict_proba_one(self, x_dict):
        x = np.array(list(x_dict.values()), dtype=np.float64)
        np.nan_to_num(x, nan=-10000, copy=False)
        x = self._scale(x)
        node = self._traverse(self.root, x)
        return node.predict_proba()


class StreamingAdaptiveForest:
    """Ensemble of streaming Hoeffding trees (simplified Adaptive Random Forest)."""

    def __init__(self, n_trees=10, max_depth=8):
        self.trees = [
            StreamingHoeffdingTree(max_depth=max_depth, grace_period=30 + i * 10)
            for i in range(n_trees)
        ]

    def learn_one(self, x_dict, y):
        for tree in self.trees:
            # Poisson(1) bootstrap — each tree sees each sample ~63% of the time
            weight = np.random.poisson(1)
            for _ in range(weight):
                tree.learn_one(x_dict, y)

    def predict_proba_one(self, x_dict):
        probs = [t.predict_proba_one(x_dict) for t in self.trees]
        avg_0 = sum(p.get(0, 0.5) for p in probs) / len(probs)
        avg_1 = sum(p.get(1, 0.5) for p in probs) / len(probs)
        return {0: avg_0, 1: avg_1}
