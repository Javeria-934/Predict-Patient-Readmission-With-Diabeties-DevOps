"""
Model Architecture Definitions and Estimators.
Implements:
1. Logistic Regression (Baseline, balanced class weight)
2. Random Forest (Ensemble trees, balanced class weight)
3. XGBoost (Gradient boosting, scale_pos_weight derived from training set)
4. Deep Neural Network (PyTorch feed-forward DNN with multiple hidden layers,
   BatchNorm, ReLU, Dropout, Adam, and BCEWithLogitsLoss with pos_weight).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import RANDOM_STATE


# ==============================================================================
# Model D: Deep Neural Network (PyTorch Implementation)
# ==============================================================================
class PyTorchDNNModule(nn.Module):
    """
    Genuine multi-layer feed-forward Deep Neural Network.
    Architecture:
      Input (D) -> Linear(H1) -> BatchNorm1d -> ReLU -> Dropout(p1)
                -> Linear(H2) -> BatchNorm1d -> ReLU -> Dropout(p2)
                -> Linear(H3) -> BatchNorm1d -> ReLU -> Dropout(p3)
                -> Linear(1)
    """
    def __init__(self, in_features, hidden_layer_sizes=(128, 64, 32), dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        dropouts = [dropout_rate, dropout_rate * 0.75, dropout_rate * 0.5]

        for i, h_dim in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            p = dropouts[i] if i < len(dropouts) else 0.1
            layers.append(nn.Dropout(p))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class PyTorchDNNClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for PyTorch Deep Neural Network.
    Enables seamless use in cross-validation, pipelines, and evaluation routines.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        lr=0.001,
        weight_decay=1e-4,
        dropout_rate=0.2,
        batch_size=512,
        epochs=8,
        pos_weight=None,
        random_state=RANDOM_STATE,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _batch_iter(self, X, y=None, shuffle=True):
        n = X.shape[0]
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            if sp.issparse(X):
                batch_X = torch.tensor(X[batch_idx].toarray(), dtype=torch.float32)
            else:
                batch_X = torch.tensor(X[batch_idx], dtype=torch.float32)

            if y is not None:
                batch_y = torch.tensor(y[batch_idx], dtype=torch.float32)
                yield batch_X, batch_y
            else:
                yield batch_X

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        in_features = X.shape[1]
        self.model_ = PyTorchDNNModule(
            in_features=in_features,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout_rate=self.dropout_rate,
        )

        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], dtype=torch.float32)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model_.train()
        for epoch in range(self.epochs):
            for bx, by in self._batch_iter(X, y, shuffle=True):
                optimizer.zero_grad()
                logits = self.model_(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X):
        self.model_.eval()
        probs_list = []
        with torch.no_grad():
            for bx in self._batch_iter(X, shuffle=False):
                logits = self.model_(bx)
                p1 = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(p1)

        p1_all = np.concatenate(probs_list)
        p0_all = 1.0 - p1_all
        return np.column_stack([p0_all, p1_all])

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ==============================================================================
# Model Factory
# ==============================================================================
def get_model(model_name, y_train=None, weighted=False, **kwargs):
    """
    Factory function returning initialized model instances.
    When weighted=False (Primary Experiment), models use unweighted configurations
    matching the published paper's reported Table 1 metrics at threshold 0.5.
    When weighted=True (Secondary Sensitivity Experiment), models use cost-sensitive
    class weights to address the 8:1 class imbalance.
    """
    if model_name == "logistic_regression":
        defaults = {
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 500,
            "class_weight": "balanced" if weighted else None,
            "random_state": RANDOM_STATE,
        }
        defaults.update(kwargs)
        return LogisticRegression(**defaults)

    elif model_name == "random_forest":
        defaults = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "class_weight": "balanced" if weighted else None,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return RandomForestClassifier(**defaults)

    elif model_name == "xgboost":
        if weighted:
            if y_train is not None:
                spw = float((y_train == 0).sum() / (y_train == 1).sum())
            else:
                spw = 7.96
        else:
            spw = 1.0

        defaults = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return XGBClassifier(**defaults)

    elif model_name == "dnn":
        if weighted:
            if y_train is not None:
                pw = float((y_train == 0).sum() / (y_train == 1).sum())
            else:
                pw = 7.96
        else:
            pw = None

        defaults = {
            "hidden_layer_sizes": (128, 64, 32),
            "lr": 0.001,
            "weight_decay": 1e-4,
            "dropout_rate": 0.2,
            "batch_size": 512,
            "epochs": 8,
            "pos_weight": pw,
            "random_state": RANDOM_STATE,
        }
        defaults.update(kwargs)
        return PyTorchDNNClassifier(**defaults)

    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        if weighted:
            spw = float((y_train == 0).sum() / (y_train == 1).sum()) if y_train is not None else 7.96
        else:
            spw = 1.0

        defaults = {
            "n_estimators": 100,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "scale_pos_weight": spw,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(kwargs)
        return LGBMClassifier(**defaults)

    elif model_name == "catboost":
        from catboost import CatBoostClassifier
        if weighted:
            spw = float((y_train == 0).sum() / (y_train == 1).sum()) if y_train is not None else 7.96
        else:
            spw = 1.0

        defaults = {
            "iterations": 150,
            "learning_rate": 0.08,
            "depth": 5,
            "scale_pos_weight": spw,
            "random_state": RANDOM_STATE,
            "verbose": 0,
            "thread_count": -1,
        }
        defaults.update(kwargs)
        return CatBoostClassifier(**defaults)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
