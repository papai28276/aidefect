"""Train tabular and classical image baselines with train-only preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             precision_recall_curve, roc_curve, classification_report)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.svm import SVC


def _binary_labels(values: pd.Series, defect_value=None) -> np.ndarray:
    """Map a selected fail value to 1, the other class to 0."""
    clean = values.dropna()
    classes = list(pd.unique(clean))
    if not classes:
        raise ValueError("The selected target column contains no non-missing labels")
    if defect_value is not None and defect_value not in classes:
        raise ValueError(f"Selected defect value {defect_value!r} is not present in the target column")
    positive = defect_value if defect_value is not None else classes[-1]
    return (values == positive).astype(int).to_numpy()


def _stratified_split(labels, test_size=.25, seed=42, min_train_per_class=2):
    """Return safe stratified train/test indices, including small datasets."""
    y=np.asarray(labels)
    classes,counts=np.unique(y,return_counts=True)
    if len(classes)<2:
        raise ValueError("Training requires at least two target classes")
    if counts.min()<min_train_per_class:
        raise ValueError(f"Each class needs at least {min_train_per_class} rows; the smallest class has {counts.min()}")
    if counts.min()<min_train_per_class+1:
        raise ValueError(f"Each class needs at least {min_train_per_class + 1} rows to keep examples in both train and test splits")
    n=len(y)
    if isinstance(test_size,float):
        if not 0<test_size<1:
            raise ValueError("Test split must be between 0 and 1")
        test_count=max(len(classes),int(np.ceil(n*test_size)))
    else:
        test_count=max(len(classes),int(test_size))
    train_count=n-test_count
    if train_count<len(classes)*min_train_per_class:
        raise ValueError("Not enough rows for the selected test split; add more examples or reduce the test split")
    return train_test_split(np.arange(n),test_size=test_count,stratify=y,random_state=seed)


def _metrics(y, prob, threshold=.5):
    pred = (prob >= threshold).astype(int)
    result = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "accuracy": accuracy_score(y, pred),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "report": classification_report(y, pred, output_dict=True, zero_division=0),
        "threshold": float(threshold),
    }
    result["pr_auc"] = average_precision_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    result["roc_auc"] = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    result["roc_curve"] = roc_curve(y, prob) if len(np.unique(y)) == 2 else None
    result["pr_curve"] = precision_recall_curve(y, prob) if len(np.unique(y)) == 2 else None
    return result


def train_tabular(frame: pd.DataFrame, target: str, defect_value=None, timestamp=None,
                  id_column=None, test_size=.25, seed=42, missed_cost=1000,
                  false_alarm_cost=100, quick=False) -> dict:
    """Compare tabular classifiers; quick mode minimizes fits for interactive use."""
    if target not in frame:
        raise ValueError("Target column was not found")
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    y = _binary_labels(frame[target], defect_value)
    drop = [target] + [c for c in (timestamp, id_column) if c and c in frame]
    X = frame.drop(columns=drop).copy()
    X = X.select_dtypes(include=["number", "bool"])
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns remain after excluding target, timestamp and ID")
    # Remove >80% missing and near-constant features before split; these rules use X only.
    X = X.loc[:, (X.isna().mean() <= .8)]
    X = X.loc[:, X.nunique(dropna=True) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable features remain after missingness and variance filters")
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    if stratify is None:
        raise ValueError("At least two rows per pass/fail class are required for a stratified evaluation")
    train_i, test_i = _stratified_split(y,test_size=test_size,seed=seed)
    X_train, X_test = X.iloc[train_i], X.iloc[test_i]
    y_train, y_test = y[train_i], y[test_i]
    pre = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True)), ("scale", StandardScaler())])
    models = {
        "Logistic regression": Pipeline([("prep", pre), ("model", __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(class_weight="balanced", max_iter=500 if quick else 1000, random_state=seed))]),
        "Random forest": Pipeline([("prep", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", RandomForestClassifier(n_estimators=64 if quick else 120, class_weight="balanced_subsample", min_samples_leaf=2, random_state=seed, n_jobs=1))]),
    }
    if not quick:
        models["HistGradientBoosting"] = Pipeline([("prep", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=.08, l2_regularization=1.0, class_weight="balanced", random_state=seed))])
    outputs, predictions, oof_predictions, calibration_predictions = {}, {}, {}, {}
    if quick:
        # A single training-only calibration slice needs one calibration fit plus one final
        # fit per model, instead of three CV fits plus a final fit.
        fit_i, calibration_i = _stratified_split(
            y_train, test_size=.2, seed=seed + 1, min_train_per_class=1
        )
    for name, model in models.items():
        if quick:
            calibration_model = clone(model)
            calibration_model.fit(X_train.iloc[fit_i], y_train[fit_i])
            threshold_prob = calibration_model.predict_proba(X_train.iloc[calibration_i])[:, 1]
            threshold_y = y_train[calibration_i]
            calibration_predictions[name] = pd.DataFrame({"actual": threshold_y, "probability": threshold_prob})
        else:
            cv_splits = min(3, int(np.bincount(y_train).min()))
            threshold_prob = cross_val_predict(
                model, X_train, y_train,
                cv=StratifiedKFold(cv_splits, shuffle=True, random_state=seed),
                method="predict_proba", n_jobs=1,
            )[:, 1]
            threshold_y = y_train
        oof_predictions[name] = pd.DataFrame({"actual": threshold_y, "probability": threshold_prob})
        candidates = np.unique(np.r_[0.01, threshold_prob, .99])
        costs = [(((threshold_prob < t) & (threshold_y == 1)).sum() * missed_cost + ((threshold_prob >= t) & (threshold_y == 0)).sum() * false_alarm_cost) for t in candidates]
        threshold = float(candidates[int(np.argmin(costs))])
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        missed = int(((prob < threshold) & (y_test == 1)).sum())
        false_alarm = int(((prob >= threshold) & (y_test == 0)).sum())
        outputs[name] = {**_metrics(y_test, prob, threshold), "missed_defects": missed,
                         "false_alarms": false_alarm,
                         "expected_test_cost": float(missed * missed_cost + false_alarm * false_alarm_cost)}
        predictions[name] = pd.DataFrame({"row_index": test_i, "actual": y_test, "probability": prob,
                                          "predicted": (prob >= threshold).astype(int), "threshold": threshold})
    if not quick:
        calibration_predictions = oof_predictions
    return {"models": outputs, "predictions": predictions,
            "calibration_predictions": calibration_predictions, "oof_predictions": oof_predictions,
            "feature_names": list(X.columns), "X_test": X_test, "y_test": y_test,
            "trained": models, "positive_value": defect_value, "train_indices": train_i, "test_indices": test_i}


def _hog_feature(image: Image.Image, pixels_per_cell=(16, 16)) -> np.ndarray:
    """Small dependency-free HOG descriptor for a 128px grayscale image."""
    image = image.convert("L").resize((128, 128))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    gy, gx = np.gradient(pixels)
    magnitude = np.hypot(gx, gy)
    orientation = np.mod(np.arctan2(gy, gx), np.pi)
    cell = pixels_per_cell[0]
    hist = np.zeros((128 // cell, 128 // cell, 9), dtype=np.float32)
    bin_pos = orientation * (9 / np.pi)
    lower = np.floor(bin_pos).astype(int) % 9
    frac = bin_pos - np.floor(bin_pos)
    for cy in range(128 // cell):
        for cx in range(128 // cell):
            sl = np.s_[cy * cell:(cy + 1) * cell, cx * cell:(cx + 1) * cell]
            m, b, f = magnitude[sl].ravel(), lower[sl].ravel(), frac[sl].ravel()
            hist[cy, cx] = np.bincount(b, weights=m * (1 - f), minlength=9) + np.bincount((b + 1) % 9, weights=m * f, minlength=9)
    blocks = []
    for y in range(hist.shape[0] - 1):
        for x in range(hist.shape[1] - 1):
            block = hist[y:y + 2, x:x + 2].ravel()
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            block = np.minimum(block, .2)
            block /= np.sqrt(np.dot(block, block) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks)


def train_images(images: list[Image.Image], labels: list[str], seed=42) -> dict:
    """Train CPU-only HOG + RBF SVM image classifier; no pretrained weights required."""
    if len(images)!=len(labels) or not images:
        raise ValueError("The image and label lists must have the same non-zero length")
    if len(set(labels)) < 2 or min(pd.Series(labels).value_counts()) < 4:
        raise ValueError("Image training needs at least two classes with four images per class")
    X = np.vstack([_hog_feature(im) for im in images])
    y = np.asarray(labels)
    train_i, test_i = _stratified_split(y,test_size=.25,seed=seed,min_train_per_class=3)
    model = Pipeline([("scale", StandardScaler()), ("model", CalibratedClassifierCV(SVC(C=5, class_weight="balanced", random_state=seed), cv=2, ensemble=False))])
    model.fit(X[train_i], y[train_i])
    pred = model.predict(X[test_i]); prob = model.predict_proba(X[test_i])
    return {"model": model, "X": X, "y": y, "train_indices": train_i, "test_indices": test_i,
            "predictions": pd.DataFrame({"actual": y[test_i], "predicted": pred}),
            "report": classification_report(y[test_i], pred, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y[test_i], pred, labels=model.classes_).tolist(),
            "classes": list(model.classes_), "test_probabilities": prob}


def permutation_importances(result: dict, model_name="Random forest", repeats=3, seed=42) -> pd.DataFrame:
    """Rank features on the held-out split with permutation importance."""
    model = result["trained"][model_name]
    X, y = result["X_test"], result["y_test"]
    # Keep process count at one for stable Windows ARM64 / joblib behavior.
    imp = permutation_importance(model, X, y, scoring="average_precision", n_repeats=min(repeats, 3), random_state=seed, n_jobs=1, max_samples=.5)
    return pd.DataFrame({"feature": X.columns, "importance_mean": imp.importances_mean, "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False)


def predict_images(model, images: list[Image.Image], filenames: list[str]) -> pd.DataFrame:
    """Score unlabeled images with a fitted HOG/SVM model."""
    if not images or len(images)!=len(filenames):
        raise ValueError("Provide at least one image and exactly one filename per image")
    features=np.vstack([_hog_feature(image) for image in images])
    probabilities=model.predict_proba(features)
    output=pd.DataFrame({"filename":filenames,"predicted_class":model.predict(features),"confidence":probabilities.max(axis=1)})
    for j,label in enumerate(model.classes_): output[f"probability_{label}"]=probabilities[:,j]
    return output

