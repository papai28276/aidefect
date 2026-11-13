"""Multimodal evaluation on explicitly verified process/image pairs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from .pipeline import _hog_feature, _stratified_split

def train_real_pairing(frame, target, images, image_names, pair_map, id_column=None, defect_value=None, seed=42, timestamp=None):
    """Train simple fusion baselines only on explicit row_id/image_filename joins."""
    if not {"row_id", "image_filename"}.issubset(pair_map.columns):
        raise ValueError("Pairing map must contain row_id and image_filename columns")
    if len(images) != len(image_names):
        raise ValueError("The image list and image-name list have different lengths")
    rows = frame.copy()
    if target not in rows:
        raise ValueError(f"Target column {target!r} was not found in the process table")
    if pair_map[["row_id", "image_filename"]].isna().any().any():
        raise ValueError("Pairing map contains blank row_id or image_filename values")
    rows["_join_id"] = rows[id_column].astype(str) if id_column and id_column in rows else rows.index.astype(str)
    if rows["_join_id"].duplicated().any():
        raise ValueError("Process row IDs must be unique to make an unambiguous pairing map")
    mapping = pair_map.copy()
    mapping["row_id"] = mapping.row_id.astype(str)
    mapping["image_filename"] = mapping.image_filename.map(lambda x: str(x).replace("\\", "/").split("/")[-1])
    if mapping["row_id"].duplicated().any():
        raise ValueError("Each process row may appear only once in the pairing map")
    if mapping["image_filename"].duplicated().any():
        raise ValueError("Each image may appear only once in the pairing map")
    name_to_index = {str(n).replace("\\", "/").split("/")[-1]: i for i, n in enumerate(image_names)}
    unknown_rows=set(mapping.row_id)-set(rows._join_id)
    unknown_images=set(mapping.image_filename)-set(name_to_index)
    if unknown_rows:
        raise ValueError(f"Pairing map contains {len(unknown_rows)} row_id value(s) that are not in the process table")
    if unknown_images:
        example=next(iter(unknown_images))
        raise ValueError(f"Pairing map image filename {example!r} was not found in the uploaded image set")
    joined = rows.merge(mapping, left_on="_join_id", right_on="row_id", how="inner", validate="one_to_one")
    if len(joined) < 12:
        raise ValueError(f"Only {len(joined)} valid row/image pairs remain; at least 12 are required")
    raw_y = joined[target]
    if raw_y.isna().any():
        raise ValueError("Paired process rows contain missing target labels")
    classes=list(pd.unique(raw_y))
    if len(classes)!=2:
        raise ValueError("Real fusion currently requires a binary pass/fail target")
    if defect_value is not None and defect_value not in classes:
        raise ValueError(f"Selected defect value {defect_value!r} is not present in the paired rows")
    positive = defect_value if defect_value is not None else classes[-1]
    y = (raw_y == positive).astype(int).to_numpy()
    if len(np.unique(y)) < 2 or np.min(np.bincount(y)) < 2:
        raise ValueError("Paired rows need at least two pass and two fail examples")
    image_idx = np.asarray([name_to_index[n] for n in joined.image_filename])
    image_x = np.vstack([_hog_feature(images[i]) for i in image_idx])
    tab_x = joined.drop(columns=[target, "_join_id", "row_id", "image_filename", id_column, timestamp], errors="ignore").select_dtypes(include=["number", "bool"])
    if tab_x.shape[1] == 0:
        raise ValueError("No numeric process features remain after excluding target, timestamp, and row ID")
    train_i, test_i = _stratified_split(y,test_size=.25,seed=seed)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(tab_x.iloc[train_i])
    tx0, tx1 = imp.transform(tab_x.iloc[train_i]), imp.transform(tab_x.iloc[test_i])
    ts = StandardScaler().fit(tx0); tx0, tx1 = ts.transform(tx0), ts.transform(tx1)
    ims = StandardScaler().fit(image_x[train_i]); ix0, ix1 = ims.transform(image_x[train_i]), ims.transform(image_x[test_i])
    def fitprob(a, b):
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed).fit(a, y[train_i])
        return model.predict_proba(b)[:, 1]
    tab = fitprob(tx0, tx1); img = fitprob(ix0, ix1)
    early = fitprob(np.column_stack([tx0, ix0]), np.column_stack([tx1, ix1])); late = (tab + img) / 2
    def metrics(p): return {"pr_auc": float(average_precision_score(y[test_i], p)), "roc_auc": float(roc_auc_score(y[test_i], p)), "f1@0.5": float(f1_score(y[test_i], p >= .5))}
    return {"metrics": {"tabular only": metrics(tab), "image only": metrics(img), "early fusion": metrics(early), "late fusion": metrics(late)},
            "predictions": {"tabular only": tab, "image only": img, "early fusion": early, "late fusion": late},
            "y_test": y[test_i], "row_ids": joined._join_id.iloc[test_i].tolist(),
            "mapping": joined[["_join_id", "image_filename"]].rename(columns={"_join_id": "row_id"}),
            "real_pairing": True}
