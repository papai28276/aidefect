"""Validated, leakage-safe input readers for common production quality files."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import io
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"


def download_secom(destination: Path) -> tuple[Path, Path]:
    """Download and unpack the original UCI SECOM files; returns data and labels."""
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "secom.zip"
    if not archive.exists():
        urllib.request.urlretrieve(SECOM_URL, archive)
    with ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if Path(member).name in {"secom.data", "secom_labels.data"}:
                target = destination / Path(member).name
                if not target.exists():
                    target.write_bytes(bundle.read(member))
    data, labels = destination / "secom.data", destination / "secom_labels.data"
    if not data.exists() or not labels.exists():
        raise ValueError("UCI archive did not contain secom.data and secom_labels.data")
    return data, labels


def read_secom(data_file, labels_file) -> pd.DataFrame:
    """Load raw SECOM whitespace-separated files. Labels -1 (pass), +1 (fail)."""
    x = pd.read_csv(data_file, sep=r"\s+", header=None, na_values="NaN")
    labels = pd.read_csv(labels_file, sep=r"\s+", header=None, usecols=[0, 1])
    if len(x) != len(labels):
        raise ValueError(f"SECOM row mismatch: {len(x)} sensors vs {len(labels)} labels")
    frame = x.copy()
    frame.columns = [f"sensor_{i:03d}" for i in range(x.shape[1])]
    frame.insert(0, "timestamp", pd.to_datetime(labels.iloc[:, 1], errors="coerce", dayfirst=True))
    frame["target"] = (labels.iloc[:, 0].astype(int) == 1).astype(int)
    frame.insert(0, "row_id", np.arange(len(frame)))
    return frame


def read_tabular_upload(file, labels_file=None) -> pd.DataFrame:
    """Read CSV/XLSX or raw SECOM sensor data (with separate labels when supplied)."""
    name = str(getattr(file, "name", file)).lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        if labels_file is not None:
            raise ValueError("A separate SECOM labels file can only be paired with raw whitespace-separated sensor data.")
        return pd.read_excel(file)
    if labels_file is not None:
        x = pd.read_csv(file, sep=r"\s+", header=None, na_values="NaN")
        y = pd.read_csv(labels_file, sep=r"\s+", header=None, usecols=[0, 1])
        if len(x) != len(y):
            raise ValueError("Sensor and label files have different row counts")
        if x.shape[1] == 0 or y.shape[1] < 2:
            raise ValueError("The uploaded SECOM sensor or labels file is empty or malformed")
        x.columns = [f"sensor_{i:03d}" for i in range(x.shape[1])]
        x.insert(0, "timestamp", pd.to_datetime(y.iloc[:, 1], errors="coerce", dayfirst=True))
        x.insert(0, "target", y.iloc[:, 0].astype(int))
        return x
    if name.endswith((".data", ".txt")):
        raise ValueError("Raw SECOM sensor data requires its matching secom_labels.data file.")
    return pd.read_csv(file, sep=None, engine="python")


def load_image_uploads(files) -> tuple[list[Image.Image], list[str], list[str]]:
    """Decode uploaded images. ZIP labels come from immediate class folders."""
    images, labels, names = [], [], []
    uploaded_bytes=0; expanded_bytes=0; decoded_pixels=0
    for uploaded in files:
        filename = getattr(uploaded, "name", "image")
        payload = uploaded.getvalue() if hasattr(uploaded, "getvalue") else Path(uploaded).read_bytes()
        uploaded_bytes+=len(payload)
        if uploaded_bytes>512*1024*1024:
            raise ValueError("The combined image uploads exceed the 512 MB session limit.")
        members = []
        if filename.lower().endswith(".zip"):
            with ZipFile(io.BytesIO(payload)) as archive:
                entries=archive.infolist()
                if len(entries)>5000:
                    raise ValueError("The image archive contains too many entries (maximum 5,000).")
                images_in_archive=[entry for entry in entries if not entry.is_dir() and entry.filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
                expanded_bytes+=sum(entry.file_size for entry in images_in_archive)
                if expanded_bytes>512*1024*1024:
                    raise ValueError("The uncompressed image archive exceeds the 512 MB session limit.")
                for entry in images_in_archive:
                    members.append((entry.filename, archive.read(entry)))
        else:
            members.append((filename, payload))
        for member_name, data in members:
            try:
                with Image.open(io.BytesIO(data)) as source:
                    if source.width * source.height > 50_000_000:
                        raise ValueError(f"Image {member_name!r} exceeds the 50-megapixel limit.")
                    decoded_pixels+=source.width*source.height
                    if decoded_pixels>500_000_000:
                        raise ValueError("The combined images exceed the 500-megapixel session limit.")
                    image = source.convert("L")
            except (UnidentifiedImageError, OSError) as exc:
                raise ValueError(f"Could not decode image: {member_name}") from exc
            parts = Path(member_name).parts
            label = parts[-2] if len(parts) > 1 else "unlabeled"
            images.append(image)
            labels.append(label)
            names.append(Path(member_name).name)
            if len(images)>2000:
                raise ValueError("The upload contains more than 2,000 images. Split it into smaller class-folder ZIPs.")
    if not images:
        raise ValueError("No supported images found. Upload PNG, JPG, BMP, or a ZIP of class folders.")
    return images, labels, names


def validate_frame(frame: pd.DataFrame, target: str, min_rows: int = 20) -> list[str]:
    """Return human-readable input validation issues."""
    errors = []
    if frame.empty:
        errors.append("The uploaded file has no rows.")
    if len(frame) < min_rows:
        errors.append(f"At least {min_rows} rows are recommended; found {len(frame)}.")
    if target not in frame.columns:
        errors.append("Choose a target column before training.")
    elif frame[target].nunique(dropna=True) < 2:
        errors.append("The target must contain at least two classes.")
    numeric = frame.drop(columns=[target], errors="ignore").select_dtypes(include=np.number)
    if numeric.empty:
        errors.append("No numeric process features were found.")
    elif (numeric.isna().mean() > .8).all():
        errors.append("All numeric features have more than 80% missing values.")
    return errors
