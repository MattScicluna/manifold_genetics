"""
Input validation utilities for manifold-genetics CLI.

Provides reusable validators for CSV, JSON, and cross-file checks
with descriptive error messages.
"""

import difflib
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when CLI input validation fails.

    Inherits from ValueError so the existing cli.py error handler
    catches it and prints the message to stderr.
    """

    pass


def _fuzzy_match(target: str, candidates: List[str], n: int = 3) -> List[str]:
    """Return close matches for a potentially misspelled name."""
    return difflib.get_close_matches(target, candidates, n=n, cutoff=0.6)


def _read_csv_columns(path: Path) -> List[str]:
    """Read only the header row of a CSV to get column names."""
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def _read_sample_ids(path: Path) -> List[str]:
    """Read the sample_id column from a CSV file."""
    df = pd.read_csv(path, usecols=["sample_id"])
    return df["sample_id"].astype(str).tolist()


# ---------------------------------------------------------------------------
# Per-file validators
# ---------------------------------------------------------------------------


def validate_embedding_csv(path: Union[str, Path]) -> None:
    """Validate an embedding CSV has the expected format.

    Checks:
        - File exists
        - Has a 'sample_id' column
        - Has at least one 'dim_*' column
        - dim_* columns contain numeric data
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Embedding file not found: {path}")

    columns = _read_csv_columns(path)

    if "sample_id" not in columns:
        raise ValidationError(
            f"Embedding CSV is missing required 'sample_id' column.\n\n"
            f"  File: {path}\n"
            f"  Columns found: {columns}\n\n"
            f"  Expected format: sample_id, dim_1, dim_2, ..."
        )

    dim_cols = [c for c in columns if c.startswith("dim_")]
    if not dim_cols:
        raise ValidationError(
            f"Embedding CSV has no 'dim_*' columns.\n\n"
            f"  File: {path}\n"
            f"  Columns found: {columns}\n\n"
            f"  Expected columns like: sample_id, dim_1, dim_2, ..."
        )

    # Check that dim columns are numeric (sample a few rows)
    sample = pd.read_csv(path, nrows=5, usecols=dim_cols)
    non_numeric = [c for c in dim_cols if not pd.api.types.is_numeric_dtype(sample[c])]
    if non_numeric:
        raise ValidationError(
            f"Embedding CSV has non-numeric dim_* columns: {non_numeric}\n\n"
            f"  File: {path}\n"
            f"  These columns must contain numeric values."
        )


def validate_labels_csv(path: Union[str, Path]) -> None:
    """Validate a labels CSV has the expected format.

    Checks:
        - File exists
        - Has a 'sample_id' column
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Labels file not found: {path}")

    columns = _read_csv_columns(path)

    if "sample_id" not in columns:
        raise ValidationError(
            f"Labels CSV is missing required 'sample_id' column.\n\n"
            f"  File: {path}\n"
            f"  Columns found: {columns}\n\n"
            f"  The first column should be 'sample_id' containing unique sample identifiers."
        )


def validate_colormap_json(path: Union[str, Path]) -> None:
    """Validate a colormap JSON file has the expected structure.

    Expected format::

        {
            "ColumnName": {
                "LabelValue": "#RRGGBB",
                ...
            },
            ...
        }

    Checks:
        - File exists
        - Valid JSON
        - Top-level is a dict
        - Each value is a dict mapping label values to color strings
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Colormap file not found: {path}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValidationError(
            f"Invalid JSON in colormap file.\n\n"
            f"  File: {path}\n"
            f"  Parse error: {e}"
        )

    if not isinstance(data, dict):
        raise ValidationError(
            f"Colormap must be a JSON object (dict), got {type(data).__name__}.\n\n"
            f"  File: {path}\n"
            f'  Expected format: {{"ColumnName": {{"LabelValue": "#RRGGBB", ...}}}}'
        )

    for key, value in data.items():
        if not isinstance(value, dict):
            raise ValidationError(
                f"Colormap entry '{key}' must be a dict mapping label values to colors, "
                f"got {type(value).__name__}.\n\n"
                f"  File: {path}\n"
                f'  Expected format: {{"{key}": {{"value1": "#RRGGBB", ...}}}}'
            )


def validate_admixture_csv(
    q_prefix: Union[str, Path], k_values: List[int]
) -> None:
    """Validate admixture CSV files for each K value.

    Checks:
        - Each {q_prefix}.{k}.csv file exists
        - Each file has 'sample_id' and 'component_*' columns
        - Component columns are numeric
    """
    q_prefix = Path(q_prefix)
    missing_files = []

    for k in k_values:
        q_path = Path(f"{q_prefix}.{k}.csv")
        if not q_path.exists():
            missing_files.append(str(q_path))

    if missing_files:
        raise ValidationError(
            f"Admixture file(s) not found:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + f"\n\n  Expected file pattern: {q_prefix}.<K>.csv"
        )

    # Validate format of each file
    for k in k_values:
        q_path = Path(f"{q_prefix}.{k}.csv")
        columns = _read_csv_columns(q_path)

        if "sample_id" not in columns:
            raise ValidationError(
                f"Admixture CSV for K={k} is missing required 'sample_id' column.\n\n"
                f"  File: {q_path}\n"
                f"  Columns found: {columns}\n\n"
                f"  Expected format: sample_id, component_1, component_2, ..."
            )

        comp_cols = [c for c in columns if c.startswith("component_")]
        if not comp_cols:
            raise ValidationError(
                f"Admixture CSV for K={k} has no 'component_*' columns.\n\n"
                f"  File: {q_path}\n"
                f"  Columns found: {columns}\n\n"
                f"  Expected columns like: sample_id, component_1, component_2, ..."
            )

        sample = pd.read_csv(q_path, nrows=5, usecols=comp_cols)
        non_numeric = [
            c for c in comp_cols if not pd.api.types.is_numeric_dtype(sample[c])
        ]
        if non_numeric:
            raise ValidationError(
                f"Admixture CSV for K={k} has non-numeric component columns: {non_numeric}\n\n"
                f"  File: {q_path}\n"
                f"  Component columns must contain numeric values (ancestry proportions)."
            )


def validate_geographic_csv(
    path: Union[str, Path],
    longitude_col: str = "longitude",
    latitude_col: str = "latitude",
) -> None:
    """Validate a geographic coordinates CSV.

    Checks:
        - File exists
        - Has 'sample_id', longitude, and latitude columns
        - Coordinate columns are numeric
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"Geographic coordinates file not found: {path}")

    columns = _read_csv_columns(path)

    missing = []
    if "sample_id" not in columns:
        missing.append("sample_id")
    if longitude_col not in columns:
        missing.append(longitude_col)
    if latitude_col not in columns:
        missing.append(latitude_col)

    if missing:
        msg = (
            f"Geographic CSV is missing required columns: {missing}\n\n"
            f"  File: {path}\n"
            f"  Required: sample_id, {longitude_col}, {latitude_col}\n"
            f"  Found: {columns}"
        )
        if longitude_col not in columns or latitude_col not in columns:
            msg += (
                f"\n\n  If your columns have different names, use:\n"
                f"    --longitude-col <name> --latitude-col <name>"
            )
        raise ValidationError(msg)

    # Check numeric types
    coord_cols = [longitude_col, latitude_col]
    sample = pd.read_csv(path, nrows=5, usecols=coord_cols)
    non_numeric = [c for c in coord_cols if not pd.api.types.is_numeric_dtype(sample[c])]
    if non_numeric:
        raise ValidationError(
            f"Geographic CSV has non-numeric coordinate columns: {non_numeric}\n\n"
            f"  File: {path}\n"
            f"  Coordinate columns must contain numeric values (decimal degrees)."
        )


# ---------------------------------------------------------------------------
# Cross-file validators
# ---------------------------------------------------------------------------


def validate_labels_colormap_match(
    labels_path: Union[str, Path],
    colormap_path: Union[str, Path],
) -> None:
    """Validate that colormap columns match labels CSV columns.

    Raises ValidationError if a colormap key is not found in labels columns.
    Logs a warning if label values in the data are missing from the colormap.
    """
    labels_path = Path(labels_path)
    colormap_path = Path(colormap_path)

    label_columns = _read_csv_columns(labels_path)

    with open(colormap_path) as f:
        colormap = json.load(f)

    for cmap_col in colormap:
        if cmap_col not in label_columns:
            suggestions = _fuzzy_match(cmap_col, label_columns)
            msg = (
                f"Colormap column '{cmap_col}' not found in labels CSV.\n\n"
                f"  Labels file: {labels_path}\n"
                f"  Labels columns: {label_columns}\n"
                f"  Colormap columns: {list(colormap.keys())}"
            )
            if suggestions:
                msg += f"\n\n  Did you mean: '{suggestions[0]}'?"
            raise ValidationError(msg)

    # Warn about label values in data that are missing from colormap
    labels_df = pd.read_csv(labels_path)
    for cmap_col, color_dict in colormap.items():
        if cmap_col in labels_df.columns:
            data_values = set(labels_df[cmap_col].dropna().unique())
            cmap_values = set(color_dict.keys())
            missing = data_values - cmap_values
            if missing:
                logger.warning(
                    f"Label values in data not found in colormap for column '{cmap_col}': "
                    f"{sorted(str(v) for v in missing)}. "
                    f"These samples will appear in default gray."
                )


def validate_sample_id_overlap(
    path1: Union[str, Path],
    path2: Union[str, Path],
    name1: str,
    name2: str,
) -> None:
    """Validate that two files share sample_ids.

    Raises ValidationError if there is zero overlap.
    Logs a warning if overlap is less than 50% of the smaller file.
    """
    path1 = Path(path1)
    path2 = Path(path2)

    try:
        ids1 = set(_read_sample_ids(path1))
    except (ValueError, KeyError):
        return  # sample_id column missing; other validators will catch this
    try:
        ids2 = set(_read_sample_ids(path2))
    except (ValueError, KeyError):
        return

    overlap = ids1 & ids2

    if len(overlap) == 0:
        raise ValidationError(
            f"No overlapping sample_ids between {name1} and {name2}.\n\n"
            f"  {name1}: {path1} ({len(ids1)} samples)\n"
            f"  {name2}: {path2} ({len(ids2)} samples)\n\n"
            f"  The files share 0 sample_ids. Check that both files use the same "
            f"sample ID format."
        )

    smaller = min(len(ids1), len(ids2))
    if smaller > 0:
        pct = len(overlap) / smaller * 100
        if pct < 50:
            logger.warning(
                f"Low sample_id overlap between {name1} and {name2}: "
                f"{pct:.0f}% ({len(overlap)}/{smaller}). "
                f"This may indicate mismatched datasets."
            )
