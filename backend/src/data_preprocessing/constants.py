from __future__ import annotations

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_SUPPORTED_EXTS = {".csv", ".tsv", ".json", ".parquet", ".feather", ".arrow", ".xlsx", ".xls"}
_HF_METADATA_FILENAMES = {"dataset_info.json", "state.json"}

_EXT_PRIORITY = {
    ".parquet": 0,
    ".arrow": 1,
    ".feather": 2,
    ".csv": 3,
    ".tsv": 4,
    ".xlsx": 5,
    ".xls": 6,
    ".json": 7,
}

_PREVIEW_ROWS = 5
_PREVIEW_MAX_COLS = 20
_MISSING_TOP_N = 20
_CATEGORICAL_TOP_COLS = 20
_CATEGORICAL_EXAMPLE_ROWS = 3
_VALUE_REPR_LIMIT = 200

_MAX_FEATHER_MB = 512
_MAX_JSON_FULL_LOAD_MB = 256
