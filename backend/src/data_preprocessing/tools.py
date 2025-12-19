from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from typing import Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from langchain_core.tools import tool


# =========================
# Constants / Config
# =========================

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


# =========================
# Path utils
# =========================

def _strip_query(path: str) -> str:
    return path.split("?", 1)[0]


def _resolve_path(path: str) -> pathlib.Path:
    return pathlib.Path(_strip_query(path)).expanduser().resolve()


# =========================
# Arrow / Parquet loaders
# =========================

def _read_arrow_ipc_sample(path: pathlib.Path, sample_size: int) -> tuple[pd.DataFrame, str]:
    """Arrow IPC file/stream을 샘플링해서 읽기 (HF shard 대응)."""

    with path.open("rb") as f:
        try:
            reader = ipc.open_file(f)
            batches: list[pa.RecordBatch] = []
            total = 0
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                batches.append(batch)
                total += batch.num_rows
                if total >= sample_size:
                    break
            table = pa.Table.from_batches(batches).slice(0, sample_size)
            return table.to_pandas(), "arrow-ipc-file"
        except pa.ArrowInvalid:
            f.seek(0)
            reader = ipc.open_stream(f)
            batches = []
            total = 0
            while total < sample_size:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                batches.append(batch)
                total += batch.num_rows
            table = pa.Table.from_batches(batches).slice(0, sample_size)
            return table.to_pandas(), "arrow-ipc-stream"


def _read_parquet_sample(path: pathlib.Path, sample_size: int) -> pd.DataFrame:
    """Parquet을 iter_batches로 샘플링."""

    pf = pq.ParquetFile(path)
    batches: list[pa.RecordBatch] = []
    total = 0
    for batch in pf.iter_batches(batch_size=min(sample_size, 4096)):
        batches.append(batch)
        total += batch.num_rows
        if total >= sample_size:
            break
    table = pa.Table.from_batches(batches).slice(0, sample_size)
    return table.to_pandas()


# =========================
# Generic table loader
# =========================

def _read_table_like(path: str, sample_size: int) -> tuple[pd.DataFrame, str]:
    """지원 포맷을 샘플링으로 읽고 (df, detected_format)을 반환."""

    p = _resolve_path(path)
    ext = p.suffix.lower()

    if ext == ".parquet":
        try:
            return _read_parquet_sample(p, sample_size), "parquet"
        except Exception:
            table = pq.read_table(p, use_threads=True)
            return table.slice(0, sample_size).to_pandas(), "parquet"

    if ext == ".arrow":
        return _read_arrow_ipc_sample(p, sample_size)

    if ext == ".feather":
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > _MAX_FEATHER_MB:
            raise ValueError(f"Feather file too large ({size_mb:.1f} MB)")
        return pd.read_feather(p).head(sample_size), "feather"

    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(p, nrows=sample_size), ext.lstrip(".")

    if ext == ".json":
        try:
            return pd.read_json(p, nrows=sample_size, lines=True), "json-lines"
        except Exception:
            try:
                return pd.read_json(p, nrows=sample_size, lines=False), "json-array"
            except Exception:
                with p.open("r", encoding="utf-8", errors="replace") as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    return pd.json_normalize(raw)[:sample_size], "json-array-normalized"
                return pd.json_normalize([raw])[:sample_size], "json-normalized"

    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        try:
            return pd.read_csv(p, sep=sep, nrows=sample_size, on_bad_lines="skip"), ext.lstrip(".")
        except Exception:
            return pd.read_csv(p, sep=None, engine="python", nrows=sample_size, on_bad_lines="skip"), "csv-auto"

    # Fallback CSV heuristics (keep behavior permissive)
    return pd.read_csv(p, sep=None, engine="python", nrows=sample_size, on_bad_lines="skip"), "csv-fallback"


# =========================
# Directory helpers
# =========================

def _pick_candidate_from_dir(root: pathlib.Path) -> pathlib.Path | None:
    candidates: list[pathlib.Path] = []
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        if f.name in _HF_METADATA_FILENAMES:
            continue
        ext = f.suffix.lower()
        if ext not in _SUPPORTED_EXTS:
            continue
        candidates.append(f)

    candidates.sort(
        key=lambda f: (
            _EXT_PRIORITY.get(f.suffix.lower(), 999),
            0 if "data" in f.parts else 1,  # HF-style data shards when present
            len(f.parts),
        )
    )
    return candidates[0] if candidates else None


def _build_image_manifest(root: pathlib.Path, allowed_exts: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # macOS zip metadata: ignore __MACOSX/ and AppleDouble files (._*)
        if "__MACOSX" in path.parts:
            continue
        if path.name.startswith("._"):
            continue
        if path.name == ".DS_Store":
            continue
        if path.suffix.lower() not in allowed_exts:
            continue
        rows.append(
            {
                "filepath": str(path.resolve()),
                "filename": path.name,
                "label": path.parent.name,
            }
        )
    if not rows:
        raise ValueError(f"No image files found under {root} with extensions {sorted(allowed_exts)}")
    return pd.DataFrame(rows)


def _normalize_image_exts(extensions: Sequence[str] | None) -> set[str]:
    allowed: Iterable[str] = extensions or tuple(e.lstrip(".") for e in sorted(_IMAGE_EXTS))
    return {f".{ext.lower().lstrip('.')}" for ext in allowed}


# =========================
# Summarization
# =========================

def _safe_repr(value: object, limit: int = _VALUE_REPR_LIMIT) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return f"<bytes len={len(value)}>"
        except Exception:
            return "<bytes>"
    if isinstance(value, dict):
        keys = list(value.keys())
        preview = ", ".join(map(str, keys[:5]))
        more = "" if len(keys) <= 5 else f", …(+{len(keys) - 5})"
        return f"<dict keys=[{preview}{more}]>"
    if isinstance(value, (list, tuple, set)):
        try:
            return f"<{type(value).__name__} len={len(value)}>"
        except Exception:
            return f"<{type(value).__name__}>"
    s = str(value)
    return s if len(s) <= limit else s[:limit] + "…"


def _summarize_dataframe(df: pd.DataFrame) -> tuple[str, str, str, str, str]:
    preview = df.head(_PREVIEW_ROWS).copy()
    truncated_note = ""
    if preview.shape[1] > _PREVIEW_MAX_COLS:
        truncated_note = f"(미리보기 컬럼 {_PREVIEW_MAX_COLS}개만 표시; 전체 {preview.shape[1]}개)\n"
        preview = preview.iloc[:, :_PREVIEW_MAX_COLS]
    preview = preview.map(_safe_repr)
    head_md = truncated_note + preview.to_markdown()

    dtypes_md = df.dtypes.astype(str).to_frame("dtype").to_markdown()

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_md = (
        missing.head(_MISSING_TOP_N).to_frame("missing").to_markdown() if not missing.empty else "(결측치 없음)"
    )

    numeric = df.select_dtypes(include="number")
    numeric_md = numeric.describe().transpose().to_markdown() if not numeric.empty else "(수치형 컬럼 없음)"

    non_numeric_cols = [c for c in df.columns if c not in numeric.columns]
    cat_rows: list[dict[str, object]] = []
    for col in non_numeric_cols[:_CATEGORICAL_TOP_COLS]:
        s = df[col]
        try:
            nunique = int(s.nunique(dropna=True))
        except Exception:
            nunique = -1
        try:
            # Prefer representative examples over head(): show top-frequency values, plus a couple rare ones.
            vc = s.dropna().map(_safe_repr).value_counts()
            top = vc.head(_CATEGORICAL_EXAMPLE_ROWS)
            tail_n = max(0, _CATEGORICAL_EXAMPLE_ROWS - len(top))
            rare = vc.tail(tail_n) if tail_n else vc.tail(0)
            top_str = ", ".join([f"{idx}({int(cnt)})" for idx, cnt in top.items()])
            rare_str = ", ".join([f"{idx}({int(cnt)})" for idx, cnt in rare.items()])
            if top_str and rare_str:
                examples = [f"top: {top_str}; rare: {rare_str}"]
            elif top_str:
                examples = [f"top: {top_str}"]
            elif rare_str:
                examples = [f"rare: {rare_str}"]
            else:
                examples = []
        except Exception:
            examples = []
        cat_rows.append({"column": col, "nunique": nunique, "examples": ", ".join(examples)})
    categorical_md = pd.DataFrame(cat_rows).to_markdown(index=False) if cat_rows else "(비수치형 컬럼 없음)"

    return head_md, dtypes_md, missing_md, numeric_md, categorical_md


# =========================
# Tools
# =========================

@tool
def load_and_sample(path: str, sample_size: int = 5000) -> str:
    """파일 또는 디렉터리를 일부 로드/인덱싱해 마크다운으로 요약 반환.

    - 지원 포맷: csv/tsv/json/jsonl/parquet/arrow/feather/excel
    - 디렉터리 입력 시: 이미지가 있으면 매니페스트(csv) 생성, 아니면 첫 지원 파일을 샘플링.
    - 샘플 크기까지만 읽어 메모리를 보호한다.
    - 실패 시 ERROR_CONTEXT 마커와 함께 원문 에러를 반환한다.
    """

    p = _resolve_path(path)

    # 디렉터리 처리: 이미지가 있으면 매니페스트 우선
    if p.is_dir():
        has_image = any(f.suffix.lower() in _IMAGE_EXTS for f in p.rglob("*"))
        if has_image:
            manifest_info = list_images_to_csv.invoke({"dir_path": str(p)})
            return manifest_info

        candidate = _pick_candidate_from_dir(p)
        if candidate is None:
            return f"ERROR_CONTEXT||NoSupportedFiles||{p} 아래에서 지원하는 데이터 파일을 찾지 못했습니다."
        p = candidate

    try:
        df, fmt = _read_table_like(str(p), sample_size=sample_size)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    head_md, dtypes_md, missing_md, numeric_md, categorical_md = _summarize_dataframe(df)

    return (
        f"data_path: {p}\n"
        f"detected_format: {fmt}\n"
        f"sample_rows: {len(df)}\n"
        "head:\n"
        f"{head_md}\n\n"
        "dtypes:\n"
        f"{dtypes_md}\n\n"
        f"missing (top {_MISSING_TOP_N}):\n"
        f"{missing_md}\n\n"
        "numeric describe:\n"
        f"{numeric_md}\n\n"
        f"categorical/examples (top {_CATEGORICAL_TOP_COLS} cols):\n"
        f"{categorical_md}\n"
    )


@tool
def list_images_to_csv(
    dir_path: str,
    output_csv: str | None = None,
    extensions: Sequence[str] | None = None,
    sample_size: int = 20,
) -> str:
    """디렉터리의 이미지 파일을 찾아 CSV 매니페스트로 저장하고 요약을 반환."""

    root = _resolve_path(dir_path)
    if not root.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    allowed_exts = _normalize_image_exts(extensions)
    df = _build_image_manifest(root, allowed_exts)

    if output_csv is None:
        output_csv_path = root / "image_index.csv"
    else:
        output_csv_path = pathlib.Path(output_csv).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    preview_md = df.head(sample_size).to_markdown()
    return (
        f"dir_path: {root}\n"
        f"found_files: {len(df)}\n"
        f"output_csv: {output_csv_path}\n"
        "columns: filepath, filename, label\n"
        "preview:\n"
        f"{preview_md}\n"
    )


__all__ = ["load_and_sample", "list_images_to_csv"]
