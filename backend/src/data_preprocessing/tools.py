from __future__ import annotations

import json
import pathlib
import time
from collections import Counter
from collections.abc import Iterable
from typing import Any, Sequence

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

# Full scan defaults
_SCAN_DEFAULT_CHUNKSIZE = 100_000
_SCAN_TIME_LIMIT_SEC = 20
_SCAN_MAX_RETURN = 200


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
# Full scan helpers
# =========================

def _resolve_scan_target(path: str) -> pathlib.Path:
    p = _resolve_path(path)
    if p.is_dir():
        candidate = _pick_candidate_from_dir(p)
        if candidate is None:
            raise ValueError(f"No supported data file found under {p}")
        return candidate
    return p


def _normalize_column_name(col: object) -> str:
    return str(col).strip().strip("\"'")


def _sanitize_columns(df: pd.DataFrame) -> None:
    df.columns = [_normalize_column_name(c) for c in df.columns]


def _get_table_iterator(
    path: pathlib.Path,
    usecols: Sequence[str] | None,
    chunksize: int,
) -> tuple[Iterable[pd.DataFrame], str]:
    ext = path.suffix.lower()

    if ext in {".csv", ".tsv"}:
        sep = "\t" if ext == ".tsv" else ","
        reader = pd.read_csv(path, sep=sep, usecols=usecols, chunksize=chunksize, on_bad_lines="skip")
        return reader, ext.lstrip(".")

    if ext == ".parquet":
        pf = pq.ParquetFile(path)
        columns = list(usecols) if usecols else None

        def _iter() -> Iterable[pd.DataFrame]:
            for batch in pf.iter_batches(batch_size=chunksize, columns=columns):
                yield batch.to_pandas()

        return _iter(), "parquet"

    if ext == ".arrow":
        def _iter_arrow() -> Iterable[pd.DataFrame]:
            with path.open("rb") as f:
                try:
                    reader = ipc.open_file(f)
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        df = batch.to_pandas()
                        if usecols:
                            df = df[list(usecols)]
                        yield df
                except pa.ArrowInvalid:
                    f.seek(0)
                    reader = ipc.open_stream(f)
                    while True:
                        try:
                            batch = reader.read_next_batch()
                        except StopIteration:
                            break
                        df = batch.to_pandas()
                        if usecols:
                            df = df[list(usecols)]
                        yield df

        return _iter_arrow(), "arrow"

    if ext == ".feather":
        df = pd.read_feather(path, columns=list(usecols) if usecols else None)
        return (df,), "feather"

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path, usecols=usecols)
        return (df,), ext.lstrip(".")

    if ext == ".json":
        try:
            reader = pd.read_json(path, lines=True, chunksize=chunksize)
            return reader, "json-lines"
        except Exception:
            df = pd.read_json(path, lines=False)
            if usecols:
                df = df[list(usecols)]
            return (df,), "json"

    # Fallback CSV
    reader = pd.read_csv(path, sep=None, engine="python", usecols=usecols, chunksize=chunksize, on_bad_lines="skip")
    return reader, "csv-fallback"

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
def inspect_input(path: str, max_files: int = 50) -> str:
    """입력 경로를 검사해 이미지 폴더/테이블 후보를 판별."""

    p = _resolve_path(path)
    if not p.exists():
        return f"ERROR_CONTEXT||FileNotFoundError||{path} 경로가 존재하지 않습니다."

    if p.is_dir():
        has_image = any(f.suffix.lower() in _IMAGE_EXTS for f in p.rglob("*") if f.is_file())
        candidates: list[str] = []
        for f in p.rglob("*"):
            if not f.is_file():
                continue
            if f.name in _HF_METADATA_FILENAMES:
                continue
            ext = f.suffix.lower()
            if ext not in _SUPPORTED_EXTS:
                continue
            candidates.append(str(f.resolve()))
            if len(candidates) >= max_files:
                break

        candidate = _pick_candidate_from_dir(p)
        if not has_image and candidate is None:
            return f"ERROR_CONTEXT||NoSupportedFiles||{p} 아래에서 지원하는 데이터 파일을 찾지 못했습니다."

        payload = {
            "input_path": str(p),
            "is_dir": True,
            "has_images": bool(has_image),
            "candidate_file": str(candidate) if candidate else "",
            "supported_files": candidates,
            "supported_count": len(candidates),
        }
        return json.dumps(payload, ensure_ascii=False)

    payload = {
        "input_path": str(p),
        "is_dir": False,
        "has_images": False,
        "candidate_file": str(p),
        "supported_files": [str(p)],
        "supported_count": 1,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def sample_table(path: str, sample_size: int = 5000) -> str:
    """표 형식 파일(또는 디렉터리)을 샘플링해 JSON으로 반환."""

    p = _resolve_path(path)

    if p.is_dir():
        has_image = any(f.suffix.lower() in _IMAGE_EXTS for f in p.rglob("*") if f.is_file())
        if has_image:
            return f"ERROR_CONTEXT||ImageFolder||{p} 아래에 이미지가 있어 테이블 샘플링을 생략합니다."
        candidate = _pick_candidate_from_dir(p)
        if candidate is None:
            return f"ERROR_CONTEXT||NoSupportedFiles||{p} 아래에서 지원하는 데이터 파일을 찾지 못했습니다."
        p = candidate

    try:
        df, fmt = _read_table_like(str(p), sample_size=sample_size)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    try:
        sample_records = json.loads(df.to_json(orient="records", date_format="iso"))
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||SampleSerializeError||{exc}"

    payload = {
        "data_path": str(p),
        "detected_format": fmt,
        "sample_rows": len(df),
        "sample": sample_records,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def summarize_table(sample_json: str) -> str:
    """sample_table JSON을 요약 Markdown으로 변환."""

    try:
        payload = json.loads(sample_json)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||InvalidJSON||{exc}"

    if not isinstance(payload, dict):
        return "ERROR_CONTEXT||InvalidPayload||sample_json은 dict 형태여야 합니다."

    sample = payload.get("sample")
    if not isinstance(sample, list):
        return "ERROR_CONTEXT||InvalidPayload||sample_json['sample']이 list가 아닙니다."

    df = pd.DataFrame(sample)
    head_md, dtypes_md, missing_md, numeric_md, categorical_md = _summarize_dataframe(df)
    data_path = payload.get("data_path", "")
    fmt = payload.get("detected_format", "")
    sample_rows = payload.get("sample_rows", len(df))

    return (
        f"data_path: {data_path}\n"
        f"detected_format: {fmt}\n"
        f"sample_rows: {sample_rows}\n"
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


# =========================
# Full scan tools
# =========================

@tool
def collect_unique_values(
    path: str,
    column: str,
    max_unique: int = 5000,
    max_values_return: int = _SCAN_MAX_RETURN,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    max_rows: int | None = None,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
) -> str:
    """특정 컬럼의 고유값을 전수 스캔으로 수집."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    start = time.time()
    unique_values: set[str] = set()
    rows_scanned = 0
    truncated = False
    time_limited = False
    row_limited = False

    try:
        iterable, fmt = _get_table_iterator(target, usecols=None, chunksize=chunksize)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    col_norm = _normalize_column_name(column)
    found = False

    for chunk in iterable:
        if time_limit_sec and (time.time() - start) > time_limit_sec:
            time_limited = True
            break
        rows_scanned += len(chunk)
        if max_rows and rows_scanned > max_rows:
            row_limited = True
            break
        _sanitize_columns(chunk)
        if col_norm not in chunk.columns:
            if not found:
                return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."
            continue
        found = True
        series = chunk[col_norm].dropna()
        if not series.empty:
            values = series.unique()
            unique_values.update(_safe_repr(v) for v in values)
        if max_unique and len(unique_values) >= max_unique:
            truncated = True
            break

    if not found:
        return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."

    values_list = sorted(unique_values)
    if max_values_return and len(values_list) > max_values_return:
        values_list = values_list[:max_values_return]

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "column": column,
        "normalized_column": col_norm,
        "rows_scanned": rows_scanned,
        "unique_count": len(unique_values),
        "unique_values_preview": values_list,
        "truncated": bool(truncated),
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def collect_rare_values(
    path: str,
    column: str,
    rare_threshold: int = 3,
    max_values_return: int = _SCAN_MAX_RETURN,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    max_rows: int | None = None,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
) -> str:
    """특정 컬럼의 희귀값(빈도 낮은 값)을 전수 스캔으로 수집."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    start = time.time()
    counts: Counter[str] = Counter()
    rows_scanned = 0
    time_limited = False
    row_limited = False

    try:
        iterable, fmt = _get_table_iterator(target, usecols=None, chunksize=chunksize)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    col_norm = _normalize_column_name(column)
    found = False

    for chunk in iterable:
        if time_limit_sec and (time.time() - start) > time_limit_sec:
            time_limited = True
            break
        rows_scanned += len(chunk)
        if max_rows and rows_scanned > max_rows:
            row_limited = True
            break
        _sanitize_columns(chunk)
        if col_norm not in chunk.columns:
            if not found:
                return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."
            continue
        found = True
        series = chunk[col_norm].dropna()
        if not series.empty:
            values = series.map(_safe_repr)
            counts.update(values.value_counts().to_dict())

    if not found:
        return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."

    rare_items = [(val, cnt) for val, cnt in counts.items() if cnt <= rare_threshold]
    rare_items.sort(key=lambda x: x[1])
    rare_preview = rare_items[:max_values_return] if max_values_return else rare_items

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "column": column,
        "normalized_column": col_norm,
        "rows_scanned": rows_scanned,
        "rare_threshold": rare_threshold,
        "rare_count": len(rare_items),
        "rare_values_preview": rare_preview,
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def detect_datetime_formats(
    path: str,
    column: str | None = None,
    max_samples: int = 2000,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    max_rows: int | None = None,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
) -> str:
    """날짜/시간 컬럼의 파싱 가능 여부를 점검."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    start = time.time()
    samples: list[str] = []
    rows_scanned = 0
    time_limited = False
    row_limited = False

    try:
        iterable, fmt = _get_table_iterator(target, usecols=None, chunksize=chunksize)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    col_norm = _normalize_column_name(column) if column else None
    found = False

    for chunk in iterable:
        if time_limit_sec and (time.time() - start) > time_limit_sec:
            time_limited = True
            break
        rows_scanned += len(chunk)
        if max_rows and rows_scanned > max_rows:
            row_limited = True
            break
        _sanitize_columns(chunk)

        target_col = col_norm
        if target_col is None:
            for c in chunk.columns:
                if "date" in c.lower() or "time" in c.lower():
                    target_col = c
                    break
        if target_col is None or target_col not in chunk.columns:
            continue

        found = True
        series = chunk[target_col].dropna().astype(str)
        remaining = max_samples - len(samples)
        if remaining <= 0:
            break
        samples.extend(series.head(remaining).tolist())
        if len(samples) >= max_samples:
            break

    if not found:
        return "ERROR_CONTEXT||MissingColumn||날짜/시간 컬럼을 찾지 못했습니다."

    parsed = pd.to_datetime(pd.Series(samples), errors="coerce")
    success_rate = float(parsed.notna().mean()) if samples else 0.0

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "column": column or "(auto-detected)",
        "rows_scanned": rows_scanned,
        "sample_size": len(samples),
        "parse_success_rate": round(success_rate, 4),
        "sample_values": samples[: min(50, len(samples))],
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def detect_encoding(path: str, sample_bytes: int = 100_000) -> str:
    """텍스트 파일의 인코딩을 간단히 추정."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    ext = target.suffix.lower()
    if ext not in {".csv", ".tsv", ".json"}:
        return f"ERROR_CONTEXT||NotApplicable||{ext} 파일에는 인코딩 추정이 필요 없습니다."

    raw = target.read_bytes()[:sample_bytes]
    candidates = ["utf-8-sig", "utf-8", "cp949", "euc-kr", "latin-1"]
    guessed = None
    for enc in candidates:
        try:
            raw.decode(enc)
            guessed = enc
            break
        except Exception:
            continue

    payload = {
        "data_path": str(target),
        "detected_format": ext.lstrip("."),
        "sample_bytes": len(raw),
        "encoding_guess": guessed or "unknown",
        "candidates": candidates,
    }
    return json.dumps(payload, ensure_ascii=False)


@tool
def column_profile(
    path: str,
    columns: Sequence[str] | None = None,
    max_columns: int = 50,
    max_rows: int | None = None,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
    sample_values_limit: int = 5,
) -> str:
    """컬럼별 타입/결측률/샘플값을 전수 스캔으로 요약."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    start = time.time()
    rows_scanned = 0
    time_limited = False
    row_limited = False
    profiles: dict[str, dict[str, Any]] = {}

    try:
        iterable, fmt = _get_table_iterator(target, usecols=None, chunksize=chunksize)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    target_cols: list[str] | None = None
    normalized_columns = [_normalize_column_name(c) for c in columns] if columns else None

    for chunk in iterable:
        if time_limit_sec and (time.time() - start) > time_limit_sec:
            time_limited = True
            break
        rows_scanned += len(chunk)
        if max_rows and rows_scanned > max_rows:
            row_limited = True
            break
        _sanitize_columns(chunk)

        if target_cols is None:
            if normalized_columns:
                missing = [c for c in normalized_columns if c not in chunk.columns]
                if missing:
                    return f"ERROR_CONTEXT||MissingColumn||{missing} 컬럼을 찾지 못했습니다."
                target_cols = list(normalized_columns)
            else:
                target_cols = list(chunk.columns[:max_columns])

        for col in target_cols:
            if col not in chunk.columns:
                continue
            s = chunk[col]
            info = profiles.setdefault(
                col,
                {
                    "dtype": str(s.dtype),
                    "missing": 0,
                    "total": 0,
                    "samples": set(),
                },
            )
            info["total"] += len(s)
            info["missing"] += int(s.isna().sum())
            if len(info["samples"]) < sample_values_limit:
                values = s.dropna().unique()
                for v in values:
                    info["samples"].add(_safe_repr(v))
                    if len(info["samples"]) >= sample_values_limit:
                        break

    if not profiles:
        return "ERROR_CONTEXT||EmptyData||스캔할 데이터가 없습니다."

    summary = []
    for col, info in profiles.items():
        total = info["total"] or 0
        missing = info["missing"] or 0
        missing_rate = round(missing / total, 6) if total else 0.0
        summary.append(
            {
                "column": col,
                "dtype": info["dtype"],
                "total": total,
                "missing": missing,
                "missing_rate": missing_rate,
                "samples": sorted(info["samples"]),
            }
        )

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "rows_scanned": rows_scanned,
        "columns_profiled": len(summary),
        "profiles": summary,
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


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


__all__ = [
    "collect_unique_values",
    "collect_rare_values",
    "detect_datetime_formats",
    "detect_encoding",
    "column_profile",
    "inspect_input",
    "sample_table",
    "summarize_table",
    "list_images_to_csv",
]
