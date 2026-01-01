from __future__ import annotations

import csv
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
# 상수 / 설정
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
_MAX_JSON_FULL_LOAD_MB = 256

# 전수 스캔 기본값
_SCAN_DEFAULT_CHUNKSIZE = 100_000
_SCAN_TIME_LIMIT_SEC = 60
_SCAN_MAX_RETURN = 200


# =========================
# 경로 유틸
# =========================

def _strip_query(path: str) -> str:
    return path.split("?", 1)[0]


def _resolve_path(path: str) -> pathlib.Path:
    return pathlib.Path(_strip_query(path)).expanduser().resolve()


def _load_json_payload(path: pathlib.Path) -> Any:
    """JSON 파일을 UTF-8 BOM 대응으로 안전하게 로드."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > _MAX_JSON_FULL_LOAD_MB:
        raise ValueError(f"JSON file too large for full load ({size_mb:.1f} MB)")
    with path.open("r", encoding="utf-8-sig") as f:
        text = f.read().strip()
    if not text:
        raise ValueError("Empty JSON content")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        preview = text[:200].replace("\n", "\\n")
        raise ValueError(f"Invalid JSON content: {exc}. preview={preview!r}") from exc


def _detect_text_format(path: pathlib.Path) -> str:
    """텍스트 파일의 실제 포맷을 감지 (json/csv/tsv)."""

    with path.open("rb") as f:
        raw = f.read(4096)
    if not raw or not raw.strip():
        raise ValueError("Empty file content")

    try:
        sample = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        sample = raw.decode("latin-1", errors="ignore")

    stripped = sample.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return "json"

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t"])
        return "tsv" if dialect.delimiter == "\t" else "csv"
    except Exception:
        if "\t" in sample and sample.count("\t") >= sample.count(","):
            return "tsv"
        return "csv"


# =========================
# Arrow / Parquet 로더
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
# 일반 테이블 로더
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
        try:
            return pd.read_excel(p, nrows=sample_size), ext.lstrip(".")
        except Exception:
            detected = _detect_text_format(p)
            sep = "\t" if detected == "tsv" else ","
            return pd.read_csv(p, sep=sep, nrows=sample_size, on_bad_lines="skip"), detected

    if ext in {".json", ".csv", ".tsv"}:
        detected = _detect_text_format(p)
        if detected == "json":
            try:
                return pd.read_json(p, nrows=sample_size, lines=True, encoding="utf-8-sig"), "json-lines"
            except Exception:
                try:
                    return pd.read_json(p, nrows=sample_size, lines=False, encoding="utf-8-sig"), "json-array"
                except Exception:
                    raw = _load_json_payload(p)
                    if isinstance(raw, list):
                        return pd.json_normalize(raw)[:sample_size], "json-array-normalized"
                    return pd.json_normalize([raw])[:sample_size], "json-normalized"

        sep = "\t" if detected == "tsv" else ","
        try:
            return pd.read_csv(p, sep=sep, nrows=sample_size, on_bad_lines="skip"), detected
        except Exception:
            return pd.read_csv(p, sep=None, engine="python", nrows=sample_size, on_bad_lines="skip"), "csv-auto"

    # CSV 추정 폴백(관대한 파싱 유지)
    return pd.read_csv(p, sep=None, engine="python", nrows=sample_size, on_bad_lines="skip"), "csv-fallback"


# =========================
# 디렉터리 헬퍼
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
            0 if "data" in f.parts else 1,  # HF 스타일 데이터 샤드 우선
            len(f.parts),
        )
    )
    return candidates[0] if candidates else None


def _build_image_manifest(root: pathlib.Path, allowed_exts: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        # macOS zip 메타데이터: __MACOSX/ 와 AppleDouble(._*) 제외
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
# 전수 스캔 헬퍼
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

    if ext in {".json", ".csv", ".tsv"}:
        detected = _detect_text_format(path)
        if detected == "json":
            try:
                reader = pd.read_json(path, lines=True, chunksize=chunksize, encoding="utf-8-sig")
                iterator = iter(reader)
                first = next(iterator)
                if usecols:
                    first = first[list(usecols)]

                def _iter_lines() -> Iterable[pd.DataFrame]:
                    yield first
                    for chunk in iterator:
                        if usecols:
                            chunk = chunk[list(usecols)]
                        yield chunk

                return _iter_lines(), "json-lines"
            except Exception:
                try:
                    df = pd.read_json(path, lines=False, encoding="utf-8-sig")
                    if usecols:
                        df = df[list(usecols)]
                    return (df,), "json-array"
                except Exception:
                    raw = _load_json_payload(path)
                    if isinstance(raw, list):
                        def _iter_json_list() -> Iterable[pd.DataFrame]:
                            for i in range(0, len(raw), chunksize):
                                df = pd.json_normalize(raw[i:i + chunksize])
                                if usecols:
                                    df = df[list(usecols)]
                                yield df

                        return _iter_json_list(), "json-array-normalized"
                    df = pd.json_normalize(raw if isinstance(raw, list) else [raw])
                    if usecols:
                        df = df[list(usecols)]
                    return (df,), "json-normalized"

        sep = "\t" if detected == "tsv" else ","
        try:
            reader = pd.read_csv(
                path,
                sep=sep,
                usecols=usecols,
                chunksize=chunksize,
                on_bad_lines="skip",
            )
            return reader, detected
        except Exception:
            reader = pd.read_csv(
                path,
                sep=None,
                engine="python",
                usecols=usecols,
                chunksize=chunksize,
                on_bad_lines="skip",
            )
            return reader, "csv-fallback"

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

    if ext == ".xlsx":
        try:
            def _iter_xlsx() -> Iterable[pd.DataFrame]:
                import openpyxl

                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                try:
                    ws = wb.active
                    rows = ws.iter_rows(values_only=True)
                    header = next(rows, None)
                    if header is None:
                        yield pd.DataFrame()
                        return
                    columns = ["" if c is None else str(c).strip() for c in header]
                    if usecols:
                        wanted = {_normalize_column_name(c) for c in usecols}
                        indices = [i for i, c in enumerate(columns) if _normalize_column_name(c) in wanted]
                        out_cols = [columns[i] for i in indices]
                    else:
                        indices = None
                        out_cols = columns
                    chunk_rows: list[list[object]] = []
                    for row in rows:
                        if indices is not None:
                            chunk_rows.append([row[i] if i < len(row) else None for i in indices])
                        else:
                            chunk_rows.append(list(row))
                        if len(chunk_rows) >= chunksize:
                            yield pd.DataFrame(chunk_rows, columns=out_cols)
                            chunk_rows = []
                    if chunk_rows:
                        yield pd.DataFrame(chunk_rows, columns=out_cols)
                finally:
                    wb.close()

            return _iter_xlsx(), "xlsx"
        except Exception:
            try:
                df = pd.read_excel(path, usecols=usecols)
                return (df,), "xlsx"
            except Exception:
                detected = _detect_text_format(path)
                sep = "\t" if detected == "tsv" else ","
                reader = pd.read_csv(
                    path,
                    sep=sep,
                    usecols=usecols,
                    chunksize=chunksize,
                    on_bad_lines="skip",
                )
                return reader, detected

    if ext == ".xls":
        try:
            df = pd.read_excel(path, usecols=usecols)
            return (df,), "xls"
        except Exception:
            detected = _detect_text_format(path)
            sep = "\t" if detected == "tsv" else ","
            reader = pd.read_csv(
                path,
                sep=sep,
                usecols=usecols,
                chunksize=chunksize,
                on_bad_lines="skip",
            )
            return reader, detected

    # CSV 폴백
    reader = pd.read_csv(path, sep=None, engine="python", usecols=usecols, chunksize=chunksize, on_bad_lines="skip")
    return reader, "csv-fallback"

# =========================
# 요약
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
            # head() 대신 대표 예시: 최빈값 위주 + 일부 희귀값
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
# 분기:  LLM 툴콜
# =========================

# 툴: 고유값 수집
@tool
def collect_unique_values(
    path: str,
    column: str,
    max_unique: int | None = 5000,
    max_values_return: int | None = _SCAN_MAX_RETURN,
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


# 툴: 매핑 커버리지 점검
@tool
def mapping_coverage_report(
    path: str,
    column: str,
    mapping_keys: Sequence[str] | None = None,
    max_unique: int | None = 5000,
    max_values_return: int | None = _SCAN_MAX_RETURN,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    max_rows: int | None = None,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
) -> str:
    """매핑 딕셔너리 키와 데이터 고유값을 비교해 누락을 점검."""

    if not mapping_keys:
        return "ERROR_CONTEXT||InvalidArgs||mapping_keys가 비어 있습니다."

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

    mapping_set = {_safe_repr(k) for k in mapping_keys}
    missing_in_mapping = sorted(unique_values - mapping_set)
    extra_mapping_keys = sorted(mapping_set - unique_values)

    missing_preview = missing_in_mapping
    extra_preview = extra_mapping_keys
    if max_values_return and len(missing_preview) > max_values_return:
        missing_preview = missing_preview[:max_values_return]
    if max_values_return and len(extra_preview) > max_values_return:
        extra_preview = extra_preview[:max_values_return]

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "column": column,
        "normalized_column": col_norm,
        "rows_scanned": rows_scanned,
        "unique_count": len(unique_values),
        "mapping_key_count": len(mapping_set),
        "missing_in_mapping_count": len(missing_in_mapping),
        "missing_in_mapping_preview": missing_preview,
        "extra_mapping_keys_count": len(extra_mapping_keys),
        "extra_mapping_keys_preview": extra_preview,
        "truncated": bool(truncated),
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


# 툴: 희귀값 수집
@tool
def collect_rare_values(
    path: str,
    column: str,
    rare_threshold: int = 3,
    max_values_return: int | None = _SCAN_MAX_RETURN,
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


# 툴: 파싱 가능성 점검
@tool
def detect_parseability(
    path: str,
    column: str,
    parsers: Sequence[str] | None = None,
    max_samples: int | None = 2000,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    max_rows: int | None = None,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
) -> str:
    """특정 컬럼이 datetime/숫자/불리언 등으로 파싱 가능한지 점검."""

    try:
        target = _resolve_scan_target(path)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    if not column:
        return "ERROR_CONTEXT||InvalidArgs||column 인자가 필요합니다."

    chosen = [p.lower() for p in (parsers or ["datetime", "numeric", "bool"]) if isinstance(p, str)]
    if not chosen:
        return "ERROR_CONTEXT||InvalidArgs||parsers 인자가 비어 있습니다."

    start = time.time()
    samples: list[str] = []
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
            return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."

        found = True
        series = chunk[col_norm].dropna().astype(str)
        if max_samples is None:
            samples.extend(series.tolist())
        else:
            remaining = max_samples - len(samples)
            if remaining <= 0:
                break
            samples.extend(series.head(remaining).tolist())
            if len(samples) >= max_samples:
                break

    if not found:
        return f"ERROR_CONTEXT||MissingColumn||{column} 컬럼을 찾지 못했습니다."

    s = pd.Series(samples)
    results: dict[str, float] = {}

    if "datetime" in chosen:
        parsed = pd.to_datetime(s, errors="coerce")
        results["datetime"] = round(float(parsed.notna().mean()) if len(s) else 0.0, 4)

    if "numeric" in chosen:
        parsed = pd.to_numeric(s, errors="coerce")
        results["numeric"] = round(float(parsed.notna().mean()) if len(s) else 0.0, 4)

    if "bool" in chosen:
        lowered = s.str.strip().str.lower()
        bool_tokens = {"true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"}
        results["bool"] = round(float(lowered.isin(bool_tokens).mean()) if len(s) else 0.0, 4)

    payload = {
        "data_path": str(target),
        "detected_format": fmt,
        "column": column,
        "rows_scanned": rows_scanned,
        "sample_size": len(samples),
        "parse_success_rate": results,
        "sample_values": samples[: min(50, len(samples))],
        "time_limited": bool(time_limited),
        "row_limited": bool(row_limited),
    }
    return json.dumps(payload, ensure_ascii=False)


# 툴: 인코딩 추정
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


# 툴: 컬럼 프로파일링
@tool
def column_profile(
    path: str,
    columns: Sequence[str] | None = None,
    max_columns: int | None = 50,
    max_rows: int | None = None,
    time_limit_sec: int = _SCAN_TIME_LIMIT_SEC,
    chunksize: int = _SCAN_DEFAULT_CHUNKSIZE,
    sample_values_limit: int | None = 5,
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
                target_cols = list(chunk.columns) if max_columns is None else list(chunk.columns[:max_columns])

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
            if sample_values_limit is None or len(info["samples"]) < sample_values_limit:
                values = s.dropna().unique()
                for v in values:
                    info["samples"].add(_safe_repr(v))
                    if sample_values_limit is not None and len(info["samples"]) >= sample_values_limit:
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


# =========================
# 내보내기 목록
# =========================
__all__ = [
    # 전수 스캔(고유값/매핑/희귀)
    "collect_unique_values",
    "mapping_coverage_report",
    "collect_rare_values",
    # 파싱/인코딩/프로파일
    "detect_parseability",
    "detect_encoding",
    "column_profile",
]
