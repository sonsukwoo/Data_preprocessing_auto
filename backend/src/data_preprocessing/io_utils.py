from __future__ import annotations

import csv
import json
import pathlib
from typing import Any, Sequence, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from .constants import (
    _EXT_PRIORITY,
    _HF_METADATA_FILENAMES,
    _IMAGE_EXTS,
    _MAX_FEATHER_MB,
    _MAX_JSON_FULL_LOAD_MB,
    _SUPPORTED_EXTS,
)

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
