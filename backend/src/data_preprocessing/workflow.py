from __future__ import annotations

# =========================
# 표준 라이브러리
# =========================
import csv
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
from uuid import uuid4

# =========================
# 외부 라이브러리
# =========================
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

# =========================
# 내부 모듈: 모델/프롬포트
# =========================
from .models import CodeBlocks, RequirementsPayload, State
from .prompts import (
    REQUIREMENTS_SYSTEM_PROMPT,
    REQUIREMENTS_USER_TEMPLATE,
    code_gen_prompt,
    reflect_prompt,
)

# =========================
# 내부 모듈: 툴
# =========================
from .tools import (
    collect_rare_values,
    collect_unique_values,
    column_profile,
    detect_parseability,
    detect_encoding,
    mapping_coverage_report,
)

# =========================
# 내부 모듈: 공용 유틸
# =========================
from .common_utils import (
    append_trace,
    cleanup_dir,
    detect_sample_fallback,
    extract_input_path,
    extract_last_message_text,
    extract_user_request,
    now_iso,
    prepare_execution_workdir,
    promote_staged_outputs,
    run_generated_code,
    safe_format_json_like,
    safe_last_message,
    truncate_text,
    diff_generation,
    write_internal_trace_markdown,
)


# =========================
# 내부 유틸: 입력 검사/샘플링 (노드에 병합)
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


def _strip_query(path: str) -> str:
    return path.split("?", 1)[0]


def _resolve_path(path: str) -> Path:
    return Path(_strip_query(path)).expanduser().resolve()


def _load_json_payload(path: Path) -> Any:
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


def _detect_text_format(path: Path) -> str:
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


def _read_arrow_ipc_sample(path: Path, sample_size: int) -> tuple[pd.DataFrame, str]:
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


def _read_parquet_sample(path: Path, sample_size: int) -> pd.DataFrame:
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


def _read_table_like(path: str, sample_size: int) -> tuple[pd.DataFrame, str]:
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

    return pd.read_csv(p, sep=None, engine="python", nrows=sample_size, on_bad_lines="skip"), "csv-fallback"


def _pick_candidate_from_dir(root: Path) -> Path | None:
    candidates: list[Path] = []
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
            0 if "data" in f.parts else 1,
            len(f.parts),
        )
    )
    return candidates[0] if candidates else None


def _build_image_manifest(root: Path, allowed_exts: set[str]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
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


def inspect_input(path: str, max_files: int = 50) -> str:
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


def sample_table(path: str, sample_size: int = 5000) -> str:
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


def summarize_table(sample_json: str) -> str:
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


def list_images_to_csv(
    dir_path: str,
    output_csv: str | None = None,
    extensions: Sequence[str] | None = None,
    sample_size: int = 20,
) -> str:
    root = _resolve_path(dir_path)
    if not root.is_dir():
        raise ValueError(f"Directory not found: {dir_path}")

    allowed_exts = _normalize_image_exts(extensions)
    df = _build_image_manifest(root, allowed_exts)

    if output_csv is None:
        output_csv_path = root / "image_index.csv"
    else:
        output_csv_path = Path(output_csv).expanduser().resolve()
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


# =========================
# 내부 유틸: run_planned_tools 노드 전용
# =========================
# - run_planned_tools 노드에서만 호출됨
def _toolcall_parse_entry(tool_call: object) -> tuple[str, dict[str, object], str]:
    if hasattr(tool_call, "name"):
        name = str(getattr(tool_call, "name", "") or "")
        raw_args = getattr(tool_call, "args", None)
        if raw_args is None:
            args: dict[str, object] = {}
        elif hasattr(raw_args, "model_dump"):
            args = raw_args.model_dump(exclude_none=True)
        else:
            args = dict(raw_args) if isinstance(raw_args, dict) else {}
        reason = str(getattr(tool_call, "reason", "") or "")
        return name, args, reason
    if isinstance(tool_call, dict):
        name = str(tool_call.get("name") or "")
        args = dict(tool_call.get("args") or {})
        reason = str(tool_call.get("reason") or "")
        return name, args, reason
    return "", {}, ""


def _toolcall_apply_defaults(
    name: str,
    args: dict[str, object],
    default_path: str,
    timeout_only_tools: set[str],
    limit_keys_by_tool: dict[str, list[str]],
) -> dict[str, object]:
    if "path" not in args and default_path:
        args["path"] = default_path
    elif default_path and isinstance(args.get("path"), str):
        raw_path = str(args.get("path", "")).strip()
        if raw_path in {"data_path", "<data_path>", "path", "<path>"}:
            args["path"] = default_path
    if name in timeout_only_tools:
        args["time_limit_sec"] = 60
        for key in limit_keys_by_tool.get(name, []):
            args[key] = None
    return args


def _toolcall_build_dedup_key(name: str, args: dict[str, object]) -> str:
    return json.dumps({"name": name, "args": args}, ensure_ascii=False, sort_keys=True, default=str)


def _toolcall_format_reports(tool_reports: list[dict[str, object]], base_context: str) -> str:
    if not tool_reports:
        return base_context
    lines = ["", "tool_reports:"]
    for rep in tool_reports:
        lines.append(f"- name: {rep['name']}")
        if rep.get("reason"):
            lines.append(f"  reason: {rep['reason']}")
        args_text = json.dumps(rep.get("args") or {}, ensure_ascii=False, default=str)
        lines.append(f"  args: {args_text}")
        lines.append("  output:")
        lines.append(truncate_text(str(rep.get("output") or ""), limit=4000))
    return (base_context + "\n" if base_context else "") + "\n".join(lines)

# =========================
# 내부 유틸: code_check 노드 전용
# =========================
# - code_check 노드에서만 호출됨

def _exec_run_generated_code(
    imports: str,
    code: str,
) -> tuple[dict[str, object], str, str | None, str]:
    execution_workdir = prepare_execution_workdir()
    exec_globals, output_preview, error_kind, error_detail = run_generated_code(
        imports, code, execution_workdir
    )
    exec_globals["_execution_workdir"] = execution_workdir
    return exec_globals, output_preview, error_kind, error_detail


def _code_check_error_response(
    state: State,
    code_solution: CodeBlocks,
    error_detail: str,
    execution_stdout: str,
    *,
    error_kind: str | None = None,
) -> dict[str, object]:
    if error_kind == "system_exit":
        error_message = [
            ("user", f"Your solution called exit() during execution (this is not allowed): {error_detail}")
        ]
    else:
        error_message = [("user", f"Your solution failed during execution: {error_detail}")]
    return {
        "generation": code_solution,
        "messages": error_message,
        "error": "yes",
        "phase": "executing",
        "execution_stdout": execution_stdout,
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "code_check",
                "phase": "executing",
                "iterations": int(state.get("iterations", 0) or 0),
                "error": "yes",
                "error_detail": error_detail,
                "execution_stdout": execution_stdout,
                "generation": {"imports": code_solution.imports, "code": code_solution.code},
            },
        ),
    }

# =========================
# 내부 유틸: 사용자 친화 오류 노드 전용
# =========================
# - friendly_error 노드에서만 호출됨

def _build_friendly_error_prompt(
    user_request: str,
    *,
    exc_name: str | None = None,
    exc_msg: str | None = None,
    raw_error: str | None = None,
) -> str:
    if exc_name is not None and exc_msg is not None:
        return (
            "다음 에러를 비개발자가 이해할 수 있게 한글로 짧게 설명하고, 해결 방법을 한 줄로 제안하세요.\n"
            f"요청: {user_request}\n"
            f"에러명: {exc_name}\n"
            f"에러내용: {exc_msg}\n"
            "출력 형식: 원인: ...\n대처: ..."
        )
    return (
        "다음 전처리 실행이 최종적으로 실패했습니다. 비개발자가 이해할 수 있게 한글로 짧게 설명하고, "
        "사용자가 바로 시도할 수 있는 해결 방법을 1~3개 제안하세요.\n"
        f"요청: {user_request}\n"
        f"에러내용(일부):\n{raw_error or ''}\n"
        "출력 형식:\n원인: ...\n대처: ...\n"
    )


def _friendly_error_response(
    state: State,
    llm_gpt: ChatOpenAI,
    *,
    user_request: str,
    node_name: str,
    prompt: str,
    trace_payload: dict[str, object],
) -> dict[str, object]:
    resp = llm_gpt.invoke(prompt)
    return {
        "final_user_messages": [("assistant", resp.content)],
        "error": "yes",
        "phase": "finalizing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": node_name,
                "phase": "finalizing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "assistant_message": resp.content,
                **trace_payload,
            },
        ),
    }

# =========================
# 내부 유틸: 검증 노드 전용
# =========================
# - validate 노드에서만 호출됨


def _validation_error_response(
    state: State,
    report: object,
    execution_workdir: Path | None,
    msg: str,
    *,
    last_message: str | None = None,
    stdout_tail: str | None = None,
    extra_trace: dict[str, object] | None = None,
) -> dict[str, object]:
    if execution_workdir:
        cleanup_dir(execution_workdir)
    trace_payload: dict[str, object] = {
        "ts": now_iso(),
        "node": "validate",
        "phase": "validating",
        "iterations": int(state.get("iterations", 0) or 0),
        "error": "yes",
        "validation_report": report,
        "last_message": last_message or msg,
    }
    if stdout_tail is not None:
        trace_payload["execution_stdout_tail"] = stdout_tail
    if extra_trace:
        trace_payload.update(extra_trace)
    return {
        "messages": [("user", msg)],
        "error": "yes",
        "phase": "validating",
        **append_trace(state, trace_payload),
    }


def _validation_fail_missing_report(
    state: State,
    report: object,
    execution_workdir: Path | None,
) -> dict[str, object]:
    stdout_tail = (state.get("execution_stdout") or "")[-2000:]
    msg = (
        "Validation failed: missing __validation_report__.\n"
        "Your script MUST set a JSON-serializable dict named __validation_report__ with at least {ok: bool, issues: [...] }.\n"
        f"Execution stdout (tail):\n{stdout_tail}"
    )
    return _validation_error_response(
        state,
        report,
        execution_workdir,
        msg,
        last_message=safe_last_message(state.get("messages", [])),
        stdout_tail=stdout_tail,
    )


def _validation_fail_missing_ok_issues(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
) -> dict[str, object]:
    msg = (
        "Validation failed: __validation_report__ must include keys 'ok' and 'issues'.\n"
        f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}"
    )
    return _validation_error_response(
        state,
        report,
        execution_workdir,
        msg,
        last_message=safe_last_message(state.get("messages", [])),
    )


def _validation_fail_missing_metrics(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
) -> dict[str, object]:
    msg = (
        "Validation failed: __validation_report__ must include a dict 'metrics' to justify ok=True.\n"
        f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}"
    )
    return _validation_error_response(
        state,
        report,
        execution_workdir,
        msg,
        last_message=safe_last_message(state.get("messages", [])),
    )


def _validation_fail_missing_requirements(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
    reqs: list,
) -> dict[str, object]:
    req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
    msg = (
        "Validation failed: missing __validation_report__['requirements'].\n"
        "Your script MUST report requirement-level pass/fail in __validation_report__['requirements'].\n"
        f"Required:\n{req_list}\n"
    )
    return _validation_error_response(state, report, execution_workdir, msg)


def _validation_fail_requirements(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
    reqs: list,
    missing_ids: list[str],
    failed_ids: list[str],
) -> dict[str, object]:
    req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
    msg = (
        "Validation failed: user requirements not satisfied.\n"
        f"Missing requirement ids: {missing_ids}\n"
        f"Failed requirement ids: {failed_ids}\n"
        "All requirements (must satisfy):\n"
        f"{req_list}\n"
        f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=4000)}"
    )
    return _validation_error_response(state, report, execution_workdir, msg)


def _validation_fail_missing_placeholder_metrics(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
    missing_placeholder_metrics: list[str],
) -> dict[str, object]:
    msg = (
        "Validation failed: ok=True but placeholder/fallback coverage metrics are missing.\n"
        "For each filled/added column, include '<col>_placeholder' (or '<col>_fallback') and set ok=False when it > 0.\n"
        f"Columns requiring placeholder metrics: {missing_placeholder_metrics}\n"
        f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}"
    )
    return _validation_error_response(state, report, execution_workdir, msg)


def _validation_fail_missing_mapping(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
    missing_mapping_count: int | None,
    missing_mapping: list[str],
) -> dict[str, object]:
    msg = (
        "Validation failed: mapping coverage is incomplete.\n"
        f"missing_mapping_count: {missing_mapping_count}\n"
        f"missing_mapping: {missing_mapping}\n"
        f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=4000)}"
    )
    return _validation_error_response(state, report, execution_workdir, msg)


def _validation_fail_silent_miss(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
    issues_list: list[str],
    metric_flags: list[str],
) -> dict[str, object]:
    if metric_flags:
        issues_list = issues_list + [f"metrics indicate potential silent miss: {x}" for x in metric_flags]
    report_json = safe_format_json_like(report, limit=6000)
    issue_text = "\n".join(f"- {x}" for x in issues_list) if issues_list else "(no issues list provided)"
    msg = (
        "Validation failed (silent-miss guardrail).\n"
        f"Issues:\n{issue_text}\n\n"
        f"__validation_report__ (truncated):\n{report_json}"
    )
    return _validation_error_response(
        state,
        report,
        execution_workdir,
        msg,
        last_message=safe_last_message(state.get("messages", [])),
    )


def _validation_success_response(
    state: State,
    report: dict[str, object],
    execution_workdir: Path | None,
) -> dict[str, object]:
    moved = []
    if execution_workdir:
        try:
            moved = promote_staged_outputs(execution_workdir)
        finally:
            cleanup_dir(execution_workdir)
    if moved:
        return {
            "error": "no",
            "phase": "validating",
            "output_files": moved,
            "messages": [("user", f"Outputs promoted: {moved}")],
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "validate",
                    "phase": "validating",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "no",
                    "validation_report": report,
                    "output_files": moved,
                },
            ),
        }
    return {
        "error": "no",
        "phase": "validating",
        "output_files": [],
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "validate",
                "phase": "validating",
                "iterations": int(state.get("iterations", 0) or 0),
                "error": "no",
                "validation_report": report,
                "output_files": [],
            },
        ),
    }


def _validation_coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "ok", "pass", "passed"}:
            return True
        if v in {"false", "0", "no", "n", "fail", "failed"}:
            return False
    return None


def _validation_coerce_number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _validation_collect_list(*candidates: object) -> set[str]:
    items: list[str] = []
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str):
            raw = c.strip()
            if not raw:
                continue
            if raw.startswith(("[", "{")) and raw.endswith(("]", "}")):
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = None
                if isinstance(parsed, (list, tuple, set)):
                    for x in parsed:
                        s = str(x).strip().strip("'\"")
                        if s:
                            items.append(s)
                    continue
                if isinstance(parsed, str):
                    raw = parsed.strip()
            raw = raw.strip("'\"")
            if raw:
                items.append(raw)
            continue
        if isinstance(c, (list, tuple, set)):
            for x in c:
                s = str(x).strip().strip("'\"")
                if s:
                    items.append(s)
    return set(items)


def _validation_coerce_thresholds(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    thresholds: dict[str, float] = {}
    for k, v in value.items():
        num = _validation_coerce_number(v)
        if num is not None:
            thresholds[str(k)] = float(num)
    return thresholds


def _validation_extract_policy(
    report: dict[str, object],
    metrics: object,
) -> tuple[set[str], dict[str, float], set[str], set[str]]:
    if isinstance(metrics, dict):
        allowed_missing_cols = _validation_collect_list(
            report.get("allowed_missing"),
            report.get("allowed_missing_cols"),
            report.get("missing_allowlist"),
            metrics.get("allowed_missing"),
            metrics.get("allowed_missing_cols"),
            metrics.get("missing_allowlist"),
        )
        missing_thresholds = _validation_coerce_thresholds(
            report.get("missing_thresholds")
            or report.get("allowed_missing_thresholds")
            or metrics.get("missing_thresholds")
            or metrics.get("allowed_missing_thresholds")
        )
        placeholder_required = _validation_collect_list(
            report.get("placeholder_required"),
            report.get("required_placeholder"),
            metrics.get("placeholder_required"),
            metrics.get("required_placeholder"),
        )
        placeholder_optional = _validation_collect_list(
            report.get("placeholder_optional"),
            report.get("optional_placeholder"),
            metrics.get("placeholder_optional"),
            metrics.get("optional_placeholder"),
        )
    else:
        allowed_missing_cols = set()
        missing_thresholds = {}
        placeholder_required = set()
        placeholder_optional = set()
    return allowed_missing_cols, missing_thresholds, placeholder_required, placeholder_optional


def _validation_extract_requirements(
    report: dict[str, object],
    metrics: object,
) -> dict[str, object] | None:
    report_reqs = report.get("requirements")
    if report_reqs is None and isinstance(metrics, dict):
        report_reqs = metrics.get("requirements")
    return report_reqs if isinstance(report_reqs, dict) else None


def _validation_eval_requirements(
    reqs: list,
    report_reqs: dict[str, object],
) -> tuple[list[str], list[str]]:
    missing_ids: list[str] = []
    failed_ids: list[str] = []
    for r in reqs:
        if r.id not in report_reqs:
            missing_ids.append(r.id)
            continue
        v = report_reqs.get(r.id)
        if isinstance(v, dict):
            passed = _validation_coerce_bool(v.get("ok"))
            if passed is None:
                passed = bool(v.get("ok"))
        else:
            passed = _validation_coerce_bool(v)
            if passed is None:
                passed = False
        if not passed:
            failed_ids.append(r.id)
    return missing_ids, failed_ids


def _validation_missing_placeholder_metrics(
    metrics: dict[str, object],
    allowed_missing_cols: set[str],
    placeholder_required: set[str],
    placeholder_optional: set[str],
) -> list[str]:
    prefixes: set[str] = set()
    for k in metrics.keys():
        ks = str(k)
        if ks.endswith("_missing"):
            prefixes.add(ks[: -len("_missing")])
        elif ks.endswith("_empty"):
            prefixes.add(ks[: -len("_empty")])

    missing_placeholder_metrics: list[str] = []
    for p in prefixes:
        if p in allowed_missing_cols:
            continue
        if p in placeholder_optional:
            continue
        if placeholder_required and p not in placeholder_required:
            continue
        has_placeholder = any(
            str(k).startswith(p) and any(tok in str(k).lower() for tok in ("placeholder", "fallback"))
            for k in metrics.keys()
        )
        if not has_placeholder:
            missing_placeholder_metrics.append(p)
    return missing_placeholder_metrics


def _validation_extract_missing_mapping(
    metrics: dict[str, object],
) -> tuple[int | None, list[str]]:
    missing_mapping: list[str] = []
    missing_mapping_count: int | None = None
    for k, v in metrics.items():
        key = str(k).lower()
        if key.endswith("_missing_mapping_count") and isinstance(v, (int, float)) and v > 0:
            missing_mapping_count = int(v)
        if "missing_mapping" in key and isinstance(v, (list, tuple)) and v:
            missing_mapping.extend([str(x) for x in v[:50]])
    return missing_mapping_count, missing_mapping


def _validation_collect_metric_flags(
    metrics: dict[str, object],
    allowed_missing_cols: set[str],
    missing_thresholds: dict[str, float],
    placeholder_optional: set[str],
) -> list[str]:
    metric_flags: list[str] = []
    for k, v in metrics.items():
        if type(v) is bool:
            continue
        num = _validation_coerce_number(v)
        if num is None:
            continue
        key = str(k).lower()
        if any(tok in key for tok in ("missing", "empty", "placeholder", "fallback")) and num > 0:
            col = None
            for suffix in ("_missing", "_empty", "_placeholder", "_fallback"):
                if str(k).endswith(suffix):
                    col = str(k)[: -len(suffix)]
                    break
            if col:
                if col in allowed_missing_cols:
                    continue
                if col in missing_thresholds:
                    total_missing = 0.0
                    miss_key = f"{col}_missing"
                    empty_key = f"{col}_empty"
                    total_missing += float(_validation_coerce_number(metrics.get(miss_key)) or 0.0)
                    total_missing += float(_validation_coerce_number(metrics.get(empty_key)) or 0.0)
                    if total_missing <= float(missing_thresholds[col]):
                        continue
                if col in placeholder_optional:
                    continue
            metric_flags.append(f"{k}={num}")
    return metric_flags


# =========================
# 내부 유틸 끝
# =========================



# =========================
# 노드 함수: 실제 작업 수행
# =========================

# inspect_input_node: 입력 경로 검사(파일/폴더/이미지 여부 판단)
def inspect_input_node(state: State):
    """inspect_input을 실행해 입력 상태를 구조화."""
    user_request = state.get("user_request") or extract_user_request(state.get("messages", []))
    input_path = extract_input_path(user_request or "")
    context_candidate: Optional[str] = None
    inspect_info: Optional[dict[str, Any]] = None

    if not input_path:
        context_candidate = (
            "ERROR_CONTEXT||MissingInputPath||"
            "업로드된 파일/폴더 경로를 찾지 못했습니다. 파일을 업로드한 뒤 요청 앞에 경로가 포함되었는지 확인하세요."
        )
    else:
        try:
            inspect_str = inspect_input(path=input_path)
            if isinstance(inspect_str, str) and inspect_str.startswith("ERROR_CONTEXT||"):
                context_candidate = inspect_str
            else:
                try:
                    parsed = json.loads(inspect_str)
                except Exception as exc:  # noqa: BLE001
                    context_candidate = f"ERROR_CONTEXT||InvalidJSON||{exc}"
                else:
                    if isinstance(parsed, dict):
                        inspect_info = parsed
                    else:
                        context_candidate = "ERROR_CONTEXT||InvalidPayload||inspect_input 결과가 dict가 아닙니다."
        except Exception as exc:  # noqa: BLE001
            context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    if inspect_info is None and context_candidate is None:
        context_candidate = "ERROR_CONTEXT||InvalidPayload||inspect_input 결과가 비어 있습니다."

    return {
        "inspect_result": inspect_info,
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "inspect_input_node",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "input_path": input_path,
                "inspect_result": inspect_info,
                "context_candidate": context_candidate,
            },
        ),
    }


# run_image_manifest: 이미지 폴더를 CSV 매니페스트로 정리
def run_image_manifest(state: State):
    """이미지 폴더를 list_images_to_csv로 매니페스트화."""
    user_request = state.get("user_request", "")
    context_candidate: Optional[str] = None
    try:
        inspect_info = state.get("inspect_result") or {}
        dir_path = inspect_info.get("input_path") or ""
        if not dir_path:
            context_candidate = "ERROR_CONTEXT||MissingInputPath||이미지 폴더 경로를 찾지 못했습니다."
        else:
            context_candidate = list_images_to_csv(dir_path=dir_path)
    except Exception as exc:  # noqa: BLE001
        context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    return {
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_image_manifest",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context_candidate": context_candidate,
            },
        ),
    }


# run_sample_and_summarize: 테이블 샘플링 + 요약 통합
def run_sample_and_summarize(state: State):
    """sample_table + summarize_table을 한 노드에서 수행."""
    user_request = state.get("user_request", "")
    context_candidate: Optional[str] = None
    sample_json: Optional[str] = None
    summary_context: Optional[str] = None

    try:
        inspect_info = state.get("inspect_result") or {}
        target = inspect_info.get("candidate_file") or inspect_info.get("input_path") or ""
        if not target:
            context_candidate = "ERROR_CONTEXT||MissingInputPath||테이블 경로를 찾지 못했습니다."
        else:
            sample_json = sample_table(path=target, sample_size=5000)
        if isinstance(sample_json, str) and sample_json.startswith("ERROR_CONTEXT||"):
            context_candidate = sample_json
        else:
            summary_context = summarize_table(sample_json=sample_json)
            context_candidate = summary_context
    except Exception as exc:  # noqa: BLE001
        context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    return {
        "sample_json": sample_json,
        "summary_context": summary_context,
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_sample_and_summarize",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context_candidate": context_candidate,
            },
        ),
    }


# build_context: 최종 컨텍스트 확정
def build_context(state: State):
    """중간 결과(context_candidate)를 최종 context로 확정."""
    context_candidate = state.get("context_candidate")
    if not context_candidate:
        raise ValueError("context_candidate가 비어 있습니다.")
    user_request = state.get("user_request", "")
    return {
        "context": context_candidate,
        "user_request": user_request,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "build_context",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context": context_candidate,
            },
        ),
    }


# chatbot: 요구사항 추출 + 툴 선택
def chatbot(state: State, llm_gpt: ChatOpenAI):
    """요구사항 구조화 + 툴 선택."""
    user_req = state.get("user_request") or extract_user_request(state.get("messages", []))
    context = state.get("context") or ""
    messages = list(state.get("messages", []) or [])

    reqs = []
    requirements_prompt = ""
    planned_tools = []
    reqs_error = None
    try:
        structurer = llm_gpt.with_structured_output(RequirementsPayload)
        payload = structurer.invoke(
            [
                ("system", REQUIREMENTS_SYSTEM_PROMPT),
                (
                    "user",
                    REQUIREMENTS_USER_TEMPLATE.format(user_request=user_req, context=context),
                ),
            ]
        )
        if isinstance(payload, RequirementsPayload):
            reqs = payload.requirements or []
            requirements_prompt = (payload.requirements_prompt or "").strip()
            planned_tools = payload.tool_calls or []
            if reqs:
                requirements_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
                if requirements_prompt:
                    requirements_prompt = f"{requirements_prompt}\n\n요구사항 목록:\n{requirements_list}"
                else:
                    requirements_prompt = requirements_list
    except Exception as exc:  # noqa: BLE001
        reqs_error = f"{type(exc).__name__}: {exc}"

    return {
        "messages": messages,
        "user_request": user_req,
        "requirements": reqs,
        "requirements_prompt": requirements_prompt or None,
        "planned_tools": planned_tools or None,
        "phase": "analyzing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "chatbot",
                "phase": "analyzing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_req,
                "requirements": [r.model_dump() for r in reqs],
                "requirements_prompt": requirements_prompt or None,
                "planned_tools": [t.model_dump() for t in (planned_tools or [])],
                "tool_calls": [t.model_dump() for t in (planned_tools or [])],
                "requirements_error": reqs_error,
            },
        ),
    }


# run_planned_tools: LLM이 선택한 툴을 실행해 컨텍스트 보강
def run_planned_tools(state: State):
    """요구사항에서 선택한 툴을 실행해 컨텍스트를 확장."""
    context = state.get("context") or ""
    planned = state.get("planned_tools") or []
    inspect_info = state.get("inspect_result") or {}
    default_path = inspect_info.get("candidate_file") or inspect_info.get("input_path") or ""

    tool_map = {
        "collect_unique_values": collect_unique_values,
        "mapping_coverage_report": mapping_coverage_report,
        "collect_rare_values": collect_rare_values,
        "detect_parseability": detect_parseability,
        "detect_encoding": detect_encoding,
        "column_profile": column_profile,
    }

    tool_reports: list[dict[str, Any]] = []
    seen: set[str] = set()

    timeout_only_tools = {
        "collect_unique_values",
        "mapping_coverage_report",
        "collect_rare_values",
        "detect_parseability",
        "column_profile",
    }
    limit_keys_by_tool = {
        "collect_unique_values": ["max_rows", "max_unique", "max_values_return"],
        "mapping_coverage_report": ["max_rows", "max_unique", "max_values_return"],
        "collect_rare_values": ["max_rows", "max_values_return"],
        "detect_parseability": ["max_rows", "max_samples"],
        "column_profile": ["max_rows", "max_columns", "sample_values_limit"],
    }

    for tool_call in planned:
        name, args, reason = _toolcall_parse_entry(tool_call)
        if not name:
            continue
        args = _toolcall_apply_defaults(
            name,
            args,
            default_path,
            timeout_only_tools,
            limit_keys_by_tool,
        )
        key = _toolcall_build_dedup_key(name, args)
        if key in seen:
            continue
        seen.add(key)

        tool_fn = tool_map.get(name)
        if tool_fn is None:
            output = f"ERROR_CONTEXT||UnknownTool||{name}"
        else:
            try:
                output = tool_fn.invoke(args)
            except Exception as exc:  # noqa: BLE001
                output = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

        tool_reports.append(
            {
                "name": name,
                "args": args,
                "reason": reason,
                "output": output,
            }
        )

    context = _toolcall_format_reports(tool_reports, context)

    return {
        "context": context,
        "tool_reports": tool_reports or None,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_planned_tools",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "tool_reports": tool_reports,
            },
        ),
    }

# generate: 컨텍스트+요청으로 코드 초안 생성
def generate(state: State, llm_coder: ChatOpenAI, llm_gpt: ChatOpenAI):
    """코드 생성 LLM 호출."""
    context = state.get("context", "")
    user_request = state.get("user_request", "")
    output_formats = state.get("output_formats") or "csv"
    reqs = state.get("requirements") or []
    requirements_prompt = (state.get("requirements_prompt") or "").strip()
    requirements_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs]) if reqs else ""
    requirements_prompt_text = requirements_prompt or requirements_list or "(none)"
    requirement_ids = ", ".join([r.id for r in reqs]) if reqs else "(none)"

    generated_code = llm_coder.invoke(
        code_gen_prompt.format_messages(
            context=context,
            user_request=user_request,
            output_formats=output_formats,
            requirements_prompt=requirements_prompt_text,
            requirement_ids=requirement_ids,
        )
    )
    code_structurer = llm_gpt.with_structured_output(CodeBlocks)
    code_solution = code_structurer.invoke(generated_code.content)

    messages = [
        (
            "assistant",
            f"Imports: {code_solution.imports}\nCode: {code_solution.code}",
        ),
    ]

    return {
        "generation": code_solution,
        "messages": messages,
        "phase": "generating",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "generate",
                "phase": "generating",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "output_formats": output_formats,
                "requirements": [r.model_dump() for r in (reqs or [])],
                "generation": {"imports": code_solution.imports, "code": code_solution.code},
            },
        ),
    }

# code_check: 생성 코드 실행 및 에러 감지
def code_check(state: State):
    """생성된 코드 실행 및 에러 감지."""
    code_solution = state["generation"]
    imports = code_solution.imports
    code = code_solution.code

    sample_fallback_reason = detect_sample_fallback(code)
    if sample_fallback_reason:
        error_detail = (
            "샘플 데이터 생성(폴백) 로직이 감지되어 실행을 중단합니다. "
            "실제 파일이 없으면 즉시 실패해야 하며, 샘플 데이터를 만들면 안 됩니다.\n"
            f"감지 사유: {sample_fallback_reason}"
        )
        return _code_check_error_response(
            state,
            code_solution,
            error_detail,
            "",
        )

    exec_globals, output_preview, error_kind, error_detail = _exec_run_generated_code(
        imports, code
    )
    execution_workdir = exec_globals.get("_execution_workdir")
    if error_kind:
        if execution_workdir:
            cleanup_dir(execution_workdir)
        return _code_check_error_response(
            state,
            code_solution,
            error_detail,
            output_preview,
            error_kind=error_kind,
        )
    preview_tail = output_preview[-2000:] if len(output_preview) > 2000 else output_preview
    message = [("user", f"Execution preview (truncated):\n{preview_tail}")]
    validation_report = exec_globals.get("__validation_report__")
    return {
        "generation": code_solution,
        "messages": message,
        "error": "no",
        "phase": "executing",
        "execution_stdout": output_preview,
        "validation_report": validation_report if isinstance(validation_report, dict) else None,
        "execution_workdir": str(execution_workdir) if execution_workdir else "",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "code_check",
                "phase": "executing",
                "iterations": int(state.get("iterations", 0) or 0),
                "error": "no",
                "execution_stdout": output_preview,
                "validation_report": validation_report if isinstance(validation_report, dict) else None,
                "generation": {"imports": imports, "code": code},
            },
        ),
    }

# validate: 실행 결과(검증 리포트)를 기반으로 누락(silent miss) 감지
def validate(state: State):
    """실행이 성공했더라도 검증 리포트에 문제가 있으면 실패로 간주."""
    report = state.get("validation_report")
    reqs = state.get("requirements") or []
    workdir_str = state.get("execution_workdir") or ""
    execution_workdir = Path(workdir_str) if workdir_str else None

    # 생성 스크립트가 validation report를 만들지 못하면, 강제로 수정(reflect) 루프로 보낸다.
    if not isinstance(report, dict):
        return _validation_fail_missing_report(state, report, execution_workdir)

    if "ok" not in report or "issues" not in report:
        return _validation_fail_missing_ok_issues(state, report, execution_workdir)

    ok_raw = report.get("ok")
    ok = _validation_coerce_bool(ok_raw)
    if ok is None:
        ok = False

    # 방어적 처리: metrics가 결측/placeholder/fallback 사용을 시사하면 ok=True를 그대로 신뢰하지 않는다.
    metrics = report.get("metrics")
    if ok is True and not isinstance(metrics, dict):
        return _validation_fail_missing_metrics(state, report, execution_workdir)

    # 결측/placeholder 정책: 기본은 엄격, 명시된 경우만 완화
    allowed_missing_cols, missing_thresholds, placeholder_required, placeholder_optional = _validation_extract_policy(
        report, metrics
    )

    # 사용자 요구사항 강제: 요구사항 누락/실패 시 리팩트(reflect) 루프로 보낸다.
    if reqs:
        report_reqs = _validation_extract_requirements(report, metrics)
        if report_reqs is None:
            return _validation_fail_missing_requirements(state, report, execution_workdir, reqs)
        missing_ids, failed_ids = _validation_eval_requirements(reqs, report_reqs)
        if missing_ids or failed_ids:
            return _validation_fail_requirements(
                state,
                report,
                execution_workdir,
                reqs,
                missing_ids,
                failed_ids,
            )

    # metrics에 "<col>_missing"/"<col>_empty"가 있으면, 대응하는 placeholder/fallback 메트릭도 요구한다.
    # "정보 없음" 같은 placeholder로 미매핑 값을 숨기는 것을 방지한다.
    if isinstance(metrics, dict):
        missing_placeholder_metrics = _validation_missing_placeholder_metrics(
            metrics,
            allowed_missing_cols,
            placeholder_required,
            placeholder_optional,
        )
        if ok is True and missing_placeholder_metrics:
            return _validation_fail_missing_placeholder_metrics(
                state,
                report,
                execution_workdir,
                missing_placeholder_metrics,
            )

        # 매핑 누락 감지: *_missing_mapping(_count) 지표가 있으면 반드시 실패 처리
        missing_mapping_count, missing_mapping = _validation_extract_missing_mapping(metrics)
        if missing_mapping_count is not None or missing_mapping:
            return _validation_fail_missing_mapping(
                state,
                report,
                execution_workdir,
                missing_mapping_count,
                missing_mapping,
            )

    metric_flags: list[str] = []
    if isinstance(metrics, dict):
        metric_flags = _validation_collect_metric_flags(
            metrics,
            allowed_missing_cols,
            missing_thresholds,
            placeholder_optional,
        )

    if ok is True and not metric_flags:
        return _validation_success_response(state, report, execution_workdir)

    # ok=True가 명시되지 않으면 실패로 처리(요구사항 누락(silent miss) 방지)
    issues = report.get("issues")
    if issues is None:
        issues = report.get("errors")
    if isinstance(issues, str):
        issues_list = [issues]
    elif isinstance(issues, list):
        issues_list = [str(x) for x in issues[:50]]
    else:
        issues_list = []

    return _validation_fail_silent_miss(
        state,
        report,
        execution_workdir,
        issues_list,
        metric_flags,
    )


# reflect: 실행 오류를 기반으로 수정 코드 재생성
def reflect(state: State, llm_coder: ChatOpenAI, llm_gpt: ChatOpenAI):
    """에러 발생 시 수정 코드 생성."""
    error = extract_last_message_text(state.get("messages", []))
    code_solution = state["generation"]
    code_solution_str = (
        f"Imports: {code_solution.imports} \n Code: {code_solution.code}"
    )
    prev_generation = {
        "imports": getattr(code_solution, "imports", ""),
        "code": getattr(code_solution, "code", ""),
    }

    corrected_code = llm_coder.invoke(
        reflect_prompt.format_messages(error=error, code_solution=code_solution_str)
    )
    code_structurer = llm_gpt.with_structured_output(CodeBlocks)
    reflections = code_structurer.invoke(corrected_code.content)
    next_generation = {"imports": reflections.imports, "code": reflections.code}
    diff_text, diff_summary = diff_generation(prev_generation, next_generation)

    messages = [
        (
            "assistant",
            f"Imports: {reflections.imports} \n Code: {reflections.code}",
        )
    ]

    return {
        "generation": reflections,
        "messages": messages,
        "iterations": state["iterations"] + 1,
        "phase": "refactoring",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "reflect",
                "phase": "refactoring",
                "iterations": int(state.get("iterations", 0) or 0) + 1,
                "error_input": error,
                "generation": {"imports": reflections.imports, "code": reflections.code},
                "prev_generation": prev_generation,
                "diff_summary": diff_summary,
                "diff": diff_text,
            },
        ),
    }


# friendly_error: 오류 컨텍스트/최종 오류를 비개발자용 한글 메시지로 변환
def friendly_error(state: State, llm_gpt: ChatOpenAI):
    """원문 에러 메시지를 비개발자용 한글 요약으로 변환."""
    context = state.get("context", "")
    user_request = state.get("user_request", "")
    if isinstance(context, str) and context.startswith("ERROR_CONTEXT||"):
        parts = context.split("||", 2)  # ERROR_CONTEXT||<exc_name>||<exc_message>
        exc_name = parts[1] if len(parts) > 1 else "UnknownError"
        exc_msg = parts[2] if len(parts) > 2 else context
        prompt = _build_friendly_error_prompt(
            user_request,
            exc_name=exc_name,
            exc_msg=exc_msg,
        )
        trace_payload = {
            "error_source": "context",
            "context": context,
            "exc_name": exc_name,
            "exc_msg": exc_msg,
        }
    else:
        raw_error = extract_last_message_text(state.get("messages", []))
        raw_error = raw_error[-4000:] if isinstance(raw_error, str) else str(raw_error)
        prompt = _build_friendly_error_prompt(
            user_request,
            raw_error=raw_error,
        )
        trace_payload = {
            "error_source": "final",
            "raw_error": raw_error,
        }
    return _friendly_error_response(
        state,
        llm_gpt,
        user_request=user_request,
        node_name="friendly_error",
        prompt=prompt,
        trace_payload=trace_payload,
    )


# =========================
# 노드 함수 끝
# =========================

# =========================
# 라우터: 조건 분기 함수
# =========================
# route_after_inspect_input: 검사 결과에 따라 이미지/샘플/에러 분기
def route_after_inspect_input(state: State):
    ctx = state.get("context_candidate", "")
    if isinstance(ctx, str) and ctx.startswith("ERROR_CONTEXT||"):
        return "build_context"
    info = state.get("inspect_result") or {}
    if isinstance(info, dict) and info.get("has_images"):
        return "run_image_manifest"
    return "run_sample_and_summarize"

# route_after_context: 최종 컨텍스트에 오류 마커가 있으면 친절한 에러 노드로 분기
# route_after_context: 컨텍스트 오류 여부에 따른 분기
def route_after_context(state: State):
    """컨텍스트에 오류 마커가 있으면 친절한 에러 노드로 분기."""
    ctx = state.get("context", "")
    if isinstance(ctx, str) and ctx.startswith("ERROR_CONTEXT||"):
        return "friendly_error"
    return "chatbot"

# decide_to_finish: 실패 시 reflect/최종에러 분기 결정
def decide_to_finish(state: State, max_iterations: int):
    """실패 시 reflect/최종에러 분기 결정."""
    error = state["error"]
    iterations = state["iterations"]
    if error == "no":
        return "end"
    if iterations == max_iterations:
        return "friendly_error"
    return "reflect"


# route_after_execution: code_check 이후 분기
def route_after_execution(state: State, max_iterations: int):
    """code_check 이후: 실행 성공이면 validate로, 실패면 reflect/최종에러로."""
    if state.get("error") == "no":
        return "validate"
    return decide_to_finish(state, max_iterations=max_iterations)
# =========================
# 라우터 끝
# =========================

# =========================
# 그래프 빌드: 노드/엣지 추가
# =========================
def build_graph(
    llm_model: str = "gpt-4o-mini",
    coder_model: str = "gpt-4.1",
    max_iterations: int = 5,
):
    """LangGraph를 구성하고 컴파일된 그래프를 반환."""
    llm_gpt = ChatOpenAI(model=llm_model)
    llm_coder = ChatOpenAI(model=coder_model, temperature=0)

    graph_builder = StateGraph(State)

    # ===== 노드: 그래프에 노드 추가 =====
    # 입력 검사 + 샘플링/요약 단계 구성
    graph_builder.add_node("inspect_input_node", inspect_input_node)
    graph_builder.add_node("run_image_manifest", run_image_manifest)
    graph_builder.add_node("run_sample_and_summarize", run_sample_and_summarize)
    graph_builder.add_node("build_context", build_context)
    # 요구사항 정리 + 툴 선택 + 툴 실행
    graph_builder.add_node("chatbot", partial(chatbot, llm_gpt=llm_gpt))
    graph_builder.add_node("run_planned_tools", run_planned_tools)
    # 에러/코드 생성/실행/검증/리플렉트
    graph_builder.add_node("friendly_error", partial(friendly_error, llm_gpt=llm_gpt))
    graph_builder.add_node("generate", partial(generate, llm_coder=llm_coder, llm_gpt=llm_gpt))
    graph_builder.add_node("code_check", code_check)
    graph_builder.add_node("validate", validate)
    graph_builder.add_node("reflect", partial(reflect, llm_coder=llm_coder, llm_gpt=llm_gpt))
    # ===== 노드 끝 =====

    # ===== 엣지: 노드 연결 정의 =====
    graph_builder.add_edge(START, "inspect_input_node")

    # inspect_input 이후: 이미지/샘플/에러 분기
    graph_builder.add_conditional_edges(
        "inspect_input_node",
        route_after_inspect_input,
        {
            "run_image_manifest": "run_image_manifest",
            "run_sample_and_summarize": "run_sample_and_summarize",
            "build_context": "build_context",
        },
    )

    # sample+summarize/image 결과는 build_context로 모아서 통일
    graph_builder.add_edge("run_sample_and_summarize", "build_context")
    graph_builder.add_edge("run_image_manifest", "build_context")

    # build_context 이후: 오류면 friendly_error, 정상이면 chatbot
    graph_builder.add_conditional_edges(
        "build_context",
        route_after_context,
        {"friendly_error": "friendly_error", "chatbot": "chatbot"},
    )

    # chatbot 이후: 선택된 툴 실행
    graph_builder.add_edge("chatbot", "run_planned_tools")

    # tool 실행 후: 코드 생성
    graph_builder.add_edge("run_planned_tools", "generate")

    # generate → code_check
    graph_builder.add_edge("generate", "code_check")

    # code_check 이후: 성공 → validate, 실패 → reflect/final
    graph_builder.add_conditional_edges(
        "code_check",
        partial(route_after_execution, max_iterations=max_iterations),
        {"validate": "validate", "reflect": "reflect", "friendly_error": "friendly_error"},
    )

    # validate 이후: 통과면 END, 실패면 reflect/final
    graph_builder.add_conditional_edges(
        "validate",
        partial(decide_to_finish, max_iterations=max_iterations),
        {"end": END, "reflect": "reflect", "friendly_error": "friendly_error"},
    )

    # reflect → code_check
    graph_builder.add_edge("reflect", "code_check")

    # friendly_error → END
    graph_builder.add_edge("friendly_error", END)
    # ===== 엣지 끝 =====

    return graph_builder.compile()
# =========================
# 그래프 빌드 끝
# =========================

# =========================
# 엔트리포인트: 그래프 실행 함수
# =========================
def run_request(
    request: str,
    max_iterations: int = 3,
    llm_model: str = "gpt-4o-mini",
    coder_model: str = "gpt-4.1",
    output_formats: str | None = None,
) -> Dict[str, Any]:
    """사용자 요청을 받아 그래프를 실행하고 결과를 반환."""
    graph = build_graph(llm_model=llm_model, coder_model=coder_model, max_iterations=max_iterations)
    run_id = uuid4().hex
    initial_state: Dict[str, Any] = {
        "run_id": run_id,
        "messages": [("user", request)],
        "iterations": 0,
        "error": "",
        "context": "",
        "context_candidate": None,
        "generation": None,
        "phase": None,
        "user_request": request,
        "output_formats": output_formats,
        "final_user_messages": None,
        "trace": [],
        "llm_model": llm_model,
        "coder_model": coder_model,
        "tool_call_name": None,
        "tool_call_args": None,
        "planned_tools": None,
        "tool_reports": None,
        "inspect_result": None,
        "sample_json": None,
        "summary_context": None,
    }
    result = graph.invoke(initial_state)
    if isinstance(result, dict):
        trace_name = write_internal_trace_markdown(result)
        if trace_name:
            files = result.get("output_files") or []
            if not isinstance(files, list):
                files = []
            if trace_name not in files:
                files.append(trace_name)
            result["output_files"] = files
    return result
# =========================
# 엔트리포인트 끝
# =========================


__all__ = ["build_graph", "run_request"]
