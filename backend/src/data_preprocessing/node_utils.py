from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from langchain_openai import ChatOpenAI

from .common_utils import (
    append_trace,
    cleanup_dir,
    now_iso,
    prepare_execution_workdir,
    promote_staged_outputs,
    run_generated_code,
    safe_format_json_like,
    safe_last_message,
    truncate_text,
)
from .constants import (
    _CATEGORICAL_EXAMPLE_ROWS,
    _CATEGORICAL_TOP_COLS,
    _EXT_PRIORITY,
    _HF_METADATA_FILENAMES,
    _IMAGE_EXTS,
    _MAX_FEATHER_MB,
    _MAX_JSON_FULL_LOAD_MB,
    _MISSING_TOP_N,
    _PREVIEW_MAX_COLS,
    _PREVIEW_ROWS,
    _SUMMARY_SAMPLE_SIZE,
    _SUMMARY_SAMPLE_SIZE_BY_EXT,
    _SUPPORTED_EXTS,
    _VALUE_REPR_LIMIT,
)
from .models import CodeBlocks, State


from .io_utils import (
    _strip_query,
    _resolve_path,
    _load_json_payload,
    _detect_text_format,
    _read_arrow_ipc_sample,
    _read_parquet_sample,
    _read_table_like,
    _pick_candidate_from_dir,
    _build_image_manifest,
    _normalize_image_exts,
)


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


def _choose_sample_size(path: Path) -> int:
    ext = path.suffix.lower()
    return _SUMMARY_SAMPLE_SIZE_BY_EXT.get(ext, _SUMMARY_SAMPLE_SIZE)


def sample_table(path: str, sample_size: int | None = None) -> str:
    p = _resolve_path(path)

    if p.is_dir():
        has_image = any(f.suffix.lower() in _IMAGE_EXTS for f in p.rglob("*") if f.is_file())
        if has_image:
            return f"ERROR_CONTEXT||ImageFolder||{p} 아래에 이미지가 있어 테이블 샘플링을 생략합니다."
        candidate = _pick_candidate_from_dir(p)
        if candidate is None:
            return f"ERROR_CONTEXT||NoSupportedFiles||{p} 아래에서 지원하는 데이터 파일을 찾지 못했습니다."
        p = candidate

    if sample_size is None:
        sample_size = _choose_sample_size(p)

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
    columns = [str(c) for c in df.columns]
    columns_md = (
        pd.DataFrame({"column": columns}).to_markdown(index=False) if columns else "(컬럼 없음)"
    )
    dtypes_md = (
        df.dtypes.astype(str)
        .rename("dtype")
        .reset_index()
        .rename(columns={"index": "column"})
        .to_markdown(index=False)
    )
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        missing_df = missing.head(_MISSING_TOP_N).reset_index()
        missing_df.columns = ["column", "missing"]
        denom = max(len(df), 1)
        missing_df["missing_rate"] = (missing_df["missing"] / denom).round(6)
        missing_md = missing_df.to_markdown(index=False)
    else:
        missing_md = "(결측치 없음)"
    data_path = payload.get("data_path", "")
    fmt = payload.get("detected_format", "")
    sample_rows = payload.get("sample_rows", len(df))

    return (
        f"data_path: {data_path}\n"
        f"detected_format: {fmt}\n"
        f"sample_rows: {sample_rows}\n"
        f"columns: {len(columns)}\n"
        "columns_list:\n"
        f"{columns_md}\n\n"
        "dtypes:\n"
        f"{dtypes_md}\n\n"
        f"missing (top {_MISSING_TOP_N}):\n"
        f"{missing_md}\n"
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
    allowed_keys_by_tool: dict[str, set[str]],
) -> dict[str, object]:
    if "path" not in args and default_path:
        args["path"] = default_path
    elif default_path and isinstance(args.get("path"), str):
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            args["path"] = default_path
        else:
            lowered = raw_path.lower()
            if raw_path in {"data_path", "<data_path>", "path", "<path>"} or any(
                token in lowered
                for token in (
                    "path/to",
                    "your/data",
                    "your/file",
                    "example",
                    "sample",
                    "<data_path>",
                    "<path>",
                    "{path}",
                    "{data_path}",
                )
            ):
                args["path"] = default_path
            else:
                try:
                    candidate = _resolve_path(raw_path)
                except Exception:
                    candidate = None
                if candidate is not None and not candidate.exists():
                    try:
                        if _resolve_path(default_path).exists():
                            args["path"] = default_path
                    except Exception:
                        args["path"] = default_path
    if name in timeout_only_tools:
        args["time_limit_sec"] = 60
    if name in allowed_keys_by_tool:
        allowed = allowed_keys_by_tool[name]
        args = {k: v for k, v in args.items() if k in allowed}
    # Drop None/empty placeholders to keep tool args clean.
    args = {k: v for k, v in args.items() if v is not None}
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
    iterations: int | None = None,
) -> dict[str, object]:
    resp = llm_gpt.invoke(prompt)
    if iterations is None:
        iterations = int(state.get("iterations", 0) or 0)
    return {
        "final_user_messages": [("assistant", resp.content)],
        "error": "yes",
        "phase": "error",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": node_name,
                "phase": "error",
                "iterations": iterations,
                "user_request": user_request,
                "assistant_message": resp.content,
                **trace_payload,
            },
        ),
    }

# =========================
# 내부 유틸: 검증 노드 전용
# =========================


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
    moved: list[str] = []
    if execution_workdir:
        try:
            moved = promote_staged_outputs(execution_workdir)
        finally:
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
    if moved:
        trace_payload["output_files"] = moved
    if stdout_tail is not None:
        trace_payload["execution_stdout_tail"] = stdout_tail
    if extra_trace:
        trace_payload.update(extra_trace)
    return {
        "messages": [("user", msg)],
        "error": "yes",
        "phase": "validating",
        "output_files": moved,
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
    if value is None:
        return None
    # numpy 스칼라를 먼저 파이썬 타입으로 정규화
    if hasattr(value, "item") and type(value).__module__.startswith("numpy"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return value
    # numpy.bool_ 등 bool 유사 타입 지원
    if type(value).__name__ in {"bool_", "bool8"}:
        return bool(value)
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


def _validation_normalize_requirements(
    report_reqs: dict[str, object],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in report_reqs.items():
        if isinstance(value, dict):
            ok_raw = value.get("ok")
            ok = _validation_coerce_bool(ok_raw)
            if ok is None and ok_raw is not None:
                try:
                    ok = bool(ok_raw)
                except Exception:
                    ok = None
            if ok is not None:
                updated = dict(value)
                updated["ok"] = ok
                normalized[key] = updated
            else:
                normalized[key] = value
            continue
        ok = _validation_coerce_bool(value)
        if ok is None:
            ok = bool(value) if isinstance(value, (int, float, bool)) else False
        normalized[key] = ok
    return normalized


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
        if any(tok in key for tok in ("before", "after", "raw", "input", "original", "orig")):
            continue
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
