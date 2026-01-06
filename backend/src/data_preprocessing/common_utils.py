from __future__ import annotations

import contextlib
import difflib
import io
import json
import os
import re
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from .models import State

# ========================== 시간/경로 유틸 ==========================
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def outputs_root_dir() -> Path:
    # backend/src/data_preprocessing/common_utils.py 기준 parents[2] == backend/
    return Path(__file__).resolve().parents[2] / "outputs"

# ========================== Used by: inspect_input_node ==========================
_PATH_HINT_EXTS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".parquet",
    ".feather",
    ".arrow",
    ".xlsx",
    ".xls",
    ".zip",
}


def extract_input_path(user_request: str) -> str | None:
    text = (user_request or "").strip()
    if not text:
        return None
    if text[0] in {"'", '"'}:
        quote = text[0]
        end = text.find(quote, 1)
        if end > 1:
            candidate = text[1:end].strip()
            if candidate:
                return candidate
    candidate = text.split()[0].strip()
    if not candidate:
        return None
    if candidate.startswith(("s3://", "/", "./", "../", "~")):
        return candidate
    if "/" in candidate or "\\" in candidate:
        return candidate
    if Path(candidate).suffix.lower() in _PATH_HINT_EXTS:
        return candidate
    try:
        if Path(candidate).expanduser().resolve().exists():
            return candidate
    except Exception:
        pass
    return None

# ========================== Trace/메시지 유틸 ==========================
def append_trace(state: State | dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    existing = None
    if isinstance(state, dict):
        existing = state.get("trace")
    else:
        existing = getattr(state, "trace", None)
    trace = list(existing or [])
    trace.append(event)
    return {"trace": trace}


def extract_last_message_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, tuple) and len(last) >= 2:
        return str(last[1])
    if hasattr(last, "content"):
        return str(getattr(last, "content", ""))
    return str(last)


def safe_last_message(messages: list[Any]) -> str:
    try:
        return extract_last_message_text(messages)
    except Exception:
        return ""


def extract_user_request(messages: list[Any]) -> str:
    if not messages:
        return ""
    first = messages[0]
    if isinstance(first, tuple):
        return first[1]
    if hasattr(first, "content"):
        return getattr(first, "content", "")
    return ""

# ========================== 포맷/텍스트 유틸 ==========================
def truncate_text(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n…(truncated)"


def safe_format_json_like(obj: object, limit: int = 6000) -> str:
    """디버그 메시지용 안전한 포맷팅(최대한 시도하며, 예외를 발생시키지 않음)."""
    try:
        return truncate_text(json.dumps(obj, ensure_ascii=False, indent=2), limit=limit)
    except Exception as exc:  # noqa: BLE001
        return truncate_text(f"<non-json-serializable: {type(exc).__name__}: {exc}>\n{repr(obj)}", limit=limit)


def detect_sample_fallback(code: str) -> str | None:
    """샘플 데이터 생성/폴백 로직을 휴리스틱으로 감지."""
    if not code:
        return None
    lower = code.lower()
    if "sample_data" in lower:
        return "sample_data variable found"
    if "create a sample dataframe" in lower:
        return "sample dataframe comment"
    lines = code.splitlines()
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if "if not" in line_lower and "exists(" in line_lower:
            window = "\n".join(lines[i : i + 15]).lower()
            if "pd.dataframe" in window or "sample_data" in window:
                return "pd.DataFrame created under missing-file guard"
    return None


# ========================== Used by: code_check ==========================
def prepare_execution_workdir() -> Path:
    staging_root = outputs_root_dir() / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    execution_workdir = staging_root / f"run_{uuid4().hex}"
    execution_workdir.mkdir(parents=True, exist_ok=True)
    return execution_workdir


# 필요한 경우 최상위 또는 함수 내부에서 requests를 import합니다.
# 최상위 imports에 추가하지 않았으므로, 인라인 import를 허용하거나 사용 가능하다고 가정합니다.
# 하지만 깔끔함을 위해 imports를 확인하는 것이 좋습니다. 안전을 위해 인라인 import를 사용합니다.

def run_generated_code(
    imports: str,
    code: str,
    execution_workdir: Path,
) -> tuple[dict[str, Any], str, str | None, str | None]:
    
    # Executor URL 정의
    # Docker 네트워크에서 'executor'는 호스트 이름입니다. 기본 포트 8000.
    EXECUTOR_URL = os.environ.get("EXECUTOR_URL", "http://executor:8000/execute")

    # workdir 상대 경로 계산 (backend/outputs 기준)
    # 예: /app/backend/outputs/_staging/run_123 -> _staging/run_123
    try:
        root_dir = outputs_root_dir().resolve()
        relative_workdir = str(execution_workdir.resolve().relative_to(root_dir))
    except ValueError:
        # 경로가 일치하지 않으면 기본값 사용 (혹은 에러 처리)
        relative_workdir = ""

    payload = {
        "code": code,
        "imports": imports,
        "workdir": relative_workdir
    }

    try:
        # 타임아웃을 설정하여 요청 (예: 긴 처리를 위해 60초)
        response = requests.post(EXECUTOR_URL, json=payload, timeout=60)
        
        # 네트워크/HTTP 에러 확인
        if response.status_code != 200:
            return {}, "", "connection_error", f"Executor returned status {response.status_code}: {response.text}"
            
        data = response.json()
        
        # 응답 파싱
        stdout_captured = data.get("stdout", "")
        error_kind = data.get("error_kind")
        error_detail = data.get("error_detail")
        results = data.get("results", {})
        
        # 결과를 통해 exec_globals 재구성
        # 요청한 항목(validation_report 등)만 반환받으므로
        # 실제 globals의 부분집합이지만, 워크플로우에는 충분합니다.
        exec_globals = results
        
        if error_kind:
            # 일관성을 위해 에러 상세 내용을 stdout에 추가
            if not error_detail:
                error_detail = "Unknown error occurred in executor."
            full_log = stdout_captured + "\n" + error_detail
            return exec_globals, stdout_captured, error_kind, full_log
            
        return exec_globals, stdout_captured, None, None

    except Exception as e:
        return {}, "", "execution_request_failed", f"Failed to connect to executor: {str(e)}"


# ========================== 파일/출력 유틸 ==========================
def cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def promote_staged_outputs(execution_workdir: Path) -> list[str]:
    """execution_workdir 아래에서 생성된 ./outputs를 backend/outputs로 옮기고, 옮긴 파일명을 반환."""
    src = execution_workdir / "outputs"
    if not src.exists():
        return []

    dst_root = outputs_root_dir()
    dst_root.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for item in src.iterdir():
        target = dst_root / item.name
        if target.exists():
            # 충돌 방지: 기존 파일을 보존하고 새 파일명을 부여
            target = dst_root / f"{item.stem}_{uuid4().hex[:8]}{item.suffix}"
        shutil.move(str(item), str(target))
        moved.append(target.name)
    return moved


# ========================== 코드/변경 유틸 ==========================
def generation_to_text(gen: object) -> str:
    if gen is None:
        return ""
    if isinstance(gen, dict):
        imports = str(gen.get("imports") or "")
        code = str(gen.get("code") or "")
    else:
        imports = str(getattr(gen, "imports", "") or "")
        code = str(getattr(gen, "code", "") or "")
    return f"{imports}\n\n{code}".strip()


def diff_generation(prev: object, nxt: object) -> tuple[str, dict[str, int]]:
    prev_text = generation_to_text(prev)
    next_text = generation_to_text(nxt)
    diff_lines = list(
        difflib.unified_diff(
            prev_text.splitlines(),
            next_text.splitlines(),
            fromfile="before.py",
            tofile="after.py",
            lineterm="",
        )
    )
    added = 0
    removed = 0
    for line in diff_lines:
        if line.startswith(("---", "+++", "@@")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    summary = {"added": added, "removed": removed}
    return "\n".join(diff_lines), summary


# ========================== 내부 기록(Trace) 작성 ==========================
def write_internal_trace_markdown(state: dict[str, Any]) -> str | None:
    """누적된 trace를 Markdown 파일로 backend/outputs에 저장하고 파일명을 반환."""
    trace = state.get("trace") or []
    if not isinstance(trace, list) or not trace:
        return None

    run_id = str(state.get("run_id") or "").strip()
    if not run_id:
        return None

    out_dir = outputs_root_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"run_{run_id}_internal_trace_내부기록.md"
    path = (out_dir / filename).resolve()
    try:
        path.relative_to(out_dir.resolve())
    except Exception:
        return None

    def _md_code(lang: str, text: str) -> str:
        return f"```{lang}\n{text}\n```"

    def _truncate(text: str, limit: int = 8000) -> str:
        text = str(text)
        return text if len(text) <= limit else text[:limit] + "\n…(truncated)"

    def _derive_decision_basis(ev: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        report = ev.get("validation_report")
        if isinstance(report, dict):
            ok = report.get("ok")
            reasons.append(f"validation_report.ok={ok}")
            issues = report.get("issues")
            if isinstance(issues, list) and issues:
                reasons.append(f"issues: {issues[:10]}")
            elif isinstance(issues, str) and issues:
                reasons.append(f"issues: {issues}")
            reqs = report.get("requirements")
            if isinstance(reqs, dict):
                failed = []
                for k, v in reqs.items():
                    if isinstance(v, bool):
                        passed = v
                    elif isinstance(v, dict):
                        passed = bool(v.get("ok"))
                    else:
                        passed = False
                    if not passed:
                        failed.append(k)
                if failed:
                    reasons.append(f"requirements_failed: {failed}")
            metrics = report.get("metrics")
            if isinstance(metrics, dict):
                flagged = []
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and v > 0:
                        key = str(k).lower()
                        if any(tok in key for tok in ("missing", "empty", "placeholder", "fallback")):
                            flagged.append(f"{k}={v}")
                if flagged:
                    reasons.append(f"metrics_flags: {flagged}")
        error_detail = ev.get("error_detail") or ev.get("raw_error")
        if error_detail:
            reasons.append(f"execution_error: {_truncate(error_detail, 500)}")
        last_message = ev.get("last_message")
        if last_message:
            reasons.append(f"last_message: {_truncate(last_message, 300)}")
        error_input = ev.get("error_input")
        if error_input:
            reasons.append(f"reflect_input: {_truncate(error_input, 300)}")
        return reasons

    def _summarize_metrics(metrics: dict[str, Any], max_items: int = 6) -> list[str]:
        items: list[str] = []
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                items.append(f"{k}={v}")
            if len(items) >= max_items:
                break
        return items

    def _build_decision_basis_report(report: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        ok = report.get("ok")
        reasons.append(f"validation_report.ok={ok}")
        issues = report.get("issues")
        if isinstance(issues, list) and issues:
            reasons.append(f"issues: {issues[:10]}")
        elif isinstance(issues, str) and issues:
            reasons.append(f"issues: {issues}")
        reqs = report.get("requirements")
        if isinstance(reqs, dict):
            failed = []
            for k, v in reqs.items():
                if isinstance(v, bool):
                    passed = v
                elif isinstance(v, dict):
                    passed = bool(v.get("ok"))
                else:
                    passed = False
                if not passed:
                    failed.append(k)
            reasons.append(f"requirements_total={len(reqs)}")
            if failed:
                reasons.append(f"requirements_failed: {failed}")
        metrics = report.get("metrics")
        if isinstance(metrics, dict):
            m = _summarize_metrics(metrics)
            if m:
                reasons.append(f"metrics: {m}")
        return reasons

    lines: list[str] = []
    lines.append("# 내부 기록 (Internal Trace)")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- created_at: `{now_iso()}`")
    lines.append(f"- output_formats: `{state.get('output_formats')}`")
    reflect_events = [ev for ev in trace if isinstance(ev, dict) and ev.get("node") == "reflect"]
    lines.append(f"- reflect_count: `{len(reflect_events)}`")
    lines.append("")

    final_files = state.get("output_files") or []
    if isinstance(final_files, list):
        lines.append("## 최종 산출물")
        lines.append("")
        if final_files:
            for f in final_files:
                lines.append(f"- `{f}`")
        else:
            lines.append("- (none)")
        lines.append("")

    lines.append("## 이벤트 로그")
    lines.append("")

    prev_validation_report: dict[str, Any] | None = None

    for idx, ev in enumerate(trace, start=1):
        if not isinstance(ev, dict):
            lines.append(f"### Step {idx}")
            lines.append("")
            lines.append(_md_code("text", str(ev)))
            lines.append("")
            continue

        node = str(ev.get("node") or "")
        phase = str(ev.get("phase") or "")
        ts = str(ev.get("ts") or "")
        it = ev.get("iterations")
        title_bits = [b for b in [node or "event", phase] if b]
        title = " / ".join(title_bits)
        lines.append(f"### Step {idx}: {title}")
        lines.append("")
        if ts:
            lines.append(f"- ts: `{ts}`")
        if it is not None:
            lines.append(f"- iterations: `{it}`")
        if "error" in ev:
            lines.append(f"- error: `{ev.get('error')}`")
        lines.append("")

        if ev.get("user_request"):
            lines.append("**user_request**")
            lines.append("")
            lines.append(_md_code("text", str(ev.get("user_request"))))
            lines.append("")

        if ev.get("requirements"):
            lines.append("**requirements**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(ev.get("requirements"), ensure_ascii=False, indent=2)))
            lines.append("")

        if ev.get("context"):
            lines.append("**context (sampling/summary)**")
            lines.append("")
            lines.append(_md_code("text", str(ev.get("context"))))
            lines.append("")

        if ev.get("generation"):
            gen = ev.get("generation") or {}
            if isinstance(gen, dict):
                imports = str(gen.get("imports") or "")
                code = str(gen.get("code") or "")
                lines.append("**generated imports**")
                lines.append("")
                lines.append(_md_code("python", imports))
                lines.append("")
                lines.append("**generated code**")
                lines.append("")
                lines.append(_md_code("python", code))
                lines.append("")
            else:
                lines.append("**generation**")
                lines.append("")
                lines.append(_md_code("text", str(gen)))
                lines.append("")

        if ev.get("tool_calls") is not None:
            lines.append("**tool_calls**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(ev.get("tool_calls"), ensure_ascii=False, indent=2, default=str)))
            lines.append("")

        if ev.get("assistant_message"):
            lines.append("**assistant_message**")
            lines.append("")
            lines.append(_md_code("text", str(ev.get("assistant_message"))))
            lines.append("")

        if ev.get("error_detail"):
            lines.append("**error_detail**")
            lines.append("")
            lines.append(_md_code("text", str(ev.get("error_detail"))))
            lines.append("")

        if ev.get("execution_stdout"):
            lines.append("**execution_stdout**")
            lines.append("")
            lines.append(_md_code("text", str(ev.get("execution_stdout"))))
            lines.append("")

        if ev.get("validation_report") is not None:
            lines.append("**validation_report**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(ev.get("validation_report"), ensure_ascii=False, indent=2, default=str)))
            lines.append("")

        decision_basis = _derive_decision_basis(ev)
        report = ev.get("validation_report")
        if isinstance(report, dict):
            decision_basis.extend(_build_decision_basis_report(report))
            if prev_validation_report is not None:
                prev_ok = prev_validation_report.get("ok")
                cur_ok = report.get("ok")
                if prev_ok != cur_ok:
                    decision_basis.append(f"validation_ok_changed: {prev_ok} -> {cur_ok}")
            prev_validation_report = report
        if decision_basis:
            lines.append("**decision_basis**")
            lines.append("")
            lines.append(_md_code("text", "\n".join(f"- {x}" for x in decision_basis)))
            lines.append("")

        if ev.get("diff_summary") is not None:
            lines.append("**reflect_diff_summary**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(ev.get("diff_summary"), ensure_ascii=False, indent=2, default=str)))
            lines.append("")

        if ev.get("diff"):
            lines.append("**reflect_diff**")
            lines.append("")
            lines.append(_md_code("diff", _truncate(ev.get("diff"))))
            lines.append("")

        if ev.get("output_files"):
            lines.append("**output_files**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(ev.get("output_files"), ensure_ascii=False, indent=2)))
            lines.append("")

        known = {
            "ts",
            "node",
            "phase",
            "iterations",
            "user_request",
            "requirements",
            "context",
            "generation",
            "tool_calls",
            "assistant_message",
            "error",
            "error_detail",
            "execution_stdout",
            "validation_report",
            "output_files",
            "raw_error",
            "last_message",
            "output_formats",
            "execution_stdout_tail",
            "error_input",
            "prev_generation",
            "diff_summary",
            "diff",
        }
        extras = {k: v for k, v in ev.items() if k not in known}
        if extras:
            lines.append("**extras**")
            lines.append("")
            lines.append(_md_code("json", json.dumps(extras, ensure_ascii=False, indent=2, default=str)))
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return filename


__all__ = [
    "append_trace",
    "cleanup_dir",
    "detect_sample_fallback",
    "diff_generation",
    "extract_input_path",
    "extract_last_message_text",
    "extract_user_request",
    "now_iso",
    "outputs_root_dir",
    "prepare_execution_workdir",
    "promote_staged_outputs",
    "run_generated_code",
    "safe_format_json_like",
    "safe_last_message",
    "truncate_text",
    "write_internal_trace_markdown",
]
