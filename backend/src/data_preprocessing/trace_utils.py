from __future__ import annotations

import json
import difflib
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _outputs_root_dir() -> Path:
    # backend/src/data_preprocessing/trace_utils.py 기준 parents[2] == backend/
    return Path(__file__).resolve().parents[2] / "outputs"


def _generation_to_text(gen: object) -> str:
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
    prev_text = _generation_to_text(prev)
    next_text = _generation_to_text(nxt)
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


def write_internal_trace_markdown(state: dict[str, Any]) -> str | None:
    """누적된 trace를 Markdown 파일로 backend/outputs에 저장하고 파일명을 반환."""
    trace = state.get("trace") or []
    if not isinstance(trace, list) or not trace:
        return None

    run_id = str(state.get("run_id") or "").strip()
    if not run_id:
        return None

    out_dir = _outputs_root_dir()
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
    lines.append(f"- created_at: `{_now_iso()}`")
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


__all__ = ["diff_generation", "write_internal_trace_markdown"]
