from __future__ import annotations

# =========================
# 표준 라이브러리
# =========================
import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

# =========================
# 외부 라이브러리
# =========================
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

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
    diff_generation,
    write_internal_trace_markdown,
)

# =========================
# 내부 모듈: 워크플로우 유틸
# =========================
from .node_utils import (
    inspect_input,
    list_images_to_csv,
    sample_table,
    summarize_table,
    _toolcall_apply_defaults,
    _toolcall_build_dedup_key,
    _toolcall_format_reports,
    _toolcall_parse_entry,
    _exec_run_generated_code,
    _code_check_error_response,
    _build_friendly_error_prompt,
    _friendly_error_response,
    _validation_coerce_bool,
    _validation_collect_metric_flags,
    _validation_eval_requirements,
    _validation_extract_missing_mapping,
    _validation_extract_policy,
    _validation_extract_requirements,
    _validation_fail_missing_mapping,
    _validation_fail_missing_metrics,
    _validation_fail_missing_ok_issues,
    _validation_fail_missing_placeholder_metrics,
    _validation_fail_missing_report,
    _validation_fail_missing_requirements,
    _validation_fail_requirements,
    _validation_fail_silent_miss,
    _validation_missing_placeholder_metrics,
    _validation_success_response,
)


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
