from __future__ import annotations

import contextlib
import io
import json
import os
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .models import CodeBlocks, State
from .prompts import code_gen_prompt, reflect_prompt
from .tools import inspect_input, list_images_to_csv, sample_table, summarize_table, load_and_sample
from .common_utils import (
    append_trace,
    cleanup_dir,
    detect_sample_fallback,
    extract_last_message_text,
    extract_requirements_from_user_request,
    extract_user_request,
    now_iso,
    outputs_root_dir,
    promote_staged_outputs,
    safe_format_json_like,
    safe_last_message,
    diff_generation,
    write_internal_trace_markdown,
)


def add_requirements(state: State):
    """사용자 요청에서 요구사항을 뽑아 state에 저장."""
    user_request = state.get("user_request", "")
    reqs = extract_requirements_from_user_request(user_request)
    return {
        "requirements": reqs,
        "phase": "analyzing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "add_requirements",
                "phase": "analyzing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "requirements": [r.model_dump() for r in reqs],
            },
        ),
    }

# ========================== 노드 함수: 실제 작업을 수행하는 노드 정의 ==================================
# chatbot: 첫 LLM 호출 + 툴 호출 유도, 사용자 요청 저장
def chatbot(state: State, llm_with_tools: ChatOpenAI):
    """첫 번째 LLM 호출 및 user_request 유지."""
    system = (
        "You can call tools. If the user message contains a local filesystem path, you MUST call "
        "`inspect_input` with that path first. "
        "If it is an image folder, use `list_images_to_csv`. "
        "Otherwise, you can call `sample_table` and then `summarize_table`."
    )
    response = llm_with_tools.invoke([("system", system), *state["messages"]])
    user_req = state.get("user_request") or extract_user_request(state.get("messages", []))
    return {
        "messages": [response],
        "user_request": user_req,
        "phase": "analyzing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "chatbot",
                "phase": "analyzing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_req,
                "tool_calls": getattr(response, "tool_calls", None),
                "assistant_message": getattr(response, "content", None),
            },
        ),
    }

# add_context: LLM tool_call 중 실제 실행할 대상을 선택
def add_context(state: State):
    """LLM이 반환한 tool_calls 중 첫 번째 유효 호출을 선택해 저장."""
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")

    tool_calls = getattr(message, "tool_calls", []) or []
    selected: Optional[dict[str, Any]] = None
    for tc in tool_calls:
        name = tc.get("name")
        if name in {"inspect_input", "sample_table", "summarize_table", "list_images_to_csv", "load_and_sample"}:
            selected = tc
            break

    if not selected:
        raise ValueError("No supported tool call found in LLM response.")

    tool_name = str(selected.get("name") or "")
    tool_args = selected.get("args") or {}
    user_request = state.get("user_request", "")
    return {
        "tool_call_name": tool_name,
        "tool_call_args": tool_args,
        "context_candidate": None,
        "inspect_result": None,
        "sample_json": None,
        "summary_context": None,
        "user_request": user_request,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "add_context",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "tool_calls": tool_calls,
                "tool_call_selected": {"name": tool_name, "args": tool_args},
            },
        ),
    }


# run_inspect: 입력 경로 검사(파일/폴더/이미지 여부 판단)
def run_inspect(state: State):
    """inspect_input을 실행해 입력 상태를 구조화."""
    args = state.get("tool_call_args") or {}
    user_request = state.get("user_request", "")
    context_candidate: Optional[str] = None
    inspect_info: Optional[dict[str, Any]] = None
    try:
        inspect_str = inspect_input.invoke(args)
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

    return {
        "inspect_result": inspect_info,
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_inspect",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
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
        tool_name = state.get("tool_call_name")
        args = state.get("tool_call_args") or {}
        if tool_name != "list_images_to_csv":
            inspect_info = state.get("inspect_result") or {}
            dir_path = inspect_info.get("input_path") or ""
            args = {"dir_path": dir_path}
        context_candidate = list_images_to_csv.invoke(args)
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


# run_sample: 테이블 파일 샘플링 수행
def run_sample(state: State):
    """sample_table을 실행해 샘플 JSON 생성."""
    user_request = state.get("user_request", "")
    tool_name = state.get("tool_call_name")
    tool_args = state.get("tool_call_args") or {}
    context_candidate: Optional[str] = None
    sample_json: Optional[str] = None
    try:
        if tool_name == "sample_table":
            args = tool_args
        else:
            inspect_info = state.get("inspect_result") or {}
            target = inspect_info.get("candidate_file") or inspect_info.get("input_path") or ""
            sample_size = tool_args.get("sample_size", 5000)
            args = {"path": target, "sample_size": sample_size}
        sample_json = sample_table.invoke(args)
        if isinstance(sample_json, str) and sample_json.startswith("ERROR_CONTEXT||"):
            context_candidate = sample_json
    except Exception as exc:  # noqa: BLE001
        context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    return {
        "sample_json": sample_json,
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_sample",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context_candidate": context_candidate,
            },
        ),
    }


# run_summarize: 샘플 JSON 요약 수행
def run_summarize(state: State):
    """summarize_table을 실행해 컨텍스트 요약 생성."""
    user_request = state.get("user_request", "")
    tool_name = state.get("tool_call_name")
    tool_args = state.get("tool_call_args") or {}
    context_candidate: Optional[str] = None
    summary_context: Optional[str] = None
    try:
        if tool_name == "summarize_table":
            sample_json = tool_args.get("sample_json", "")
        else:
            sample_json = state.get("sample_json", "")
        if not sample_json:
            context_candidate = "ERROR_CONTEXT||MissingSample||sample_json이 비어 있습니다."
        else:
            summary_context = summarize_table.invoke({"sample_json": sample_json})
            context_candidate = summary_context
    except Exception as exc:  # noqa: BLE001
        context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    return {
        "summary_context": summary_context,
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_summarize",
                "phase": "sampling",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context_candidate": context_candidate,
            },
        ),
    }


# run_load_and_sample: 통합 샘플링(호환성 유지)
def run_load_and_sample(state: State):
    """load_and_sample을 실행해 컨텍스트를 직접 생성."""
    user_request = state.get("user_request", "")
    tool_args = state.get("tool_call_args") or {}
    context_candidate: Optional[str] = None
    try:
        context_candidate = load_and_sample.invoke(tool_args)
    except Exception as exc:  # noqa: BLE001
        context_candidate = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"

    return {
        "context_candidate": context_candidate,
        "phase": "sampling",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "run_load_and_sample",
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

# friendly_error: 로딩 실패 등 오류 컨텍스트를 비개발자용 한글 메시지로 변환
def friendly_error(state: State, llm_gpt: ChatOpenAI):
    """원문 에러 메시지를 비개발자용 한글 요약으로 변환."""
    context = state.get("context", "")
    user_request = state.get("user_request", "")

    parts = context.split("||", 2)  # ERROR_CONTEXT||<exc_name>||<exc_message>
    exc_name = parts[1] if len(parts) > 1 else "UnknownError"
    exc_msg = parts[2] if len(parts) > 2 else context

    prompt = (
        "다음 에러를 비개발자가 이해할 수 있게 한글로 짧게 설명하고, 해결 방법을 한 줄로 제안하세요.\n"
        f"요청: {user_request}\n"
        f"에러명: {exc_name}\n"
        f"에러내용: {exc_msg}\n"
        "출력 형식: 원인: ...\n대처: ..."
    )

    resp = llm_gpt.invoke(prompt)
    return {
        "final_user_messages": [("assistant", resp.content)],
        "error": "yes",
        "phase": "finalizing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "friendly_error",
                "phase": "finalizing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "context": context,
                "assistant_message": resp.content,
            },
        ),
    }


def final_friendly_error(state: State, llm_gpt: ChatOpenAI):
    """최종 실패 시점의 에러를 비개발자용 한글 요약으로 변환."""

    user_request = state.get("user_request", "")
    raw_error = extract_last_message_text(state.get("messages", []))
    # 프롬프트가 너무 커지는 것을 방지(스택 트레이스는 매우 길 수 있음)
    raw_error = raw_error[-4000:] if isinstance(raw_error, str) else str(raw_error)

    prompt = (
        "다음 전처리 실행이 최종적으로 실패했습니다. 비개발자가 이해할 수 있게 한글로 짧게 설명하고, "
        "사용자가 바로 시도할 수 있는 해결 방법을 1~3개 제안하세요.\n"
        f"요청: {user_request}\n"
        f"에러내용(일부):\n{raw_error}\n"
        "출력 형식:\n원인: ...\n대처: ...\n"
    )
    resp = llm_gpt.invoke(prompt)
    return {
        "final_user_messages": [("assistant", resp.content)],
        "error": "yes",
        "phase": "finalizing",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "final_friendly_error",
                "phase": "finalizing",
                "iterations": int(state.get("iterations", 0) or 0),
                "user_request": user_request,
                "raw_error": raw_error,
                "assistant_message": resp.content,
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
    requirements_text = "\n".join([f"- {r.id}: {r.text}" for r in reqs]) if reqs else "(none)"

    generated_code = llm_coder.invoke(
        code_gen_prompt.format_messages(
            context=context,
            user_request=user_request,
            output_formats=output_formats,
            requirements=requirements_text,
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
        error_message = [("user", error_detail)]
        return {
            "generation": code_solution,
            "messages": error_message,
            "error": "yes",
            "phase": "executing",
            "execution_stdout": "",
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "code_check",
                    "phase": "executing",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "error_detail": error_detail,
                    "execution_stdout": "",
                    "generation": {"imports": imports, "code": code},
                },
            ),
        }

    stdout_buffer = io.StringIO()
    exec_globals: Dict[str, Any] = {}
    prev_cwd = os.getcwd()
    staging_root = outputs_root_dir() / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    execution_workdir = staging_root / f"run_{uuid4().hex}"
    execution_workdir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(str(execution_workdir))
        with contextlib.redirect_stdout(stdout_buffer):
            exec(imports + "\n" + code, exec_globals)
    except SystemExit as exc:
        os.chdir(prev_cwd)
        cleanup_dir(execution_workdir)
        captured = stdout_buffer.getvalue()
        error_detail = captured + "\n" + f"SystemExit: {getattr(exc, 'code', exc)}\n" + traceback.format_exc()
        error_message = [("user", f"Your solution called exit() during execution (this is not allowed): {error_detail}")]
        return {
            "generation": code_solution,
            "messages": error_message,
            "error": "yes",
            "phase": "executing",
            "execution_stdout": captured,
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "code_check",
                    "phase": "executing",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "error_detail": error_detail,
                    "execution_stdout": captured,
                    "generation": {"imports": imports, "code": code},
                },
            ),
        }
    except BaseException as exc:  # noqa: BLE001
        # 호스트에서 인터럽트가 오면 서버가 정상 종료되도록 그대로 전파
        if isinstance(exc, KeyboardInterrupt):
            raise
        os.chdir(prev_cwd)
        cleanup_dir(execution_workdir)
        captured = stdout_buffer.getvalue()
        error_detail = captured + "\n" + traceback.format_exc()
        error_message = [("user", f"Your solution failed during execution: {error_detail}")]
        return {
            "generation": code_solution,
            "messages": error_message,
            "error": "yes",
            "phase": "executing",
            "execution_stdout": captured,
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "code_check",
                    "phase": "executing",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "error_detail": error_detail,
                    "execution_stdout": captured,
                    "generation": {"imports": imports, "code": code},
                },
            ),
        }
    finally:
        # 이후 stdout 추출이 실패하더라도, 작업 디렉터리는 최대한 원복
        try:
            os.chdir(prev_cwd)
        except Exception:  # noqa: BLE001
            pass

    output_preview = stdout_buffer.getvalue()
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
        "execution_workdir": str(execution_workdir),
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
        if execution_workdir:
            cleanup_dir(execution_workdir)
        stdout_tail = (state.get("execution_stdout") or "")[-2000:]
        error_message = [
            (
                "user",
                "Validation failed: missing __validation_report__.\n"
                "Your script MUST set a JSON-serializable dict named __validation_report__ with at least {ok: bool, issues: [...] }.\n"
                f"Execution stdout (tail):\n{stdout_tail}",
            )
        ]
        return {
            "messages": error_message,
            "error": "yes",
            "phase": "validating",
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "validate",
                    "phase": "validating",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "validation_report": report,
                    "execution_stdout_tail": stdout_tail,
                    "last_message": safe_last_message(state.get("messages", [])),
                },
            ),
        }

    if "ok" not in report or "issues" not in report:
        if execution_workdir:
            cleanup_dir(execution_workdir)
        error_message = [
            (
                "user",
                "Validation failed: __validation_report__ must include keys 'ok' and 'issues'.\n"
                f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}",
            )
        ]
        return {
            "messages": error_message,
            "error": "yes",
            "phase": "validating",
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "validate",
                    "phase": "validating",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "validation_report": report,
                    "last_message": safe_last_message(state.get("messages", [])),
                },
            ),
        }

    ok = report.get("ok")

    # 방어적 처리: metrics가 결측/placeholder/fallback 사용을 시사하면 ok=True를 그대로 신뢰하지 않는다.
    metrics = report.get("metrics")
    if ok is True and not isinstance(metrics, dict):
        if execution_workdir:
            cleanup_dir(execution_workdir)
        error_message = [
            (
                "user",
                "Validation failed: __validation_report__ must include a dict 'metrics' to justify ok=True.\n"
                f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}",
            )
        ]
        return {
            "messages": error_message,
            "error": "yes",
            "phase": "validating",
            **append_trace(
                state,
                {
                    "ts": now_iso(),
                    "node": "validate",
                    "phase": "validating",
                    "iterations": int(state.get("iterations", 0) or 0),
                    "error": "yes",
                    "validation_report": report,
                    "last_message": safe_last_message(state.get("messages", [])),
                },
            ),
        }

    # 사용자 요구사항 강제: 요구사항 누락/실패 시 리팩트(reflect) 루프로 보낸다.
    if reqs:
        report_reqs = report.get("requirements")
        if report_reqs is None and isinstance(metrics, dict):
            report_reqs = metrics.get("requirements")

        if not isinstance(report_reqs, dict):
            if execution_workdir:
                cleanup_dir(execution_workdir)
            req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
            msg = (
                "Validation failed: missing __validation_report__['requirements'].\n"
                "Your script MUST report requirement-level pass/fail in __validation_report__['requirements'].\n"
                f"Required:\n{req_list}\n"
            )
            return {
                "messages": [("user", msg)],
                "error": "yes",
                "phase": "validating",
                **append_trace(
                    state,
                    {
                        "ts": now_iso(),
                        "node": "validate",
                        "phase": "validating",
                        "iterations": int(state.get("iterations", 0) or 0),
                        "error": "yes",
                        "validation_report": report,
                        "last_message": msg,
                    },
                ),
            }

        missing_ids: list[str] = []
        failed_ids: list[str] = []
        for r in reqs:
            if r.id not in report_reqs:
                missing_ids.append(r.id)
                continue
            v = report_reqs.get(r.id)
            if isinstance(v, bool):
                passed = v
            elif isinstance(v, dict):
                passed = bool(v.get("ok"))
            else:
                passed = False
            if not passed:
                failed_ids.append(r.id)

        if missing_ids or failed_ids:
            if execution_workdir:
                cleanup_dir(execution_workdir)
            req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
            msg = (
                "Validation failed: user requirements not satisfied.\n"
                f"Missing requirement ids: {missing_ids}\n"
                f"Failed requirement ids: {failed_ids}\n"
                "All requirements (must satisfy):\n"
                f"{req_list}\n"
                f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=4000)}"
            )
            return {
                "messages": [("user", msg)],
                "error": "yes",
                "phase": "validating",
                **append_trace(
                    state,
                    {
                        "ts": now_iso(),
                        "node": "validate",
                        "phase": "validating",
                        "iterations": int(state.get("iterations", 0) or 0),
                        "error": "yes",
                        "validation_report": report,
                        "last_message": msg,
                    },
                ),
            }

    # metrics에 "<col>_missing"/"<col>_empty"가 있으면, 대응하는 placeholder/fallback 메트릭도 요구한다.
    # "정보 없음" 같은 placeholder로 미매핑 값을 숨기는 것을 방지한다.
    if isinstance(metrics, dict):
        prefixes: set[str] = set()
        for k in metrics.keys():
            ks = str(k)
            if ks.endswith("_missing"):
                prefixes.add(ks[: -len("_missing")])
            elif ks.endswith("_empty"):
                prefixes.add(ks[: -len("_empty")])

        missing_placeholder_metrics: list[str] = []
        for p in prefixes:
            has_placeholder = any(
                str(k).startswith(p) and any(tok in str(k).lower() for tok in ("placeholder", "fallback"))
                for k in metrics.keys()
            )
            if not has_placeholder:
                missing_placeholder_metrics.append(p)

        if ok is True and missing_placeholder_metrics:
            if execution_workdir:
                cleanup_dir(execution_workdir)
            msg = (
                "Validation failed: ok=True but placeholder/fallback coverage metrics are missing.\n"
                "For each filled/added column, include '<col>_placeholder' (or '<col>_fallback') and set ok=False when it > 0.\n"
                f"Columns requiring placeholder metrics: {missing_placeholder_metrics}\n"
                f"__validation_report__ (truncated):\n{safe_format_json_like(report, limit=3000)}"
            )
            return {
                "messages": [("user", msg)],
                "error": "yes",
                "phase": "validating",
                **append_trace(
                    state,
                    {
                        "ts": now_iso(),
                        "node": "validate",
                        "phase": "validating",
                        "iterations": int(state.get("iterations", 0) or 0),
                        "error": "yes",
                        "validation_report": report,
                        "last_message": msg,
                    },
                ),
            }

    metric_flags: list[str] = []
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if type(v) is bool:
                continue
            if isinstance(v, (int, float)):
                key = str(k).lower()
                if any(tok in key for tok in ("missing", "empty", "placeholder", "fallback")) and v > 0:
                    metric_flags.append(f"{k}={v}")

    if ok is True and not metric_flags:
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

    if metric_flags:
        issues_list = issues_list + [f"metrics indicate potential silent miss: {x}" for x in metric_flags]

    report_json = safe_format_json_like(report, limit=6000)
    issue_text = "\n".join(f"- {x}" for x in issues_list) if issues_list else "(no issues list provided)"
    error_message = [
        (
            "user",
            "Validation failed (silent-miss guardrail).\n"
            f"Issues:\n{issue_text}\n\n"
            f"__validation_report__ (truncated):\n{report_json}",
        )
    ]
    if execution_workdir:
        cleanup_dir(execution_workdir)
    return {
        "messages": error_message,
        "error": "yes",
        "phase": "validating",
        **append_trace(
            state,
            {
                "ts": now_iso(),
                "node": "validate",
                "phase": "validating",
                "iterations": int(state.get("iterations", 0) or 0),
                "error": "yes",
                "validation_report": report,
                "last_message": safe_last_message(state.get("messages", [])),
            },
        ),
    }


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


# ===================================== 노드 함수 끝 =============================

# ======================================= 라우터: 조건 분기 함수 영역 ======================
# route_tool_call: 선택된 tool_call에 따라 실행 노드 분기
def route_tool_call(state: State):
    tool_name = state.get("tool_call_name") or ""
    if tool_name == "inspect_input":
        return "run_inspect"
    if tool_name == "sample_table":
        return "run_sample"
    if tool_name == "summarize_table":
        return "run_summarize"
    if tool_name == "list_images_to_csv":
        return "run_image_manifest"
    if tool_name == "load_and_sample":
        return "run_load_and_sample"
    return END

# route_after_inspect: 검사 결과에 따라 이미지/샘플/에러 분기
def route_after_inspect(state: State):
    ctx = state.get("context_candidate", "")
    if isinstance(ctx, str) and ctx.startswith("ERROR_CONTEXT||"):
        return "build_context"
    info = state.get("inspect_result") or {}
    if isinstance(info, dict) and info.get("has_images"):
        return "run_image_manifest"
    return "run_sample"

# route_after_sample: 샘플링 실패 여부에 따라 요약/에러 분기
def route_after_sample(state: State):
    ctx = state.get("context_candidate", "")
    if isinstance(ctx, str) and ctx.startswith("ERROR_CONTEXT||"):
        return "build_context"
    return "run_summarize"

# route_after_context: 최종 컨텍스트에 오류 마커가 있으면 친절한 에러 노드로 분기
def route_after_context(state: State):
    """컨텍스트에 오류 마커가 있으면 친절한 에러 노드로 분기."""
    ctx = state.get("context", "")
    if isinstance(ctx, str) and ctx.startswith("ERROR_CONTEXT||"):
        return "friendly_error"
    return "generate"

# guardrail_route: chatbot 단계에서 툴 호출 여부로 분기하는 간단한 라우터
def guardrail_route(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "add_context"
    return END

def decide_to_finish(state: State, max_iterations: int):
    """실패 시 reflect/최종에러 분기 결정."""
    error = state["error"]
    iterations = state["iterations"]
    if error == "no":
        return "end"
    if iterations == max_iterations:
        return "final_friendly_error"
    return "reflect"


def route_after_execution(state: State, max_iterations: int):
    """code_check 이후: 실행 성공이면 validate로, 실패면 reflect/최종에러로."""
    if state.get("error") == "no":
        return "validate"
    return decide_to_finish(state, max_iterations=max_iterations)
# ============================================== 라우터 끝 ==============================

# ================================ 그래프 빌드: 노드/엣지 추가 블록 ===========================
def build_graph(
    llm_model: str = "gpt-4o-mini",
    coder_model: str = "gpt-4.1",
    max_iterations: int = 5,
):
    """LangGraph를 구성하고 컴파일된 그래프를 반환."""
    llm_gpt = ChatOpenAI(model=llm_model)
    llm_coder = ChatOpenAI(model=coder_model, temperature=0)
    llm_with_tools = llm_gpt.bind_tools(
        tools=[inspect_input, sample_table, summarize_table, list_images_to_csv]
    )

    graph_builder = StateGraph(State)

    # ===== 노드: 그래프에 노드 추가 =====
    # 요구사항 추출
    graph_builder.add_node("add_requirements", add_requirements)
    # 요청 해석 + tool call 결정
    graph_builder.add_node("chatbot", partial(chatbot, llm_with_tools=llm_with_tools))
    # tool call 선택/기록
    graph_builder.add_node("add_context", add_context)
    # 입력 검사/샘플링/요약 단계 분리
    graph_builder.add_node("run_inspect", run_inspect)
    graph_builder.add_node("run_image_manifest", run_image_manifest)
    graph_builder.add_node("run_sample", run_sample)
    graph_builder.add_node("run_summarize", run_summarize)
    graph_builder.add_node("run_load_and_sample", run_load_and_sample)
    graph_builder.add_node("build_context", build_context)
    # 에러/코드 생성/실행/검증/리플렉트
    graph_builder.add_node("friendly_error", partial(friendly_error, llm_gpt=llm_gpt))
    graph_builder.add_node("final_friendly_error", partial(final_friendly_error, llm_gpt=llm_gpt))
    graph_builder.add_node("generate", partial(generate, llm_coder=llm_coder, llm_gpt=llm_gpt))
    graph_builder.add_node("code_check", code_check)
    graph_builder.add_node("validate", validate)
    graph_builder.add_node("reflect", partial(reflect, llm_coder=llm_coder, llm_gpt=llm_gpt))
    # ===== 노드 끝 =====


    # ===== 엣지: 노드 연결 정의 =====
    graph_builder.add_edge(START, "add_requirements")
    graph_builder.add_edge("add_requirements", "chatbot")

    # chatbot 이후: 툴 호출이 있으면 add_context, 없으면 END
    graph_builder.add_conditional_edges(
        "chatbot",
        guardrail_route,
        {"add_context": "add_context", END: END},
    )

    # add_context 이후: tool_call 종류에 따라 분기
    graph_builder.add_conditional_edges(
        "add_context",
        route_tool_call,
        {
            "run_inspect": "run_inspect",
            "run_sample": "run_sample",
            "run_summarize": "run_summarize",
            "run_image_manifest": "run_image_manifest",
            "run_load_and_sample": "run_load_and_sample",
            END: END,
        },
    )

    # inspect 이후: 이미지/샘플/에러 분기
    graph_builder.add_conditional_edges(
        "run_inspect",
        route_after_inspect,
        {
            "run_image_manifest": "run_image_manifest",
            "run_sample": "run_sample",
            "build_context": "build_context",
        },
    )

    # sample 이후: 요약/에러 분기
    graph_builder.add_conditional_edges(
        "run_sample",
        route_after_sample,
        {"run_summarize": "run_summarize", "build_context": "build_context"},
    )

    # summarize/image/load 결과는 build_context로 모아서 통일
    graph_builder.add_edge("run_summarize", "build_context")
    graph_builder.add_edge("run_image_manifest", "build_context")
    graph_builder.add_edge("run_load_and_sample", "build_context")

    # build_context 이후: 오류면 friendly_error, 정상이면 generate
    graph_builder.add_conditional_edges(
        "build_context",
        route_after_context,
        {"friendly_error": "friendly_error", "generate": "generate"},
    )

    # generate → code_check
    graph_builder.add_edge("generate", "code_check")

    # code_check 이후: 성공 → validate, 실패 → reflect/final
    graph_builder.add_conditional_edges(
        "code_check",
        partial(route_after_execution, max_iterations=max_iterations),
        {"validate": "validate", "reflect": "reflect", "final_friendly_error": "final_friendly_error"},
    )

    # validate 이후: 통과면 END, 실패면 reflect/final
    graph_builder.add_conditional_edges(
        "validate",
        partial(decide_to_finish, max_iterations=max_iterations),
        {"end": END, "reflect": "reflect", "final_friendly_error": "final_friendly_error"},
    )

    # reflect → code_check
    graph_builder.add_edge("reflect", "code_check")

    # friendly_error → END
    graph_builder.add_edge("friendly_error", END)
    # final_friendly_error → END
    graph_builder.add_edge("final_friendly_error", END)
    # ===== 엣지 끝 =====

    return graph_builder.compile()
# ===================================== 그래프 빌드 끝 =================================

# ====================== 엔트리포인트: 그래프 실행 함수 =======================================
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
# ===== 엔트리포인트 끝 =====


__all__ = ["build_graph", "run_request"]
