from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shutil
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from .models import CodeBlocks, Requirement, State
from .prompts import code_gen_prompt, reflect_prompt
from .tools import list_images_to_csv, load_and_sample

# ==================================== 유틸: 공용 유틸 함수 영역 =====================
def _extract_user_request(messages: list[Any]) -> str:
    if not messages:
        return ""
    first = messages[0]
    if isinstance(first, tuple):
        return first[1]
    if hasattr(first, "content"):
        return getattr(first, "content", "")
    return ""


def _extract_last_message_text(messages: list[Any]) -> str:
    if not messages:
        return ""
    last = messages[-1]
    if isinstance(last, tuple) and len(last) >= 2:
        return str(last[1])
    if hasattr(last, "content"):
        return str(getattr(last, "content", ""))
    return str(last)


def _truncate_text(s: str, limit: int = 4000) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + "\n…(truncated)"


def _safe_format_json_like(obj: object, limit: int = 6000) -> str:
    """디버그 메시지용 안전한 포맷팅(최대한 시도하며, 예외를 발생시키지 않음)."""
    try:
        return _truncate_text(json.dumps(obj, ensure_ascii=False, indent=2), limit=limit)
    except Exception as exc:  # noqa: BLE001
        return _truncate_text(f"<non-json-serializable: {type(exc).__name__}: {exc}>\n{repr(obj)}", limit=limit)


def _outputs_root_dir() -> Path:
    # backend/src/data_preprocessing/workflow.py 기준 parents[2] == backend/
    return Path(__file__).resolve().parents[2] / "outputs"


def _cleanup_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def _promote_staged_outputs(execution_workdir: Path) -> list[str]:
    """execution_workdir 아래에서 생성된 ./outputs를 backend/outputs로 옮기고, 옮긴 파일명을 반환."""
    src = execution_workdir / "outputs"
    if not src.exists():
        return []

    dst_root = _outputs_root_dir()
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

def _strip_paths_and_quotes(text: str) -> str:
    # 실행 가능한 요구사항이 아닌, 명백한 파일 경로/주소(URI) 토큰을 제거
    s = text
    s = re.sub(r"s3://\S+", " ", s)
    # 주의: 과도하게 제거하지 않도록 보수적으로 처리(경로처럼 보이는 토큰만 제거)
    s = re.sub(r"(?:[A-Za-z]:\\|/)\S+", " ", s)
    s = s.replace('"', " ").replace("'", " ")
    return s


def _extract_requirements_from_user_request(user_request: str, max_items: int = 10) -> list[Requirement]:
    """요구사항 누락(silent miss)을 줄이기 위한, 규칙 기반(결정적) 요구사항 추출(best-effort).

    토큰 사용을 줄이기 위해 의도적으로 단순하게 유지(추가 LLM 호출 없음).
    """
    text = (user_request or "").strip()
    if not text:
        return []

    text = _strip_paths_and_quotes(text)
    # 흔한 구분자(한글 접속사 + 구두점)로 분리
    parts = re.split(r"[\n\r]+|[.;!?]+|,|\b그리고\b|\b및\b|\b또는\b|\b또\b|\b또한\b|\b추가로\b", text)
    cleaned: list[str] = []
    for p in parts:
        t = re.sub(r"\s+", " ", p).strip()
        if len(t) < 2:
            continue
        cleaned.append(t)

    # 순서를 유지하면서 중복 제거
    uniq: list[str] = []
    seen: set[str] = set()
    for t in cleaned:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
        if len(uniq) >= max_items:
            break

    return [Requirement(id=f"REQ-{i+1}", text=t, severity="must") for i, t in enumerate(uniq)]


def add_requirements(state: State):
    """사용자 요청에서 요구사항을 뽑아 state에 저장."""
    user_request = state.get("user_request", "")
    reqs = _extract_requirements_from_user_request(user_request)
    return {"requirements": reqs, "phase": "analyzing"}
# ============================= 유틸 끝 ==========================================



# ========================== 노드 함수: 실제 작업을 수행하는 노드 정의 ==================================
# chatbot: 첫 LLM 호출 + 툴 호출 유도, 사용자 요청 저장
def chatbot(state: State, llm_with_tools: ChatOpenAI):
    """첫 번째 LLM 호출 및 user_request 유지."""
    system = (
        "You can call tools. If the user message contains a local filesystem path, you MUST call the tool "
        "`load_and_sample` with that path first to inspect the dataset. "
        "Do NOT call `list_images_to_csv` directly; `load_and_sample` will handle image folders when needed."
    )
    response = llm_with_tools.invoke([("system", system), *state["messages"]])
    user_req = state.get("user_request") or _extract_user_request(state.get("messages", []))
    return {"messages": [response], "user_request": user_req, "phase": "analyzing"}

# add_context: 툴 결과(샘플/매니페스트/에러)를 컨텍스트에 저장
def add_context(state: State):
    """도구 호출 결과에서 컨텍스트를 추출 (파일 샘플 or 이미지 목록)."""
    if messages := state.get("messages", []):
        message = messages[-1]
    else:
        raise ValueError("No message found in input")

    context_str = None
    for tc in getattr(message, "tool_calls", []):
        tool_name = tc.get("name")
        args = tc.get("args", {})
        try:
            if tool_name == "load_and_sample" and args:
                context_str = load_and_sample.invoke(args)
                break
            if tool_name == "list_images_to_csv" and args:
                context_str = list_images_to_csv.invoke(args)
                break
        except Exception as exc:  # noqa: BLE001
            context_str = f"ERROR_CONTEXT||{type(exc).__name__}||{exc}"
            break

    if not context_str:
        raise ValueError("No tool result found; ensure tool call executed or supported format present")

    return {"context": context_str, "user_request": state.get("user_request", ""), "phase": "sampling"}

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
    return {"final_user_messages": [("assistant", resp.content)], "error": "yes", "phase": "finalizing"}


def final_friendly_error(state: State, llm_gpt: ChatOpenAI):
    """최종 실패 시점의 에러를 비개발자용 한글 요약으로 변환."""

    user_request = state.get("user_request", "")
    raw_error = _extract_last_message_text(state.get("messages", []))
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
    return {"final_user_messages": [("assistant", resp.content)], "error": "yes", "phase": "finalizing"}

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

    return {"generation": code_solution, "messages": messages, "phase": "generating"}

# code_check: 생성 코드 실행 및 에러 감지
def code_check(state: State):
    """생성된 코드 실행 및 에러 감지."""
    code_solution = state["generation"]
    imports = code_solution.imports
    code = code_solution.code

    stdout_buffer = io.StringIO()
    exec_globals: Dict[str, Any] = {}
    prev_cwd = os.getcwd()
    staging_root = _outputs_root_dir() / "_staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    execution_workdir = staging_root / f"run_{uuid4().hex}"
    execution_workdir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(str(execution_workdir))
        with contextlib.redirect_stdout(stdout_buffer):
            exec(imports + "\n" + code, exec_globals)
    except SystemExit as exc:
        os.chdir(prev_cwd)
        _cleanup_dir(execution_workdir)
        captured = stdout_buffer.getvalue()
        error_detail = captured + "\n" + f"SystemExit: {getattr(exc, 'code', exc)}\n" + traceback.format_exc()
        error_message = [("user", f"Your solution called exit() during execution (this is not allowed): {error_detail}")]
        return {
            "generation": code_solution,
            "messages": error_message,
            "error": "yes",
            "phase": "executing",
            "execution_stdout": captured,
        }
    except BaseException as exc:  # noqa: BLE001
        # 호스트에서 인터럽트가 오면 서버가 정상 종료되도록 그대로 전파
        if isinstance(exc, KeyboardInterrupt):
            raise
        os.chdir(prev_cwd)
        _cleanup_dir(execution_workdir)
        captured = stdout_buffer.getvalue()
        error_detail = captured + "\n" + traceback.format_exc()
        error_message = [("user", f"Your solution failed during execution: {error_detail}")]
        return {
            "generation": code_solution,
            "messages": error_message,
            "error": "yes",
            "phase": "executing",
            "execution_stdout": captured,
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
            _cleanup_dir(execution_workdir)
        stdout_tail = (state.get("execution_stdout") or "")[-2000:]
        error_message = [
            (
                "user",
                "Validation failed: missing __validation_report__.\n"
                "Your script MUST set a JSON-serializable dict named __validation_report__ with at least {ok: bool, issues: [...] }.\n"
                f"Execution stdout (tail):\n{stdout_tail}",
            )
        ]
        return {"messages": error_message, "error": "yes", "phase": "validating"}

    if "ok" not in report or "issues" not in report:
        if execution_workdir:
            _cleanup_dir(execution_workdir)
        error_message = [
            (
                "user",
                "Validation failed: __validation_report__ must include keys 'ok' and 'issues'.\n"
                f"__validation_report__ (truncated):\n{_safe_format_json_like(report, limit=3000)}",
            )
        ]
        return {"messages": error_message, "error": "yes", "phase": "validating"}

    ok = report.get("ok")

    # 방어적 처리: metrics가 결측/placeholder/fallback 사용을 시사하면 ok=True를 그대로 신뢰하지 않는다.
    metrics = report.get("metrics")
    if ok is True and not isinstance(metrics, dict):
        if execution_workdir:
            _cleanup_dir(execution_workdir)
        error_message = [
            (
                "user",
                "Validation failed: __validation_report__ must include a dict 'metrics' to justify ok=True.\n"
                f"__validation_report__ (truncated):\n{_safe_format_json_like(report, limit=3000)}",
            )
        ]
        return {"messages": error_message, "error": "yes", "phase": "validating"}

    # 사용자 요구사항 강제: 요구사항 누락/실패 시 리팩트(reflect) 루프로 보낸다.
    if reqs:
        report_reqs = report.get("requirements")
        if report_reqs is None and isinstance(metrics, dict):
            report_reqs = metrics.get("requirements")

        if not isinstance(report_reqs, dict):
            if execution_workdir:
                _cleanup_dir(execution_workdir)
            req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
            msg = (
                "Validation failed: missing __validation_report__['requirements'].\n"
                "Your script MUST report requirement-level pass/fail in __validation_report__['requirements'].\n"
                f"Required:\n{req_list}\n"
            )
            return {"messages": [("user", msg)], "error": "yes", "phase": "validating"}

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
                _cleanup_dir(execution_workdir)
            req_list = "\n".join([f"- {r.id}: {r.text}" for r in reqs])
            msg = (
                "Validation failed: user requirements not satisfied.\n"
                f"Missing requirement ids: {missing_ids}\n"
                f"Failed requirement ids: {failed_ids}\n"
                "All requirements (must satisfy):\n"
                f"{req_list}\n"
                f"__validation_report__ (truncated):\n{_safe_format_json_like(report, limit=4000)}"
            )
            return {"messages": [("user", msg)], "error": "yes", "phase": "validating"}

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
                _cleanup_dir(execution_workdir)
            msg = (
                "Validation failed: ok=True but placeholder/fallback coverage metrics are missing.\n"
                "For each filled/added column, include '<col>_placeholder' (or '<col>_fallback') and set ok=False when it > 0.\n"
                f"Columns requiring placeholder metrics: {missing_placeholder_metrics}\n"
                f"__validation_report__ (truncated):\n{_safe_format_json_like(report, limit=3000)}"
            )
            return {"messages": [("user", msg)], "error": "yes", "phase": "validating"}

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
                moved = _promote_staged_outputs(execution_workdir)
            finally:
                _cleanup_dir(execution_workdir)
        if moved:
            return {
                "error": "no",
                "phase": "validating",
                "output_files": moved,
                "messages": [("user", f"Outputs promoted: {moved}")],
            }
        return {"error": "no", "phase": "validating", "output_files": []}

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

    report_json = _safe_format_json_like(report, limit=6000)
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
        _cleanup_dir(execution_workdir)
    return {"messages": error_message, "error": "yes", "phase": "validating"}


# reflect: 실행 오류를 기반으로 수정 코드 재생성
def reflect(state: State, llm_coder: ChatOpenAI, llm_gpt: ChatOpenAI):
    """에러 발생 시 수정 코드 생성."""
    error = _extract_last_message_text(state.get("messages", []))
    code_solution = state["generation"]
    code_solution_str = (
        f"Imports: {code_solution.imports} \n Code: {code_solution.code}"
    )

    corrected_code = llm_coder.invoke(
        reflect_prompt.format_messages(error=error, code_solution=code_solution_str)
    )
    code_structurer = llm_gpt.with_structured_output(CodeBlocks)
    reflections = code_structurer.invoke(corrected_code.content)

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
    }
# ===================================== 노드 함수 끝 =============================

# ======================================= 라우터: 조건 분기 함수 영역 ======================
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
    llm_with_tools = llm_gpt.bind_tools(tools=[load_and_sample, list_images_to_csv])

    graph_builder = StateGraph(State)

    # ===== 노드: 그래프에 노드 추가 =====
    graph_builder.add_node("add_requirements", add_requirements)                                    # user_request → requirements
    graph_builder.add_node("chatbot", partial(chatbot, llm_with_tools=llm_with_tools))              # 첫 LLM 호출 + tool call 유도
    graph_builder.add_node("add_context", add_context)                                              # tool 결과 → context 저장
    graph_builder.add_node("friendly_error", partial(friendly_error, llm_gpt=llm_gpt))              # 에러 메시지 친절화
    graph_builder.add_node("final_friendly_error", partial(final_friendly_error, llm_gpt=llm_gpt))  # 최종 실패 친절화
    graph_builder.add_node("generate", partial(generate, llm_coder=llm_coder, llm_gpt=llm_gpt))     # 코드 초안 생성
    graph_builder.add_node("code_check", code_check)                                                # 생성 코드 실행/검증
    graph_builder.add_node("validate", validate)                                                    # 실행 결과 검증(silent miss 방지)
    graph_builder.add_node("reflect", partial(reflect, llm_coder=llm_coder, llm_gpt=llm_gpt))       # 오류 시 수정 재생성
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

    # add_context 이후: 오류 컨텍스트면 friendly_error, 아니면 generate
    graph_builder.add_conditional_edges(
        "add_context",
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
        "generation": None,
        "phase": None,
        "user_request": request,
        "output_formats": output_formats,
        "final_user_messages": None,
    }
    return graph.invoke(initial_state)
# ===== 엔트리포인트 끝 =====


__all__ = ["build_graph", "run_request"]
