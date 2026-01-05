from __future__ import annotations

from typing import Any, Optional, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class ToolCallArgs(BaseModel):
    """툴 호출 파라미터(스키마 고정)."""

    path: Optional[str] = Field(default=None, description="데이터 파일/폴더 경로")
    column: Optional[str] = Field(default=None, description="단일 컬럼명")
    columns: Optional[list[str]] = Field(default=None, description="복수 컬럼명")
    mapping_keys: Optional[list[str]] = Field(default=None, description="매핑 키 목록")
    parsers: Optional[list[str]] = Field(default=None, description="파싱 검사 유형 목록")
    top_k: Optional[int] = Field(default=None, description="상위 빈도값 개수")
class ToolCall(BaseModel):
    """LLM이 선택한 데이터 조사 툴 호출."""

    name: str = Field(description="Tool name to call")
    args: ToolCallArgs = Field(default_factory=ToolCallArgs, description="Arguments for the tool")
    reason: str = Field(default="", description="툴 선택 이유(짧게, 한국어)")


class Requirement(BaseModel):
    """사용자 요구사항을 검증 가능한 단위로 쪼갠 항목."""

    id: str = Field(description="Stable requirement id, e.g. REQ-1")
    text: str = Field(description="Original requirement text (natural language)")
    severity: Literal["must", "should"] = Field(default="must", description="Whether this requirement must pass")


class RequirementsPayload(BaseModel):
    """LLM이 추출한 요구사항 구조."""

    requirements: list[Requirement] = Field(default_factory=list)
    requirements_prompt: str = Field(
        default="",
        description="Code-generation-friendly summary of user requirements",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="Optional tool calls for data inspection prior to code generation.",
    )


class ReflectPlanPayload(BaseModel):
    """리플렉트 단계에서 툴 추가 여부/수정 코드 여부를 함께 결정."""

    action: Literal["generate_code", "plan_tools"] = Field(
        default="generate_code", description="Whether to generate code or plan extra tools."
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Additional tool calls if action=plan_tools."
    )
    imports: str = Field(default="", description="Imports block (when action=generate_code)")
    code: str = Field(default="", description="Code block (when action=generate_code)")


class CodeBlocks(BaseModel):
    """LLM이 생성한 코드를 구조화해 담는 모델."""

    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


class State(MessagesState):
    """모든 노드가 공유하는 LangGraph 상태."""

    run_id: str
    error: str
    context: Optional[str]
    generation: Optional[CodeBlocks]
    iterations: int
    phase: Optional[str] = None
    user_request: str
    output_formats: Optional[str]
    requirements: Optional[list[Requirement]] = None
    requirements_prompt: Optional[str] = None
    planned_tools: Optional[list[ToolCall]] = None
    tool_reports: Optional[list[dict[str, Any]]] = None
    reflect_action: Optional[str] = None
    tool_plan_origin: Optional[str] = None
    output_files: Optional[list[str]] = None
    tool_call_name: Optional[str] = None
    tool_call_args: Optional[dict[str, Any]] = None
    inspect_result: Optional[dict[str, Any]] = None
    sample_json: Optional[str] = None
    summary_context: Optional[str] = None
    context_candidate: Optional[str] = None
    execution_stdout: Optional[str] = None
    validation_report: Optional[dict[str, Any]] = None
    execution_workdir: Optional[str] = None
    final_user_messages: Optional[list[Any]] = None
    trace: Optional[list[dict[str, Any]]] = None


__all__ = [
    "ToolCallArgs",
    "ToolCall",
    "Requirement",
    "RequirementsPayload",
    "ReflectPlanPayload",
    "CodeBlocks",
    "State",
]
