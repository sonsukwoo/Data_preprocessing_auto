from __future__ import annotations

from typing import Any, Optional, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field


class Requirement(BaseModel):
    """사용자 요구사항을 검증 가능한 단위로 쪼갠 항목."""

    id: str = Field(description="Stable requirement id, e.g. REQ-1")
    text: str = Field(description="Original requirement text (natural language)")
    severity: Literal["must", "should"] = Field(default="must", description="Whether this requirement must pass")


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


__all__ = ["Requirement", "CodeBlocks", "State"]
