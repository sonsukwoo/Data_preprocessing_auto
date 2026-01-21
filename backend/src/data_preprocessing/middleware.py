"""
LLM 제어를 위한 미들웨어 모듈.

3개의 미들웨어를 체인으로 구성하여 LLM 응답을 제어합니다:
1. ValidationEnforcementMiddleware: 검증 코드 구조 보장
2. ForbiddenPatternMiddleware: 금지 패턴 자동 제거
3. OutputFormatEnforcerMiddleware: 출력 형식 자동 정제
"""
from __future__ import annotations

import ast
import logging
import re
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI

from .models import MiddlewareTraceItem

logger = logging.getLogger(__name__)


# =========================
# 1. ValidationEnforcementMiddleware
# =========================
class ValidationEnforcementMiddleware:
    """검증 코드 구조를 자동으로 강제하는 미들웨어.
    
    - __validation_report__ 존재 여부 확인
    - 누락 시 검증 리포트만 생성하여 append (최대 2회)
    - 기존 코드 품질 보존
    - 2회 실패 시 None 반환 (reflect로 전달)
    """
    
    def __init__(self, llm: ChatOpenAI, max_retries: int = 2):
        self.llm = llm
        self.max_retries = max_retries
    
    def invoke_with_structure_enforcement(
        self, 
        messages: List[Any], 
        structured_output_class: Any,
        requirements: Optional[List[Any]] = None
    ) -> Any:
        """구조가 보장된 코드 생성 (검증 리포트만 재시도)."""
        try:
            # 1차 시도: 전체 코드 생성
            response = self.llm.with_structured_output(structured_output_class).invoke(messages)
            
            # 검증 코드 존재 확인
            if self._has_validation_assignment(response.code):
                logger.info("검증 코드 생성 성공 (1차 시도)")
                self._add_middleware_trace(response, MiddlewareTraceItem(
                    middleware="ValidationEnforcement",
                    action="validation_found",
                    attempt=1
                ))
                return response
            
            # 재시도: 검증 리포트만 생성하여 추가
            logger.warning(f"검증 코드 누락, 검증 리포트만 추가 재시도 (최대 {self.max_retries}회)")
            
            for attempt in range(self.max_retries):
                # 기존 코드는 유지하고, 검증 리포트만 생성
                validation_report = self._generate_validation_report_only(
                    messages, 
                    existing_code=response.code,
                    attempt=attempt,
                    requirements=requirements or []
                )
                
                if validation_report:
                    # 기존 코드에 검증 리포트 append
                    response.code = response.code + "\n\n" + validation_report
                    logger.info(f"검증 리포트 추가 성공 (재시도 {attempt+1}회)")
                    self._add_middleware_trace(response, MiddlewareTraceItem(
                        middleware="ValidationEnforcement",
                        action="validation_appended",
                        attempt=attempt + 2,
                        report_length=len(validation_report)
                    ))
                    return response
            
            # 2회 재시도 실패 → None 반환 (reflect로 전달)
            logger.error(f"검증 리포트 생성 {self.max_retries}회 실패, reflect로 전달")
            self._add_middleware_trace(response, MiddlewareTraceItem(
                middleware="ValidationEnforcement",
                action="failed",
                attempt=self.max_retries + 1
            ))
            return None
            
        except Exception as e:
            logger.error(f"ValidationEnforcementMiddleware 오류: {e}")
            return None
    
    def _add_middleware_trace(self, response: Any, trace_item: MiddlewareTraceItem) -> None:
        """미들웨어 처리 내역을 response에 기록."""
        if not hasattr(response, 'middleware_trace') or response.middleware_trace is None:
            response.middleware_trace = []
        response.middleware_trace.append(trace_item)
    
    def _has_validation_assignment(self, code: str) -> bool:
        """__validation_report__ 할당이 있는지 AST로 확인."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__validation_report__':
                            return True
            return False
        except SyntaxError:
            # 구문 오류 시 문자열 검색으로 폴백
            return '__validation_report__' in code and '=' in code
    
    def _generate_validation_report_only(
        self, 
        messages: List[Any], 
        existing_code: str, 
        attempt: int,
        requirements: List[Any]
    ) -> Optional[str]:
        """검증 리포트만 생성 (기존 코드는 건드리지 않음)."""
        # 요구사항 텍스트 추출
        requirements_text = "\n".join(
            [f"- {req.id}: {req.text}" for req in requirements]
        ) if requirements else "없음"
        
        # 검증 리포트만 생성하라는 명확한 프롬프트
        validation_prompt = f"""기존 코드는 잘 작성되었으나 __validation_report__ 변수만 누락되었습니다.

요구사항 (모두 충족해야 함):
{requirements_text}

기존 코드 (일부):
```python
{existing_code[:800]}
...
```

다음 형식의 __validation_report__ 변수 할당 코드만 작성하세요:

__validation_report__ = {{
    'ok': bool,  # 요구사항 충족 여부
    'issues': list,  # 문제점 리스트
    'metrics': dict,  # 검증 근거 지표
    'requirements': dict  # 요구사항별 충족 여부
}}

주의:
- 기존 코드를 다시 작성하지 마세요
- __validation_report__ 할당 코드만 출력하세요
- 마크다운 펜스 없이 Python 코드만 출력하세요
"""
        
        try:
            # LLM에게 검증 리포트만 요청
            response = self.llm.invoke([("user", validation_prompt)])
            validation_code = response.content.strip()
            
            # 마크다운 펜스 제거
            validation_code = re.sub(r'```python\s*', '', validation_code)
            validation_code = re.sub(r'```\s*', '', validation_code)
            validation_code = validation_code.strip()
            
            # __validation_report__ 할당이 있는지 확인
            if '__validation_report__' in validation_code:
                logger.info(f"검증 리포트 생성 성공 (시도 {attempt+1}회)")
                return validation_code
            
            logger.warning(f"검증 리포트 생성 실패 (시도 {attempt+1}회): __validation_report__ 없음")
            return None
            
        except Exception as e:
            logger.error(f"검증 리포트 생성 중 오류 (시도 {attempt+1}회): {e}")
            return None


# =========================
# 2. ForbiddenPatternMiddleware
# =========================
class ForbiddenPatternMiddleware:
    """금지된 코드 패턴 자동 감지 및 제거.
    
    금지 패턴:
    - sys.exit(), exit(), quit(), os._exit()
    - argparse.ArgumentParser()
    - if __name__ == "__main__"
    - sys.argv
    """
    
    FORBIDDEN_PATTERNS = [
        (r'sys\.exit\([^)]*\)', 'sys.exit()'),
        (r'\bexit\([^)]*\)', 'exit()'),
        (r'\bquit\([^)]*\)', 'quit()'),
        (r'os\._exit\([^)]*\)', 'os._exit()'),
        (r'argparse\.ArgumentParser', 'argparse.ArgumentParser'),
        (r'if\s+__name__\s*==\s*["\']__main__["\']\s*:', 'if __name__ == "__main__"'),
        (r'sys\.argv', 'sys.argv'),
    ]
    
    def __init__(self, wrapped_middleware: ValidationEnforcementMiddleware):
        self.wrapped = wrapped_middleware
    
    def invoke_with_structure_enforcement(
        self, 
        messages: List[Any], 
        structured_output_class: Any,
        requirements: Optional[List[Any]] = None
    ) -> Any:
        """금지 패턴 제거 후 다음 미들웨어로 전달."""
        try:
            # 다음 미들웨어 호출
            response = self.wrapped.invoke_with_structure_enforcement(
                messages, structured_output_class, requirements
            )
            
            if response is None:
                return None
            
            # 금지 패턴 검사 및 제거
            cleaned_code, violations = self._remove_forbidden_patterns(response.code)
            
            if violations:
                logger.warning(f"금지된 패턴 {len(violations)}개 발견 및 제거: {violations}")
                response.code = cleaned_code
                # trace 기록
                if not hasattr(response, 'middleware_trace') or response.middleware_trace is None:
                    response.middleware_trace = []
                response.middleware_trace.append(MiddlewareTraceItem(
                    middleware="ForbiddenPattern",
                    action="patterns_removed",
                    failed_reason=f"Violations: {violations}"
                ))
            
            return response
            
        except Exception as e:
            logger.error(f"ForbiddenPatternMiddleware 오류: {e}")
            # 다음 미들웨어 결과를 그대로 반환 (오류 무시)
            try:
                return self.wrapped.invoke_with_structure_enforcement(
                    messages, structured_output_class, requirements
                )
            except Exception:
                return None
    
    def _remove_forbidden_patterns(self, code: str) -> tuple[str, list[str]]:
        """금지된 패턴 감지 및 제거 (완전 삭제)."""
        violations = []
        cleaned_code = code
        
        for pattern, name in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, cleaned_code):
                violations.append(name)
                # 해당 라인 완전 삭제 (빈 라인도 제거)
                cleaned_code = re.sub(
                    f'^.*{pattern}.*$\n?',
                    '',
                    cleaned_code,
                    flags=re.MULTILINE
                )
        
        return cleaned_code, violations


# =========================
# 3. OutputFormatEnforcerMiddleware
# =========================
class OutputFormatEnforcerMiddleware:
    """출력 형식 자동 정제.
    
    기능:
    - 마크다운 펜스 제거 (```python, ```)
    - 설명 텍스트 제거 ("Here's the code:", "아래는..." 등)
    - 불필요한 공백 정리
    """
    
    def __init__(self, wrapped_middleware: ForbiddenPatternMiddleware):
        self.wrapped = wrapped_middleware
    
    def invoke_with_structure_enforcement(
        self, 
        messages: List[Any], 
        structured_output_class: Any,
        requirements: Optional[List[Any]] = None
    ) -> Any:
        """형식 정제 후 다음 미들웨어로 전달."""
        try:
            # 다음 미들웨어 호출
            response = self.wrapped.invoke_with_structure_enforcement(
                messages, structured_output_class, requirements
            )
            
            if response is None:
                return None
            
            # 형식 정제
            original_code = response.code
            cleaned_code = self._clean_format(original_code)
            
            if original_code != cleaned_code:
                logger.info("출력 형식 정제됨 (마크다운 펜스 제거 등)")
                response.code = cleaned_code
                # trace 기록
                if not hasattr(response, 'middleware_trace') or response.middleware_trace is None:
                    response.middleware_trace = []
                response.middleware_trace.append(MiddlewareTraceItem(
                    middleware="OutputFormatEnforcer",
                    action="format_cleaned"
                ))
            
            return response
            
        except Exception as e:
            logger.error(f"OutputFormatEnforcerMiddleware 오류: {e}")
            # 다음 미들웨어 결과를 그대로 반환 (오류 무시)
            try:
                return self.wrapped.invoke_with_structure_enforcement(
                    messages, structured_output_class, requirements
                )
            except Exception:
                return None
    
    def _clean_format(self, code: str) -> str:
        """마크다운 펜스 및 불필요한 텍스트 제거."""
        # 마크다운 펜스 제거
        code = re.sub(r'```python\s*', '', code)
        code = re.sub(r'```\s*', '', code)
        
        # "Here's the code:", "아래는 코드입니다" 같은 설명 제거
        code = re.sub(
            r'^(?:Here[\'\']?s?\s+(?:the\s+)?code[:\s]*|아래는.*코드.*|다음은.*코드.*)\n?',
            '',
            code,
            flags=re.MULTILINE | re.IGNORECASE
        )
        
        # 앞뒤 공백 정리
        return code.strip()


# =========================
# 미들웨어 체인 생성 유틸리티
# =========================
def create_middleware_chain(
    base_llm: ChatOpenAI, 
    max_retries: int = 2
) -> OutputFormatEnforcerMiddleware:
    """미들웨어 체인 생성.
    
    체인 순서 (안쪽 → 바깥쪽):
    1. ValidationEnforcementMiddleware (가장 안쪽 - LLM 직접 호출)
    2. ForbiddenPatternMiddleware (중간)
    3. OutputFormatEnforcerMiddleware (가장 바깥쪽)
    
    Args:
        base_llm: 기본 LLM 인스턴스
        max_retries: 검증 리포트 재시도 최대 횟수
        
    Returns:
        OutputFormatEnforcerMiddleware: 체인의 최상위 미들웨어
    """
    # 1. 검증 코드 보장 (가장 안쪽)
    validation_mw = ValidationEnforcementMiddleware(base_llm, max_retries=max_retries)
    # 2. 금지 패턴 제거
    forbidden_mw = ForbiddenPatternMiddleware(validation_mw)
    # 3. 형식 정제 (가장 바깥쪽)
    output_format_mw = OutputFormatEnforcerMiddleware(forbidden_mw)
    
    return output_format_mw


__all__ = [
    "ValidationEnforcementMiddleware",
    "ForbiddenPatternMiddleware",
    "OutputFormatEnforcerMiddleware",
    "create_middleware_chain",
]
