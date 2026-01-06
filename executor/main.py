import sys
import os
import io
import traceback
import json
from contextlib import redirect_stdout
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI()

class ExecuteRequest(BaseModel):
    code: str
    imports: str = ""
    workdir: Optional[str] = None  # 실행할 작업 디렉토리 (상대 경로)

@app.post("/execute")
async def execute_code(req: ExecuteRequest):
    """
    제공된 파이썬 코드를 로컬 스코프에서 실행합니다.
    stdout을 캡처하고 로컬 스코프에서 발견된 검증 결과와 함께 반환합니다.
    """
    # 1. 실행 환경 준비
    # 백엔드와 마찬가지로 /app/outputs로 디렉토리를 변경합니다.
    # workdir가 제공되면 해당 하위 디렉토리로 이동합니다.
    original_cwd = os.getcwd()
    execution_workdir = "/app/outputs"
    
    if req.workdir:
        # workdir가 절대경로가 아니면 /app/outputs 기준 상대경로로 처리
        # 보안: 상위 경로(../) 접근은 기본적으로 os.makedirs/chdir에서 처리되지만,
        # 여기서는 간단히 결합합니다. (컨테이너 내부이므로 호스트 위험은 제한적)
        if not req.workdir.startswith("/"):
            execution_workdir = os.path.join(execution_workdir, req.workdir)
        else:
             execution_workdir = req.workdir
            
    os.makedirs(execution_workdir, exist_ok=True)
    
    # stdout 캡처
    stdout_capture = io.StringIO()
    
    # 단일 통합 스코프 생성 (globals=locals)
    # 이것이 스크립트 실행 시뮬레이션에 가장 적합하며 Scope 문제를 방지합니다.
    context = {"__builtins__": __builtins__}
    
    error_kind = None
    error_detail = None
    
    try:
        os.chdir(execution_workdir)
        
        # 2. Imports 실행
        # Imports는 보통 실행하기에 안전하지만, 동일한 스코프 내에서 실행합니다.
        if req.imports:
            with redirect_stdout(stdout_capture):
                # import 블록을 먼저 컴파일하고 실행합니다.
                exec(req.imports, context, context)
        
        # 3. 메인 코드 실행
        if req.code:
            with redirect_stdout(stdout_capture):
                exec(req.code, context, context)
                
    except Exception:
        # 트레이스백 캡처
        error_kind = "RuntimeError"
        error_detail = traceback.format_exc()
    finally:
        # CWD 복원
        os.chdir(original_cwd)
        
    captured_stdout = stdout_capture.getvalue()
    
    # 4. 결과 추출 (검증 리포트)
    # 에이전트는 __validation_report__ 또는 validation_report에 씁니다.
    # 백엔드로 보내기 위해 이것들을 추출합니다.
    results = {}
    target_keys = ["__validation_report__", "validation_report", "requirements", "metrics"]
    
    # 통합된 context에서 검색
    search_scope = context
    
    for key in target_keys:
        if key in search_scope:
            val = search_scope[key]
            # 안전한 직렬화 (Sanitize)
            # Numpy 타입(int64, float32 등)은 FastAPI의 기본 인코더가 처리하지 못해 500 에러를 유발합니다.
            # 따라서 json.dumps(..., default=str)로 강제 변환 후 다시 로드하여
            # 순수 Python 타입(dict, list, str, int 등)만 남깁니다.
            try:
                # 1. JSON 문자열로 변환 (알 수 없는 타입은 str로 변환)
                serialized = json.dumps(val, default=str)
                # 2. 다시 Python 객체로 복원 (Numpy 타입은 사라짐)
                sanitized_val = json.loads(serialized)
                results[key] = sanitized_val
            except Exception:
                # 위 과정 실패 시, 전체를 문자열로 변환하여 반환
                results[key] = str(val)

    return {
        "stdout": captured_stdout,
        "error_kind": error_kind,
        "error_detail": error_detail,
        "results": results,
        "execution_workdir": execution_workdir
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
