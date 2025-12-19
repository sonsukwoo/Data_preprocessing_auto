from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ensure_api_keys
from .workflow import run_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph 기반 데이터 전처리 코드 생성기")
    parser.add_argument(
        "--request",
        required=True,
        help="사용자 요청 문구 (데이터 경로 포함 권장)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="오류 발생 시 수정 시도 횟수",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="대화/툴 호출에 사용할 OpenAI 모델명",
    )
    parser.add_argument(
        "--coder-model",
        default="gpt-4.1",
        help="코드 생성/수정에 사용할 OpenAI 모델명",
    )
    parser.add_argument(
        "--env-file",
        default=Path(__file__).resolve().parents[2] / ".env",
        help="환경 변수 로드용 .env 경로",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_api_keys(args.env_file)

    result = run_request(
        request=args.request,
        max_iterations=args.max_iterations,
        llm_model=args.llm_model,
        coder_model=args.coder_model,
    )

    generation = result.get("generation")
    if generation:
        print("\n[IMPORTS]\n" + generation.imports)
        print("\n[CODE]\n" + generation.code)
    else:
        # friendly_error 경로 등으로 generation이 없을 때 HumanMessage 직렬화가 불가하므로
        # 메시지 내용을 안전하게 출력한다.
        messages = result.get("messages") or []
        if messages:
            print("\n[MESSAGES]")
            for m in messages:
                if isinstance(m, tuple) and len(m) >= 2:
                    role, content = m[0], m[1]
                    print(f"{role}: {content}")
                elif hasattr(m, "content"):
                    print(getattr(m, "content", m))
                else:
                    print(m)
        # context에 남은 오류 힌트가 있으면 같이 출력
        ctx = result.get("context")
        if isinstance(ctx, str) and ctx:
            print("\n[CONTEXT]\n" + ctx)
        # 마지막 보호: 직렬화 가능 객체만 JSON으로 출력
        safe = {k: v for k, v in result.items() if isinstance(v, (str, int, float, bool, type(None), dict, list))}
        if safe:
            print("\n[RAW RESULT]")
            print(json.dumps(safe, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
