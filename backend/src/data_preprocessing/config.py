from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def ensure_api_keys(env_path: str | Path | None = None, *, interactive: bool | None = None) -> None:
    """(선택) .env를 로드하고 OPENAI_API_KEY가 설정돼 있는지 확인.

    - interactive=True: 키가 없으면 터미널 프롬프트로 입력받음(로컬 CLI용)
    - interactive=False: 키가 없으면 예외 발생(서버/도커 등 비대화형 환경용)
    """

    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        if interactive is None:
            interactive = sys.stdin.isatty()

        if interactive:
            os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI_API_KEY: ")
            return

        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it as an environment variable "
            "(e.g., export OPENAI_API_KEY=...) or provide a .env file."
        )


__all__ = ["ensure_api_keys"]
