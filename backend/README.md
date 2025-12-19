# 데이터 전처리 에이전트 Backend (LangGraph + FastAPI)

Jupyter 노트북에서 출발한 LangGraph 기반 파이프라인을 **모듈 + API 서버 형태**로 정리한 프로젝트입니다.
브라우저 데모 UI(`../front`)가 호출하는 FastAPI 서버와, 실제 LangGraph 워크플로우/툴/프롬프트가 포함되어 있습니다.

루트 README: `../README.md`

## 프로젝트 구조
- `src/data_preprocessing/`
  - `tools.py` · 데이터 샘플링/요약 도구 (`@tool`)
  - `prompts.py` · 코드 생성·수정 프롬프트
  - `models.py` · 코드 구조(pydantic), 그래프 상태 정의
  - `workflow.py` · LangGraph 그래프 빌더 및 실행 로직
  - `cli.py` · CLI 엔트리포인트
- `Makefile` · 표준 작업 명령 모음
- `requirements.txt` · 의존성 목록
- `.env.example` · OpenAI API 키 템플릿

## 빠른 시작 (venv)
필요 패키지는 가상환경에서 설치합니다.

```bash
cd backend
make bootstrap                # .venv 생성 및 패키지 설치
source .venv/bin/activate     # 가상환경 활성화 (Windows는 .venv\\Scripts\\activate)
make run REQUEST="<데이터 요청 문구>" MAX_ITERATIONS=3
```

예시:
```bash
make run REQUEST="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv 데이터를 전처리해줘"
```

## 직접 실행(activate 후)
```bash
python -m data_preprocessing.cli --request "<요청 문구>" --max-iterations 3
```

## 환경 변수
`.env.example`를 참고해 `.env`를 만들고 `OPENAI_API_KEY`를 채우세요.

- 서버/도커 같은 비대화형 환경에서는 키가 없으면 실패합니다.
- 로컬 터미널에서 CLI 실행 시에는(tty) 필요하면 입력 프롬프트가 뜰 수 있습니다.

## 테스트
현재 자동화 테스트는 없습니다. 필요 시 `tests/` 디렉터리를 만들고 `make test`로 실행할 수 있습니다.

## FastAPI 백엔드 실행 (프런트엔드 연동)
프런트(`../front`)가 호출하는 API는 FastAPI로 제공합니다.

```bash
cd backend
make bootstrap          # 1회, 가상환경 + 의존성 설치
make serve              # uvicorn --reload, 기본 포트 8000
```

주요 엔드포인트
- `GET /health` · 헬스 체크
- `POST /run_stream` · NDJSON 스트리밍 실행(프런트 기본 사용)
- `POST /run` · 단발 실행(JSON `{question, max_iterations, output_format}`)
- `POST /upload` · 폼 업로드(필드: `files`, 다중/폴더 업로드 지원). 서버에 저장 후 로컬 경로 반환.
- `POST /s3/create_upload_session`, `POST /s3/presign_put` · (옵션) S3 presigned 업로드 지원
- `GET /downloads/{run_id}/{filename}` · 산출물 다운로드
- `GET /downloads/{run_id}/{filename}/preview?n=5` · 산출물 미리보기(표 파일만)

프런트는 도커/Nginx 환경에서 `/api`로 호출하도록 구성돼 있습니다.
