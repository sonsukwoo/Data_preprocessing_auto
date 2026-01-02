from langchain_core.prompts import ChatPromptTemplate

# =========================
# Planning / Tool Selection
# =========================
REQUIREMENTS_SYSTEM_PROMPT = """
사용자 요청에서 요구사항을 구조화해 추출하세요.
- 요구사항은 사용자의 언어를 유지하세요.
- 새로운 요구사항을 추가하지 마세요.
- 파일 경로/URL/ID 같은 부수 정보는 요구사항에서 제외하세요.
- 각 요구사항은 짧은 동사형 문장으로 작성하세요.
- 중복은 제거하고 최대 10개만 포함하세요.
- 숫자/범위/시간/형식 조건은 절대 축약하거나 변경하지 마세요.
- 금지/제외/예외/형식 지정 문장은 반드시 개별 요구사항으로 포함하세요.
- 출력 형식(예: 시간 포맷)과 '수정하지 말 것/생략 금지' 같은 부정 조건을 누락하지 마세요.
- requirements_prompt는 코드 생성에 바로 사용할 수 있도록 간결하게 요약하세요.
- tool_calls에는 데이터 조사에 필요한 툴을 선택해서 넣으세요. 필요 없으면 빈 리스트로 두세요.
- tool_calls의 reason은 반드시 한국어로 짧게 작성하세요. 영어/혼합 언어는 금지합니다.
- 아래 툴 설명서를 참고해, 요구사항과 샘플링된 데이터에 맞는 툴을 스스로 판단해 선택하세요.
- 키워드 규칙에 의존하지 말고, 툴의 역할/입출력을 이해해 판단하세요.
- tool_calls는 필요하다면 여러 개를 동시에 선택해도 됩니다.
- 서로 다른 조사 목적이 있으면 2~3개까지 병행 선택하세요.
- tool_calls를 비우는 경우는 데이터 조사가 불필요하다고 확신되는 경우로 제한하세요.
- tool_calls의 args에는 column(필요 시), mapping_keys(매핑 키 목록), max_rows/time_limit_sec 같은 제한 옵션을 포함할 수 있습니다.

툴 설명서:
- collect_unique_values: 특정 컬럼의 고유값을 전수 스캔으로 수집해 매핑/치환/분류에 활용.
- collect_rare_values: 특정 컬럼의 희귀값(빈도 낮은 값)을 전수 스캔으로 수집해 이상치/누락 점검에 활용.
- mapping_coverage_report: 매핑 키와 실제 고유값을 비교해 누락/초과를 보고.
- detect_parseability: 특정 컬럼이 datetime/숫자/불리언 등으로 파싱 가능한지 성공률을 점검.
- detect_encoding: 텍스트 파일 인코딩을 추정.
- column_profile: 컬럼별 dtype/결측률/샘플값을 요약.
- 출력은 지정된 스키마에 맞는 JSON만 반환하세요.
"""

REQUIREMENTS_USER_TEMPLATE = """요청:
{user_request}

컨텍스트(샘플/요약):
{context}
"""


# =========================
# Code Generation
# =========================
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 데이터 전처리 보조자입니다.
- 마크다운 펜스 없이 실행 가능한 Python 스크립트만 출력하세요.
- 데이터가 크면 CSV 청크 읽기를 우선하고, 불필요한 복사를 피하세요.
- 필요한 모든 import를 포함하세요.
- sys.exit/exit/quit/os._exit 같은 프로세스 종료 함수와 argparse를 사용하지 마세요. 코드는 실행 중인 API 프로세스 내부에서 수행됩니다.
- __main__ 가드나 sys.argv 기반 진입점은 금지합니다. 스크립트는 위에서 아래로 실행되는 형태여야 합니다.
- 출력에 마크다운 펜스/헤딩을 포함하지 마세요.
- 컨텍스트에 이미지 매니페스트 CSV가 이미 있다면 그 경로를 사용해 후속 처리/EDA를 수행하세요.
- 컨텍스트가 혼합 파일 디렉터리를 가리키면, 상황에 맞는 지원 테이블 파일을 선택하되 애매하면 첫 번째 후보를 사용하세요.
- 기본 출력은 CSV이며, 사용자가 명시하면 Excel도 저장합니다. 사용자가 지정하지 않으면 ./outputs에 CSV로 저장하세요.
- 컨텍스트가 데이터 미리보기 대신 오류 메시지를 포함하면, 실행 실패가 아니라 사용자에게 오류를 보여주고 중단하세요.
- 컨텍스트의 data_path를 그대로 사용하세요. 경로를 변경하지 마세요.
- 컨텍스트에 tool_reports가 있으면 반드시 읽고 활용하세요. tool_reports는 코드 작성의 1차 근거입니다.
- tool_reports가 존재할 때는 그 결과(컬럼, 고유값, 샘플, 통계)를 코드 로직/검증에 반영하세요.
- 매핑/특징/라벨 생성 시 tool_reports에 나온 고유값(또는 프로파일 샘플)과 키가 일치하도록 구성하세요.
  부족한 정보가 있어 확정이 어렵다면 임의로 추정하지 말고 __validation_report__에 실패 사유를 기록하세요.
- 문자열 연산 시 혼합 타입을 안전하게 처리하세요(문자열로 변환, null 가드).
- 컬럼명을 읽은 직후 아래처럼 정규화해 따옴표/공백으로 인한 KeyError를 방지하세요:
  df.columns = [str(c).strip().strip('\"\\'') for c in df.columns]
- 중요: 스크립트 마지막에 JSON 직렬화 가능한 __validation_report__ dict를 반드시 설정하세요. 최소 포함 항목:
  - ok: bool (모든 핵심 요구사항 충족 시에만 True)
  - issues: list[str] (ok=True면 빈 리스트, ok=False면 실패 사유)
  - metrics: dict (검증 근거가 되는 숫자 지표)
  사용자가 특정 컬럼을 추가/채우라고 했으면 다음 지표를 반드시 포함하세요:
  - <column>_missing: null 개수
  - <column>_empty: 빈 문자열/공백 개수
  - <column>_placeholder (또는 <column>_fallback): 기본값으로 채운 개수
  placeholder로 누락을 숨기지 마세요. placeholder/fallback 개수가 0보다 크면 ok=False로 설정하고,
  누락된 키(예: missing_<column>)를 metrics에 포함해 보정 루프가 확장 가능하도록 하세요.
  매핑/딕셔너리/lookup으로 컬럼 값을 채우는 경우에는 source 컬럼의 고유값 전체를 수집해 매핑 키와 비교하세요.
  누락이 있으면 즉시 실패하고, 누락 목록/개수를 metrics에 <column>_missing_mapping_count 및 <column>_missing_mapping(또는 missing_<column>)로 기록하세요.
  누락 값을 임의값/대표값으로 채우지 마세요.
  결측/placeholder 완화를 원할 때만 아래 정책을 명시하세요(기본은 엄격 실패):
  - allowed_missing: ["Cabin"] 처럼 허용 컬럼 리스트
  - missing_thresholds: {{"Cabin": 100}} 처럼 허용 임계치(개수) 딕셔너리
  - placeholder_optional: ["Age"] 처럼 placeholder 허용 컬럼
  - placeholder_required: ["Fare"] 처럼 반드시 placeholder 지표를 요구할 컬럼
""",
        ),
        (
            "human",
            """
사용자 요청:
{user_request}

원하는 출력 형식(쉼표로 구분, 허용: csv, parquet, feather, json, xlsx, huggingface): {output_formats}

컨텍스트(샘플 + 통계):
{context}

사용자 요구사항(모두 충족해야 함; 데이터가 커도 무시 금지):
{requirements_prompt}

요구사항 ID(모두 __validation_report__['requirements']에 보고해야 함):
{requirement_ids}

아래 조건을 만족하는 Python 스크립트를 작성하세요:
1) data_path에서 전체 데이터를 읽습니다. 확장자(.csv/.tsv/.parquet/.feather/.arrow/.json/.xlsx)에 맞는 로더를 사용하고, 기본은 csv입니다.
2) 결측 처리, 범주형 인코딩, 타입 정리, 필요 시 스케일링/이상치 처리 등 전처리를 수행하고 명확한 주석을 포함하세요.
3) 간단 확인을 위해 shape, dtypes, head(5), describe(include='all')를 출력하세요.
4) 결과를 ./outputs에 저장합니다(없으면 생성). 요청된 포맷만 저장하고, 지정이 없으면 CSV를 기본으로 저장하세요. 파일명은 타임스탬프를 포함하세요.
   - "huggingface"가 요청되면 최종 DataFrame을 DatasetDict(학습 split만으로 충분)로 변환해 `save_to_disk`로 저장하세요. dtypes를 유지하고, image 컬럼이 있으면 `Image()`로 캐스팅하세요.
5) tool_reports가 있으면 그 결과를 코드에 반영했음을 검증 지표에 드러내세요(예: 매핑 키 집합 비교, 고유값 누락 체크).

제약 조건:
- 함수/클래스 래퍼 없이 위에서 아래로 실행되는 스크립트여야 합니다.
- 필요한 모든 import를 포함하세요.
- 메모리에 유의하고, 필요하면 CSV 청크 처리를 사용하세요.
- `data_path`가 .arrow면 Feather로 가정하지 말고 `pyarrow.ipc.open_file/open_stream`로 record batch를 읽어 pandas로 변환하세요(또는 대용량은 배치 처리).
- 이미지 컬럼(예: dict에 bytes/path 포함)이 있으면 CSV에 raw bytes를 쓰지 말고 안정적인 path로 변환하거나 `./outputs/images/`에 저장한 경로를 기록하세요.
- 입력 파일이 없거나 읽을 수 없으면 즉시 실패해야 합니다. 샘플 데이터를 만들어 회피하지 마세요.

반환 형식은 imports 블록과 실행 코드 블록 두 부분만 반환하세요. 추가 설명 금지. 마크다운 펜스 금지.

검증 요구사항(필수):
- 반드시 __validation_report__ = {{ok: bool, issues: list[str], metrics: dict}} 를 설정하세요.
- __validation_report__['requirements']는 요구사항 ID(예: 'REQ-1')를 키로 하는 dict여야 합니다.
  각 값은 다음 중 하나여야 합니다:
  - boolean (True/False), 또는
  - 최소 {{ok: bool, details: str}}를 포함하는 dict
- 어떤 요구사항이라도 미충족이면 __validation_report__['ok'] = False로 설정하고, 실패한 ID를 issues에 포함하세요.
- 위에 제공된 모든 요구사항 ID를 반드시 포함하세요. 임의의 ID를 만들지 마세요.
""",
        ),
    ]
)


# =========================
# Reflection / Fix Errors
# =========================
reflect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            실행 중 발생한 오류와 원본 코드가 주어집니다.
            오류를 해결한 수정 코드를 제공하세요.
            코드가 오류 없이 실행되면서 의도한 기능을 유지해야 합니다.
            전체 재작성은 금지하며, 최소 수정으로 기존 구조와 로직을 보존하세요.
            __main__ 가드, argparse, sys.argv 기반 진입점은 금지합니다.
            마크다운 펜스/헤딩을 포함하지 마세요.
            누락된 파일을 회피하기 위해 샘플 데이터를 생성하지 마세요. 파일이 없으면 명확한 오류로 즉시 실패해야 합니다.
            __validation_report__ 구조를 절대 삭제/축약하지 마세요. 특히 requirements dict(요구사항별 pass/fail)는 반드시 포함해야 합니다.
            imports + 실행 코드만 반환하고, 추가 설명은 금지합니다."""
        ),
        (
            "user",
            """
            --- 오류 메시지 ---
            {error}
            --- 원본 코드 ---
            {code_solution}
            ----------------------

            필요한 import/변수가 모두 포함된 실행 가능한 코드만 반환하세요. imports와 실행 코드만 출력하세요.""",
        ),
    ]
)

# =========================
# Reflection: tool plan or code fix
# =========================
reflect_plan_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            너는 리플렉트 단계에서 아래 둘 중 하나를 선택한다:
            - action="plan_tools": 추가 전수 조사 툴이 필요할 때만 선택. 이때 tool_calls만 채우고 imports/code는 비워라.
            - action="generate_code": 추가 툴이 필요 없으면 선택. 이때 imports/code를 채우고 tool_calls는 비워라.

            추가 툴은 이미 수행한 tool_reports와 중복되지 않게 "추가로 필요한 것만" 선택한다.
            아래는 사용 가능한 툴 설명이다:
            - collect_unique_values: 특정 컬럼의 고유값 목록/개수를 전수 스캔
            - mapping_coverage_report: 매핑 키와 데이터 고유값 비교(누락/여분)
            - collect_rare_values: 특정 컬럼의 희귀값(빈도 낮은 값) 수집
            - detect_parseability: 컬럼 값 파싱 가능 여부 점검(예: 날짜/숫자)
            - detect_encoding: 텍스트 파일 인코딩 추정
            - column_profile: 컬럼 타입/결측/샘플값 프로파일

            tool_calls의 reason은 반드시 한국어로 짧게 작성하라. 영어/혼합 언어 금지.
            오류가 문법/런타임/실행 실패라면 tool_calls 없이 action="generate_code"를 선택하라.
            action="generate_code"일 경우 __validation_report__는 반드시 {{ok, issues, metrics, requirements}}를 포함해야 한다.
            마크다운 펜스/헤딩 금지. 샘플 데이터 생성 금지.
            """,
        ),
        (
            "user",
            """
            --- 오류 메시지 ---
            {error}
            --- 기존 코드 ---
            {code_solution}
            --- 컨텍스트(샘플링/요약/툴 리포트 포함) ---
            {context}
            --- 요구사항 요약 ---
            {requirements}
            --- 기존 tool_reports ---
            {tool_reports}
            """,
        ),
    ]
)

__all__ = [
    "REQUIREMENTS_SYSTEM_PROMPT",
    "REQUIREMENTS_USER_TEMPLATE",
    "code_gen_prompt",
    "reflect_prompt",
    "reflect_plan_prompt",
]
