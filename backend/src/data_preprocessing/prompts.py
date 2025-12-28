from langchain_core.prompts import ChatPromptTemplate

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
- 문자열 연산 시 혼합 타입을 안전하게 처리하세요(문자열로 변환, null 가드).
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
- 각 요구사항 ID마다 최소 1개 이상의 숫자 지표를 계산해 __validation_report__['metrics']에 '{{REQ_ID}}_' 접두사로 저장하세요.
  (예: 'REQ-1_missing', 'REQ-2_coverage'). 증거 지표가 없으면 해당 요구사항을 True로 표시하지 마세요.
- 요구사항 결과를 기본값(True)로 두지 말고, 계산된 지표로부터 도출하세요.
""",
        ),
    ]
)

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

__all__ = ["code_gen_prompt", "reflect_prompt"]
