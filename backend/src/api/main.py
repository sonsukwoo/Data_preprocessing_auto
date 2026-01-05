from __future__ import annotations

import json
import logging
import os
import re
import traceback
import asyncio
import time
import shutil
import zipfile
import unicodedata
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.config import Config
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from starlette.responses import FileResponse
from pydantic import BaseModel, Field

from data_preprocessing.config import ensure_api_keys
from data_preprocessing.workflow import build_graph, run_request, write_internal_trace_markdown

# 디렉터리 경로 정의
ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load .env once on startup so OpenAI API 키가 환경 변수에 적용됨
ensure_api_keys(ROOT_DIR / ".env")

app = FastAPI(title="Data Preprocessing Agent API", version="0.1.0")
logger = logging.getLogger(__name__)

# 모델 선택 허용 리스트 (프론트 옵션과 동일하게 유지)
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_CODER_MODEL = "gpt-4.1"
ALLOWED_LLM_MODELS = {
    "gpt-4.1-nano",
    "gpt-4o-mini",
    "gpt-4o",
}
ALLOWED_CODER_MODELS = {
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-5.1",
}

# 프론트엔드가 file:// 혹은 다른 포트에서 접근할 수 있으므로 와일드카드 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    question: str = Field(..., description="사용자 요청 문장 (업로드 경로 포함 가능)")
    max_iterations: int = Field(5, ge=1, le=10)
    output_format: str | None = Field(
        None,
        description="저장 형식 지정 (csv, parquet, feather, json, xlsx, huggingface 중 하나)",
    )
    llm_model: str | None = Field(None, description="대화/툴 호출에 사용할 모델명")
    coder_model: str | None = Field(None, description="코드 생성/수정에 사용할 모델명")


class Message(BaseModel):
    role: str
    content: str


class RunResponse(BaseModel):
    imports: str | None = None
    code: str | None = None
    messages: List[Message] = Field(default_factory=list)
    run_id: str | None = None
    output_files: List[str] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)


_RUN_TTL_SECONDS_DEFAULT = 30 * 60
_RUN_CLEANUP_INTERVAL_SECONDS_DEFAULT = 5 * 60
_RUN_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def _run_ttl_seconds() -> int:
    return _get_int_env("RUN_OUTPUT_TTL_SECONDS", _RUN_TTL_SECONDS_DEFAULT)


def _run_cleanup_interval_seconds() -> int:
    return _get_int_env("RUN_OUTPUT_CLEANUP_INTERVAL_SECONDS", _RUN_CLEANUP_INTERVAL_SECONDS_DEFAULT)


def _register_run_outputs(run_id: str, files: List[str]) -> None:
    run_id = (run_id or "").strip()
    if not run_id:
        return
    safe_files: List[str] = []
    for f in files or []:
        name = Path(str(f)).name
        if not name:
            continue
        safe_files.append(name)
    _RUN_REGISTRY[run_id] = {
        "created_at": time.time(),
        "files": safe_files,
    }


def _get_run_outputs(run_id: str) -> Tuple[float, List[str]] | None:
    item = _RUN_REGISTRY.get(run_id)
    if not isinstance(item, dict):
        return None
    created_at = item.get("created_at")
    files = item.get("files")
    if not isinstance(created_at, (int, float)) or not isinstance(files, list):
        return None
    return float(created_at), [Path(str(f)).name for f in files if str(f).strip()]


def _normalize_model(value: str | None, allowed: set[str], default: str, field_name: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        candidate = value.strip()
    else:
        candidate = str(value).strip()
    if not candidate:
        return default
    if candidate not in allowed:
        raise HTTPException(status_code=400, detail=f"허용되지 않은 {field_name}: {candidate}")
    return candidate


def _cleanup_expired_runs(now_ts: float) -> int:
    ttl = _run_ttl_seconds()
    deleted = 0
    expired = [rid for rid, meta in _RUN_REGISTRY.items() if (now_ts - float(meta.get("created_at", 0))) > ttl]
    for rid in expired:
        meta = _RUN_REGISTRY.pop(rid, None) or {}
        files = meta.get("files") or []
        for f in files:
            name = Path(str(f)).name
            if not name:
                continue
            path = (OUTPUT_DIR / name).resolve()
            try:
                path.relative_to(OUTPUT_DIR.resolve())
            except ValueError:
                continue
            path.unlink(missing_ok=True)
        deleted += 1
    return deleted


def _latest_mtime(path: Path) -> float:
    try:
        base = path.stat().st_mtime
    except Exception:
        return 0.0
    if path.is_file():
        return float(base)
    latest = float(base)
    try:
        for root, _dirs, files in os.walk(path):
            for fn in files:
                p = Path(root) / fn
                try:
                    latest = max(latest, float(p.stat().st_mtime))
                except Exception:
                    continue
    except Exception:
        return latest
    return latest


def _cleanup_old_uploads(now_ts: float) -> int:
    """UPLOAD_DIR 아래의 오래된 업로드(파일/폴더)를 TTL 기준으로 정리."""
    ttl = _run_ttl_seconds()
    deleted = 0
    root = UPLOAD_DIR.resolve()
    if not root.exists():
        return 0
    try:
        for item in root.iterdir():
            try:
                resolved = item.resolve()
                resolved.relative_to(root)
            except Exception:
                continue
            latest = _latest_mtime(resolved)
            if latest <= 0:
                continue
            if (now_ts - latest) <= ttl:
                continue
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved, ignore_errors=True)
                else:
                    resolved.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                continue
    except Exception:
        return deleted
    return deleted


def _cleanup_old_outputs(now_ts: float) -> int:
    """OUTPUT_DIR 아래의 오래된 산출물(파일/폴더)을 TTL 기준으로 정리."""
    ttl = _run_ttl_seconds()
    deleted = 0
    root = OUTPUT_DIR.resolve()
    if not root.exists():
        return 0
    try:
        for item in root.iterdir():
            # 업로드 폴더/스테이징 폴더는 별도 정리
            if item.name in {"uploads", "_staging"}:
                continue
            try:
                resolved = item.resolve()
                resolved.relative_to(root)
            except Exception:
                continue
            latest = _latest_mtime(resolved)
            if latest <= 0:
                continue
            if (now_ts - latest) <= ttl:
                continue
            try:
                if resolved.is_dir():
                    shutil.rmtree(resolved, ignore_errors=True)
                else:
                    resolved.unlink(missing_ok=True)
                deleted += 1
            except Exception:
                continue
    except Exception:
        return deleted
    return deleted


@app.on_event("startup")
async def _start_cleanup_task() -> None:
    async def _loop():
        interval = _run_cleanup_interval_seconds()
        while True:
            try:
                _cleanup_expired_runs(time.time())
                _cleanup_old_outputs(time.time())
                _cleanup_old_uploads(time.time())
            except Exception:
                # cleanup should never crash the server
                pass
            await asyncio.sleep(interval)

    asyncio.create_task(_loop())


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _safe_filename(filename: str) -> str:
    name = Path(filename).name  # drop any path components
    # macOS는 한글이 NFD(자모 분해)로 올 수 있어 NFC로 정규화
    normalized = unicodedata.normalize("NFC", name)
    # keep alnum, dash, underscore, dot, 한글
    return re.sub(r"[^A-Za-z0-9._-가-힣ㄱ-ㅎㅏ-ㅣ]", "_", normalized)


def _sanitize_parts(rel_path: str) -> Path:
    """상대 경로의 각 파트를 안전하게 정제."""
    parts = []
    for p in Path(rel_path).parts:
        if p in ("", ".", ".."):
            continue
        # macOS는 한글이 NFD(자모 분해)로 올 수 있어 NFC로 정규화
        normalized = unicodedata.normalize("NFC", p)
        # 한글 허용: 완성형(가-힣) + 자모(ㄱ-ㅎ, ㅏ-ㅣ)
        parts.append(re.sub(r"[^A-Za-z0-9._-가-힣ㄱ-ㅎㅏ-ㅣ]", "_", normalized))
    return Path(*parts)


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    """ZipSlip 방지: zip 내부 경로가 extract_dir 밖으로 나가지 않게 검증 후 해제."""
    extract_dir = extract_dir.resolve()

    def _decode_zip_name(name: str, member: zipfile.ZipInfo) -> str:
        if not name:
            return name
        # zipfile이 이미 디코딩한 이름을 기반으로 여러 후보를 만들어 최적 선택
        candidates: list[str] = [name]

        # 기본 케이스: cp437로 재인코딩한 바이트를 utf-8/cp949로 복원 시도
        try:
            raw = name.encode("cp437")
            for enc in ("utf-8", "cp949"):
                try:
                    candidates.append(raw.decode(enc))
                except Exception:
                    pass
        except Exception:
            pass

        # 모지바케(예: Bä…)는 latin-1 재인코딩이 더 잘 복원되는 경우가 있음
        try:
            raw = name.encode("latin-1")
            for enc in ("utf-8", "cp949"):
                try:
                    candidates.append(raw.decode(enc))
                except Exception:
                    pass
        except Exception:
            pass

        def _score(s: str) -> int:
            hangul = sum(
                1
                for ch in s
                if ("\uAC00" <= ch <= "\uD7A3")  # 가-힣
                or ("\u1100" <= ch <= "\u11FF")  # 자모 (초성/중성/종성)
                or ("\u3130" <= ch <= "\u318F")  # 호환 자모
            )
            replacement = s.count("\ufffd")
            return hangul * 10 - replacement * 5

        best = max(candidates, key=_score)
        return unicodedata.normalize("NFC", best)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            # Normalize name and skip empty
            name = member.filename
            if not name:
                continue
            normalized_name = _decode_zip_name(name, member)
            dest = (extract_dir / normalized_name).resolve()
            try:
                dest.relative_to(extract_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"ZIP에 잘못된 경로가 포함되어 있습니다: {name}") from exc
        # 실제 추출 시도 (이때도 이름 정규화를 적용)
        for member in zf.infolist():
            name = member.filename
            if not name:
                continue
            normalized_name = _decode_zip_name(name, member)
            target = (extract_dir / normalized_name).resolve()
            try:
                target.relative_to(extract_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"ZIP에 잘못된 경로가 포함되어 있습니다: {name}") from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, "r") as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def _maybe_extract_zip_file(path: Path) -> Path:
    """path가 zip이면 해제 후 디렉터리 경로를 반환. 아니면 그대로 반환."""
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_dir = path.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            _safe_extract_zip(path, extract_dir)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"ZIP 해제 실패: {exc}") from exc
        finally:
            path.unlink(missing_ok=True)
        return extract_dir
    return path


def _maybe_extract_single_zip_in_dir(dir_path: Path) -> Path:
    """다운로드된 디렉터리에 zip 1개만 있고 다른 파일이 없으면 자동 해제."""
    if not dir_path.is_dir():
        return dir_path
    zips = list(dir_path.glob("*.zip"))
    other_files = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() != ".zip"]
    other_dirs = [p for p in dir_path.iterdir() if p.is_dir()]
    if len(zips) == 1 and not other_files and not other_dirs:
        return _maybe_extract_zip_file(zips[0])
    return dir_path


@app.post("/upload")
async def upload(request: Request) -> Dict[str, str]:
    """단일 파일 또는 디렉터리 업로드(webkitdirectory) 지원."""
    form = await request.form(max_files=5000)
    files = list(form.getlist("files"))
    if not files:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = UPLOAD_DIR / f"{timestamp}_upload"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 단일 zip만 온 경우: 기존 동작 유지
    if len(files) == 1 and files[0].filename.lower().endswith(".zip"):
        file = files[0]
        safe_name = _safe_filename(file.filename)
        dest = base_dir / safe_name
        try:
            with dest.open("wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"파일 저장 실패: {exc}") from exc

        extract_dir = dest.with_suffix("")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            _safe_extract_zip(dest, extract_dir)
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail=f"ZIP 해제 실패: {exc}") from exc
        finally:
            dest.unlink(missing_ok=True)
        return {"path": str(extract_dir)}

    # 다중 파일(또는 단일 일반 파일): 상대 경로를 보존해 저장
    for idx, file in enumerate(files):
        rel = file.filename or f"file_{idx}"
        rel_path = _sanitize_parts(rel)
        if rel_path.name == "":
            rel_path = Path(f"file_{idx}")
        dest = (base_dir / rel_path).resolve()

        # 디렉터리 이탈 방지
        try:
            dest.relative_to(base_dir.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"잘못된 경로: {rel}") from exc

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with dest.open("wb") as f:
                content = await file.read()
                f.write(content)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"파일 저장 실패: {exc}") from exc

    return {"path": str(base_dir)}


def _serialize_messages(messages: Any) -> List[Message]:
    normalized: List[Message] = []
    for m in messages or []:
        if isinstance(m, tuple) and len(m) >= 2:
            normalized.append(Message(role=str(m[0]), content=str(m[1])))
        elif hasattr(m, "role") and hasattr(m, "content"):
            normalized.append(Message(role=str(getattr(m, "role")), content=str(getattr(m, "content"))))
        else:
            normalized.append(Message(role="assistant", content=str(m)))
    return normalized


def _serialize_result(result: Dict[str, Any]) -> RunResponse:
    generation = result.get("generation")
    final_user_messages = result.get("final_user_messages")

    def _serialize_tool_calls(val: Any) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        if not val:
            return calls
        for item in val:
            if hasattr(item, "model_dump"):
                calls.append(item.model_dump())
            elif isinstance(item, dict):
                calls.append(dict(item))
            else:
                calls.append({"name": str(item)})
        return calls

    def _get(field: str) -> str | None:
        if hasattr(generation, field):
            return getattr(generation, field)
        if isinstance(generation, dict):
            return generation.get(field)
        return None

    return RunResponse(
        imports=_get("imports"),
        code=_get("code"),
        messages=_serialize_messages(final_user_messages) if final_user_messages else [],
        run_id=str(result.get("run_id") or "") or None,
        output_files=[Path(str(x)).name for x in (result.get("output_files") or [])],
        tool_calls=_serialize_tool_calls(result.get("planned_tools")),
    )


class PreviewResponse(BaseModel):
    filename: str
    columns: List[str]
    rows: List[Dict[str, Any]]


def _coerce_jsonable(v: Any) -> Any:
    # Keep preview payload safe/compact for the browser.
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    try:
        return str(v)
    except Exception:
        return repr(v)


def _read_preview_rows(path: Path, n: int) -> tuple[list[str], list[dict[str, Any]]]:
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(path, sep=sep, nrows=n, on_bad_lines="skip")
    elif suffix == ".parquet":
        df = pd.read_parquet(path).head(n)
    elif suffix == ".json":
        # try jsonl first
        try:
            df = pd.read_json(path, lines=True).head(n)
        except Exception:
            df = pd.read_json(path, lines=False).head(n)
    else:
        raise ValueError(f"Preview not supported for extension: {suffix}")

    cols = [str(c) for c in df.columns.tolist()]
    rows: list[dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        rows.append({str(k): _coerce_jsonable(v) for k, v in rec.items()})
    return cols, rows


@app.get("/downloads/{run_id}/{filename}")
def download_output(run_id: str, filename: str) -> FileResponse:
    run_id = (run_id or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id가 비어 있습니다.")
    info = _get_run_outputs(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail="알 수 없는 run_id 이거나 만료되었습니다.")
    created_at, files = info
    ttl = _run_ttl_seconds()
    if (time.time() - created_at) > ttl:
        # Best-effort cleanup and report expired.
        _cleanup_expired_runs(time.time())
        raise HTTPException(status_code=410, detail="결과 파일이 만료되어 삭제되었습니다.")

    safe_name = Path(filename).name
    if safe_name not in set(files):
        raise HTTPException(status_code=404, detail="해당 run_id에 존재하지 않는 파일입니다.")

    path = (OUTPUT_DIR / safe_name).resolve()
    try:
        path.relative_to(OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다(삭제되었을 수 있습니다).")
    return FileResponse(path, filename=safe_name, media_type="application/octet-stream")


@app.get("/downloads/{run_id}/{filename}/preview", response_model=PreviewResponse)
def preview_output(run_id: str, filename: str, n: int = 5) -> PreviewResponse:
    run_id = (run_id or "").strip()
    if not run_id:
        raise HTTPException(status_code=400, detail="run_id가 비어 있습니다.")
    if n < 1:
        n = 1
    if n > 50:
        n = 50

    info = _get_run_outputs(run_id)
    if info is None:
        raise HTTPException(status_code=404, detail="알 수 없는 run_id 이거나 만료되었습니다.")
    created_at, files = info
    ttl = _run_ttl_seconds()
    if (time.time() - created_at) > ttl:
        _cleanup_expired_runs(time.time())
        raise HTTPException(status_code=410, detail="결과 파일이 만료되어 삭제되었습니다.")

    safe_name = Path(filename).name
    if safe_name not in set(files):
        raise HTTPException(status_code=404, detail="해당 run_id에 존재하지 않는 파일입니다.")

    path = (OUTPUT_DIR / safe_name).resolve()
    try:
        path.relative_to(OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.") from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다(삭제되었을 수 있습니다).")

    try:
        columns, rows = _read_preview_rows(path, n=n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"미리보기 생성 실패: {exc}") from exc

    return PreviewResponse(filename=safe_name, columns=columns, rows=rows)


def _normalize_output_format(fmt: str | None) -> str | None:
    """프론트엔드 select 값(alias)을 백엔드 허용 포맷 문자열로 변환."""
    if not fmt:
        return None
    fmt = fmt.strip().lower()
    alias = {
        "csv": "csv",
        "parquet": "parquet",
        "feather": "feather",
        "arrow": "feather",
        "json": "json",
        "excel": "xlsx",
        "xlsx": "xlsx",
        "huggingface": "huggingface",
        "hf": "huggingface",
    }
    return alias.get(fmt, fmt)


def _get_s3_bucket() -> str:
    bucket = os.getenv("S3_BUCKET", "").strip()
    if not bucket:
        raise HTTPException(status_code=500, detail="S3_BUCKET 환경 변수가 설정되지 않았습니다.")
    return bucket


def _get_s3_client():
    """
    Create an S3 client that generates *regional* presigned URLs.

    If presigned URLs use the global endpoint (bucket.s3.amazonaws.com) for a non-us-east-1 bucket,
    browsers may hit redirects on preflight (OPTIONS) and fail CORS checks.
    """
    region = os.getenv("AWS_REGION", "").strip() or "eu-north-1"
    cfg = Config(signature_version="s3v4", s3={"addressing_style": "virtual"})
    # Force a regional endpoint to avoid redirects during browser preflight.
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=cfg,
    )




def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError("not an s3 uri")
    rest = uri[5:]
    bucket, _, key = rest.partition("/")
    if not bucket:
        raise ValueError("missing bucket")
    return bucket, key


def _download_s3_to_upload_dir(s3_uri: str) -> str:
    """Download an s3://... object or prefix into OUTPUT_DIR/uploads and return the local path."""
    bucket, key = _parse_s3_uri(s3_uri)
    s3 = _get_s3_client()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_dir = UPLOAD_DIR / f"{timestamp}_s3download"
    base_dir.mkdir(parents=True, exist_ok=True)

    prefix = key
    is_prefix = (prefix == "") or prefix.endswith("/")
    if not is_prefix:
        local_path = (base_dir / Path(prefix).name).resolve()
        s3.download_file(bucket, prefix, str(local_path))
        local = _maybe_extract_zip_file(local_path)
        return str(local)

    token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []) or []:
            k = obj.get("Key") or ""
            if not k or k.endswith("/"):
                continue
            rel = k[len(prefix) :] if k.startswith(prefix) else k
            rel_path = Path(rel)
            dest = (base_dir / rel_path).resolve()
            try:
                dest.relative_to(base_dir.resolve())
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"잘못된 S3 키 경로: {k}") from exc
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, k, str(dest))
        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
            continue
        break
    extracted = _maybe_extract_single_zip_in_dir(base_dir)
    return str(extracted)


def _maybe_download_s3_and_rewrite_question(question: str) -> str:
    """If question starts with an s3://... path, download it and replace with local path."""
    q = question.strip()
    if not q.startswith("s3://"):
        return question
    first, _, rest = q.partition(" ")
    local_path = _download_s3_to_upload_dir(first)
    return (local_path + (" " + rest if rest else "")).strip()


class CreateUploadSessionResponse(BaseModel):
    bucket: str
    upload_id: str
    prefix: str


class PresignFile(BaseModel):
    path: str = Field(..., description="상대 경로(폴더 업로드 시 webkitRelativePath)")
    content_type: str | None = None


class PresignRequest(BaseModel):
    upload_id: str
    files: List[PresignFile]


class PresignItem(BaseModel):
    path: str
    key: str
    url: str


class PresignResponse(BaseModel):
    bucket: str
    upload_id: str
    prefix: str
    items: List[PresignItem]


@app.post("/s3/create_upload_session", response_model=CreateUploadSessionResponse)
def create_upload_session() -> CreateUploadSessionResponse:
    bucket = _get_s3_bucket()
    upload_id = uuid4().hex
    prefix = f"uploads/{upload_id}/"
    return CreateUploadSessionResponse(bucket=bucket, upload_id=upload_id, prefix=prefix)


@app.post("/s3/presign_put", response_model=PresignResponse)
def presign_put(req: PresignRequest) -> PresignResponse:
    bucket = _get_s3_bucket()
    s3 = _get_s3_client()
    prefix = f"uploads/{req.upload_id}/"

    items: List[PresignItem] = []
    for f in req.files:
        rel = str(Path(f.path).as_posix()).lstrip("/")
        if ".." in Path(rel).parts:
            raise HTTPException(status_code=400, detail=f"잘못된 경로: {f.path}")
        key = prefix + rel
        params: Dict[str, Any] = {"Bucket": bucket, "Key": key}
        if f.content_type:
            params["ContentType"] = f.content_type
        url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params=params,
            ExpiresIn=900,
        )
        items.append(PresignItem(path=f.path, key=key, url=url))

    return PresignResponse(bucket=bucket, upload_id=req.upload_id, prefix=prefix, items=items)


@app.post("/run", response_model=RunResponse)
def run(body: RunRequest) -> RunResponse:
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question 필드는 비어 있을 수 없습니다.")

    rewritten_question = _maybe_download_s3_and_rewrite_question(body.question)
    output_formats = _normalize_output_format(body.output_format)
    llm_model = _normalize_model(body.llm_model, ALLOWED_LLM_MODELS, DEFAULT_LLM_MODEL, "llm_model")
    coder_model = _normalize_model(body.coder_model, ALLOWED_CODER_MODELS, DEFAULT_CODER_MODEL, "coder_model")
    try:
        result = run_request(
            request=rewritten_question,
            max_iterations=body.max_iterations,
            output_formats=output_formats,
            llm_model=llm_model,
            coder_model=coder_model,
        )
    except SystemExit as exc:
        # Generated code may accidentally call exit(); do not crash the API process.
        raise HTTPException(status_code=400, detail=f"생성된 코드가 exit()를 호출했습니다: {getattr(exc, 'code', exc)}") from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("그래프 실행 실패")
        if os.getenv("DEBUG_TRACEBACK", "").strip():
            tb = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"그래프 실행 실패: {exc}\n{tb}") from exc
        raise HTTPException(status_code=500, detail=f"그래프 실행 실패: {exc}") from exc

    payload = _serialize_result(result)
    if payload.run_id and payload.output_files:
        _register_run_outputs(payload.run_id, payload.output_files)
    return payload


@app.post("/run_stream")
def run_stream(body: RunRequest) -> StreamingResponse:
    """Stream progress as NDJSON. Emits {type:'progress', iterations:n} and a final {type:'final', data:RunResponse}."""

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question 필드는 비어 있을 수 없습니다.")

    output_formats = _normalize_output_format(body.output_format)
    rewritten_question = _maybe_download_s3_and_rewrite_question(body.question)
    llm_model = _normalize_model(body.llm_model, ALLOWED_LLM_MODELS, DEFAULT_LLM_MODEL, "llm_model")
    coder_model = _normalize_model(body.coder_model, ALLOWED_CODER_MODELS, DEFAULT_CODER_MODEL, "coder_model")

    def _iter_ndjson():
        graph = build_graph(llm_model=llm_model, coder_model=coder_model, max_iterations=body.max_iterations)
        run_id = uuid4().hex
        initial_state: Dict[str, Any] = {
            "run_id": run_id,
            "messages": [("user", rewritten_question)],
            "iterations": 0,
            "error": "",
            "context": "",
            "generation": None,
            "phase": None,
            "user_request": rewritten_question,
            "output_formats": output_formats,
            "final_user_messages": None,
            "trace": [],
            "llm_model": llm_model,
            "coder_model": coder_model,
        }

        last_iterations = 0
        last_stage: Optional[str] = None
        last_state: Optional[Dict[str, Any]] = None
        last_tool_calls_key: Optional[str] = None
        reflect_counter = 0
        try:
            yield json.dumps({"type": "stage", "stage": "queued", "detail": "요청 접수"}, ensure_ascii=False) + "\n"
            node_to_stage = {
                "inspect_input_node": ("inspecting", "입력 검사 중"),
                "run_sample_and_summarize": ("sampling", "데이터 샘플링 중"),
                "run_image_manifest": ("sampling", "이미지 목록 생성 중"),
                "build_context": ("context", "컨텍스트 구성 중"),
                "chatbot": ("analyzing", "요구사항 정리 중"),
                "run_planned_tools": ("tooling", "툴 조사/전수 스캔 중"),
                "generate": ("generating", "스크립트 생성 중"),
                "code_check": ("executing", "스크립트 실행 중"),
                "validate": ("validating", "요구사항 검증 중"),
                "reflect": ("refactoring", "리팩트 중"),
                "friendly_error": ("finalizing", "오류 요약 중"),
                "final_friendly_error": ("finalizing", "오류 요약 중"),
            }

            # Use task start events to update stage immediately when a node begins executing.
            for mode, data in graph.stream(initial_state, stream_mode=["tasks", "values"]):
                if mode == "tasks" and isinstance(data, dict):
                    # Task start events include 'input' and 'triggers'. Finish events include 'result'.
                    name = data.get("name")
                    is_start = "input" in data and "triggers" in data
                    if is_start and isinstance(name, str) and name in node_to_stage:
                        stage, detail = node_to_stage[name]
                        if name == "reflect":
                            reflect_counter += 1
                            # reflect 진입 원인을 stage detail로 표시 (스크립트 오류 vs 검증/요구사항 실패)
                            inp = data.get("input")
                            prev_phase = ""
                            last_msg = ""
                            if isinstance(inp, dict):
                                prev_phase = str(inp.get("phase") or "")
                                msgs = inp.get("messages") or []
                                if isinstance(msgs, list) and msgs:
                                    m = msgs[-1]
                                    if isinstance(m, tuple) and len(m) >= 2:
                                        last_msg = str(m[1])
                                    elif hasattr(m, "content"):
                                        last_msg = str(getattr(m, "content", ""))
                                    else:
                                        last_msg = str(m)

                            lower_msg = last_msg.lower()
                            if not prev_phase and last_stage in {"executing", "validating"}:
                                prev_phase = last_stage

                            if prev_phase == "executing" or "failed during execution" in lower_msg:
                                detail = "스크립트 오류 수정"
                            elif prev_phase == "validating" or "validation failed" in lower_msg or "requirements" in lower_msg:
                                detail = "요구사항 검증 오류 수정"
                            else:
                                detail = "리팩트 중"
                            detail = f"리팩트 #{reflect_counter}: {detail}" if detail else f"리팩트 #{reflect_counter}"
                        if stage != last_stage:
                            last_stage = stage
                            yield json.dumps({"type": "stage", "stage": stage, "detail": detail}, ensure_ascii=False) + "\n"
                elif mode == "values" and isinstance(data, dict):
                    last_state = data
                    it = data.get("iterations", 0)
                    if isinstance(it, int) and it != last_iterations:
                        last_iterations = it
                        yield json.dumps({"type": "progress", "iterations": it}, ensure_ascii=False) + "\n"
                    planned_tools = data.get("planned_tools")
                    if isinstance(planned_tools, list):
                        serialized = []
                        for item in planned_tools:
                            if hasattr(item, "model_dump"):
                                serialized.append(item.model_dump())
                            elif isinstance(item, dict):
                                serialized.append(dict(item))
                            else:
                                serialized.append({"name": str(item)})
                        key = json.dumps(serialized, ensure_ascii=False, sort_keys=True, default=str)
                        if key != last_tool_calls_key:
                            last_tool_calls_key = key
                            yield json.dumps({"type": "tool_calls", "tool_calls": serialized}, ensure_ascii=False) + "\n"

            yield json.dumps({"type": "stage", "stage": "finalizing", "detail": "결과 정리 중"}, ensure_ascii=False) + "\n"
            final_state = last_state or initial_state
            if isinstance(final_state, dict):
                trace_name = write_internal_trace_markdown(final_state)
                if trace_name:
                    files = final_state.get("output_files") or []
                    if not isinstance(files, list):
                        files = []
                    if trace_name not in files:
                        files.append(trace_name)
                    final_state["output_files"] = files

            payload_model = _serialize_result(final_state)
            if payload_model.run_id and payload_model.output_files:
                _register_run_outputs(payload_model.run_id, payload_model.output_files)
            payload = payload_model.model_dump()
            yield json.dumps({"type": "final", "data": payload}, ensure_ascii=False) + "\n"
        except SystemExit as exc:
            yield json.dumps(
                {"type": "error", "detail": f"생성된 코드가 exit()를 호출했습니다: {getattr(exc, 'code', exc)}"},
                ensure_ascii=False,
            ) + "\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("그래프 실행 실패 (stream)")
            if os.getenv("DEBUG_TRACEBACK", "").strip():
                tb = traceback.format_exc()
                detail = f"그래프 실행 실패: {exc}\n{tb}"
            else:
                detail = f"그래프 실행 실패: {exc}"
            yield json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n"

    return StreamingResponse(_iter_ndjson(), media_type="application/x-ndjson")


# 편의를 위한 로컬 실행 진입점
def _main() -> None:
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    _main()
