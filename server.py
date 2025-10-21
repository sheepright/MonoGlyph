#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# server.py — main.py(신규)와 맞춘 API 서버 (큐/로그 스트림 확장)
# - POST /run
# - GET  /run/stream?prompt=...
# - GET  /logs/{job_id}/stream
# - GET  /status/{job_id}

import sys
import os
import re
import json
import logging
import threading
import subprocess
import shutil
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable, Iterator

from fastapi import FastAPI, HTTPException, Request, Query, Path as FPath
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ──────────────────────────────────────────────────────────────────────────────
# 로깅 설정 (루트 로거 + 모듈 로거)
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("font-pipeline.server")

class _LoggerWriter:
    """print를 로거로 연결하기 위한 writer"""
    def __init__(self, log_fn: Callable[[str], None]):
        self.log_fn = log_fn
        self._buf = ""

    def write(self, msg: str):
        self._buf += msg
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line:
                self.log_fn(line)

    def flush(self):
        if self._buf:
            self.log_fn(self._buf.rstrip("\r\n"))
            self._buf = ""

# print → 로거
sys.stdout = _LoggerWriter(logger.info)
sys.stderr = _LoggerWriter(logger.error)

app = FastAPI(title="Font Pipeline (main.py orchestrator)", version="2.1.0")

# 스크립트/출력 설정
MAIN_SCRIPT = "main.py"
DEFAULT_OUT_DIR = Path("output")  # /static으로 노출되는 루트
LOG_DIR = DEFAULT_OUT_DIR / "logs"
DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# CORS — 개발용: localhost:3000 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static으로 output 디렉터리 공개 → 링크 제공
app.mount("/static", StaticFiles(directory=str(DEFAULT_OUT_DIR), html=False), name="static")

# ──────────────────────────────────────────────────────────────────────────────
# 대기열(FIFO) 구현
#  - 동시에 1개만 실행
#  - 각 요청은 ticket을 받아 자신의 차례가 올 때까지 대기
# ──────────────────────────────────────────────────────────────────────────────
_queue_cv = threading.Condition()
_queue: List[str] = []  # ticket 목록 (job_id)
_running_job_id: Optional[str] = None  # 현재 실행 중인 job_id

def enqueue_and_run(job_id: str):
    """컨텍스트 매니저: 자신의 차례가 될 때까지 대기 후 실행, 종료 시 다음 대기자 깨움"""
    class _Ctx:
        def __enter__(self2):
            global _running_job_id
            with _queue_cv:
                _queue.append(job_id)
                # 대기 순서 안내용 로그
                logger.info("[Queue] Job %s enqueued. position=%d", job_id, _queue.index(job_id))
                # 자신의 차례가 될 때까지 대기
                while _queue[0] != job_id or _running_job_id is not None:
                    _queue_cv.wait(timeout=0.5)
                _running_job_id = job_id
                logger.info("[Queue] Job %s started.", job_id)

        def __exit__(self2, exc_type, exc, tb):
            global _running_job_id
            with _queue_cv:
                # 본인 제거 후 다음 깨움
                if _queue and _queue[0] == job_id:
                    _queue.pop(0)
                _running_job_id = None
                _queue_cv.notify_all()
                logger.info("[Queue] Job %s finished. next=%s", job_id, (_queue[0] if _queue else None))
    return _Ctx()

def current_queue_position(job_id: str) -> int:
    with _queue_cv:
        try:
            return _queue.index(job_id)
        except ValueError:
            return -1

def queue_snapshot() -> Dict[str, Any]:
    with _queue_cv:
        return {
            "running": _running_job_id,
            "queue": list(_queue),
        }

# ──────────────────────────────────────────────────────────────────────────────
# 유틸: ASCII-only 파일명 보장
# ──────────────────────────────────────────────────────────────────────────────
def ascii_sanitize(s: str, fallback: str = "") -> str:
    if not isinstance(s, str):
        s = str(s)
    normalized = unicodedata.normalize("NFKD", s)
    ascii_only = normalized.encode("ascii", "ignore").decode()
    ascii_only = ascii_only.strip()
    return ascii_only if ascii_only else fallback

def ensure_ascii_copy(ttf_path: Path) -> Path:
    name = ttf_path.name
    if all(ord(c) < 128 for c in name):
        return ttf_path  # 이미 ASCII-only

    parent = ttf_path.parent
    stem_ascii = ascii_sanitize(ttf_path.stem, "output")
    ext_ascii = ".ttf"

    candidate = parent / f"{stem_ascii}{ext_ascii}"
    if candidate.exists():
        i = 1
        while True:
            alt = parent / f"{stem_ascii}-{i}{ext_ascii}"
            if not alt.exists():
                candidate = alt
                break
            i += 1

    try:
        shutil.copy2(ttf_path, candidate)
        logger.info("Created ASCII filename copy: %s -> %s", ttf_path.name, candidate.name)
        return candidate
    except Exception as e:
        logger.error("Failed to create ASCII copy: %s", e)
        raise HTTPException(status_code=500, detail={
            "ok": False,
            "message": "Failed to prepare ASCII-only download file."
        })

# ──────────────────────────────────────────────────────────────────────────────
# stdout에서 ttf 경로 파싱 + 폴백 탐색
# (신규 main.py의 최종 출력 포맷 지원)
# ──────────────────────────────────────────────────────────────────────────────
# 예시:
# [OK] 최종 TTF:     /abs/path/to/work-YYYYMMDD_HHMMSS/output/MonoGlyph.ttf
TTF_PATH_RE = re.compile(
    r"""(?ix)
        (?:^\s*\[OK\]\s*최종\s*TTF:\s*)
        (?P<path>[^ \n\r\t]+\.ttf)\s*$
    """
)

def _parse_ttf_from_stdout(stdout: str) -> Optional[Path]:
    matches = list(TTF_PATH_RE.finditer(stdout or ""))
    if not matches:
        return None
    p = Path(matches[-1].group("path"))
    return p if p.exists() else None

def _fallback_latest_ttf(search_root: Path = Path(".")) -> Optional[Path]:
    candidates: List[Path] = []
    for p in search_root.glob("work-*"):
        out_dir = p / "output"
        candidates += list(out_dir.rglob("*.ttf"))
    candidates += list(DEFAULT_OUT_DIR.rglob("*.ttf"))
    if candidates:
        return max(candidates, key=lambda x: x.stat().st_mtime)
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 공용: 서브프로세스 실행 + 로그 수집 (+ per-job 파일 로깅)
# ──────────────────────────────────────────────────────────────────────────────
def _stream_reader(pipe, log_fn: Callable[[str], None], buf: List[str], also_file: Optional[logging.Logger]=None, level=logging.INFO):
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            buf.append(line)
            msg = line.rstrip("\r\n")
            log_fn(msg)
            if also_file:
                also_file.log(level, msg)
    finally:
        try:
            pipe.close()
        except Exception:
            pass

def _attach_job_file_logger(job_id: str) -> logging.Logger:
    """잡 전용 파일 핸들러 부착(반환 로거로만 씀). 사용 후 반드시 removeHandler 필요."""
    job_logger = logging.getLogger(f"font-pipeline.job.{job_id}")
    job_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fh = logging.FileHandler(LOG_DIR / f"{job_id}.log", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    job_logger.handlers = [fh]  # 덮어쓰기 보장(동시실행이 1개이므로 안전)
    job_logger.propagate = False
    return job_logger

def _detach_job_file_logger(job_logger: logging.Logger):
    for h in list(job_logger.handlers):
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        job_logger.removeHandler(h)

def run_and_log(
    cmd: List[str],
    job_id: str,
    timeout: Optional[int] = None,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str]:
    job_logger = _attach_job_file_logger(job_id)
    job_logger.info("[Queue] job_id=%s 시작", job_id)
    job_logger.info("[Cmd] %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_buf: List[str] = []
    stderr_buf: List[str] = []

    t_out = threading.Thread(target=_stream_reader, args=(proc.stdout, logger.info, stdout_buf, job_logger, logging.INFO), daemon=True)
    t_err = threading.Thread(target=_stream_reader, args=(proc.stderr, logger.error, stderr_buf, job_logger, logging.ERROR), daemon=True)
    t_out.start(); t_err.start()

    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("[Time] 서브프로세스 타임아웃 발생. 프로세스를 종료합니다.")
        job_logger.error("[Time] 서브프로세스 타임아웃 발생. 프로세스를 종료합니다.")
        proc.kill()
        rc = proc.wait()
    finally:
        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

    stdout_text = "".join(stdout_buf)
    stderr_text = "".join(stderr_buf)

    logger.info("[Stop] 종료 코드: %s", rc)
    job_logger.info("[Stop] 종료 코드: %s", rc)

    _detach_job_file_logger(job_logger)
    return rc, stdout_text, stderr_text

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic 모델
# ──────────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel, Field

class RunBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000, description="main.py에 전달할 프롬프트")

# ──────────────────────────────────────────────────────────────────────────────
# 진행도 보조: work 디렉터리 추정/프로빙
# ──────────────────────────────────────────────────────────────────────────────
def _sse_event(data: Dict[str, Any], event: str = "message") -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

def _find_newest_work_dir(since_ts: float) -> Optional[Path]:
    roots = [p for p in Path(".").glob("work-*") if p.is_dir()]
    if not roots:
        return None
    recent = [p for p in roots if p.stat().st_mtime >= since_ts - 1.0]
    if not recent:
        recent = roots
    return max(recent, key=lambda x: x.stat().st_mtime) if recent else None

def _progress_probe(work_dir: Path) -> int:
    code = 0
    if not work_dir or not work_dir.exists():
        return code
    d_imgs   = work_dir / "font_imgs"
    d_data   = work_dir / "font_data"
    d_result = work_dir / "result"
    d_out    = work_dir / "output"
    if d_imgs.exists() and any(d_imgs.rglob("*.png")):
        code = max(code, 250)   # gpt_api 이미지 생성
    if d_data.exists() and any(d_data.rglob("*.png")):
        code = max(code, 300)   # 전처리 완료
    if d_result.exists() and any(d_result.rglob("*.png")):
        code = max(code, 400)   # MX 추론 결과 생성
    if d_out.exists() and any(d_out.rglob("*.ttf")):
        code = max(code, 750)   # TTF 생성 진행/완료
    return code

def _progress_label(code: int) -> Optional[str]:
    if code >= 750: return "폰트 파일 생성중.."
    if code >= 400: return "전체 글자 생성중.."
    if code >= 300: return "생성된 글자 분석중.."
    if code >= 250: return "스타일 분석 및 생성중.."
    return None

# ──────────────────────────────────────────────────────────────────────────────
# 기존 최종 JSON 응답 엔드포인트 (큐 + 로그 파일 기록 추가)
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/run")
def run_endpoint(body: RunBody, request: Request):
    job_id = uuid.uuid4().hex
    cmd = [sys.executable, "-u", MAIN_SCRIPT, "-p", body.prompt]
    timeout_sec = int(os.getenv("PIPELINE_TIMEOUT", "1800"))  # 30분
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    logger.info("=== /run 요청 수신 === job_id=%s", job_id)
    logger.info("입력 프롬프트 길이: %d", len(body.prompt))

    # 큐 진입(선점 방지)
    with enqueue_and_run(job_id):
        rc, out, err = run_and_log(cmd, job_id=job_id, timeout=timeout_sec, env=env)

    ok = (rc == 0)
    ran = " ".join(cmd)

    # stdout → 최종 TTF 경로 파싱 → 없으면 폴백
    ttf_path = _parse_ttf_from_stdout(out)
    if (ttf_path is None) or (not ttf_path.exists()):
        ttf_path = _fallback_latest_ttf()

    if ok and ttf_path and ttf_path.exists():
        dist_path = ensure_ascii_copy(ttf_path)

        # /static 아래 상대경로 확보
        try:
            rel = dist_path.resolve().relative_to(DEFAULT_OUT_DIR.resolve())
        except ValueError:
            DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
            stem_safe = ascii_sanitize(dist_path.stem, "output")
            target = (DEFAULT_OUT_DIR / f"{stem_safe}.ttf").resolve()
            if target.exists():
                i = 1
                while True:
                    alt = (DEFAULT_OUT_DIR / f"{stem_safe}-{i}.ttf").resolve()
                    if not alt.exists():
                        target = alt
                        break
                    i += 1
            try:
                shutil.copy2(dist_path, target)
                logger.info("Copied TTF into output/: %s -> %s", dist_path, target)
                dist_path = target
                rel = dist_path.resolve().relative_to(DEFAULT_OUT_DIR.resolve())
            except Exception as e:
                logger.error("Failed to place TTF under /static directory: %s", e)
                raise HTTPException(status_code=500, detail={
                    "ok": False,
                    "message": "Failed to place TTF under /static directory."
                })

        download_url = request.url_for("static", path=rel.as_posix())
        safe_filename = ascii_sanitize(dist_path.name, "output.ttf")
        safe_download_url = ascii_sanitize(str(download_url), str(download_url))
        safe_ran = ascii_sanitize(ran, "python main.py -p <prompt>")

        return JSONResponse(content={
            "ok": True,
            "job_id": job_id,
            "message": "TTF file successfully generated. Download link provided.",
            "filename": safe_filename,
            "download_url": safe_download_url,
            "ran": safe_ran,
        })

    result: Dict[str, Any] = {
        "ok": ok,
        "job_id": job_id,
        "returncode": rc,
        "stdout": out,
        "stderr": err,
        "ran": ascii_sanitize(ran, "python main.py -p <prompt>"),
        "message": "TTF not generated, returning logs only." if ok else "Pipeline failed",
    }

    if not ok:
        logger.error("파이프라인 실패 (코드 %s). 로그를 JSON으로 반환합니다.", rc)
        raise HTTPException(status_code=500, detail=result)

    logger.warning("파이프라인 성공했지만 TTF 없음. 로그를 JSON으로 반환합니다.")
    return JSONResponse(content=result)

# ──────────────────────────────────────────────────────────────────────────────
# 신규: SSE 실시간 진행 + 로그 스트림 (큐 포함)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/run/stream")
def run_stream(prompt: str = Query(..., min_length=1, max_length=4000), request: Request = None):
    """
    GET /run/stream?prompt=...
    - 대기열 진입 → queue_position 이벤트 송신
    - 자신의 차례 진입 시 queue_enter 이벤트 송신
    - 파일시스템 변화 기반 진행(progress) 이벤트 송신
    - 표준출력/에러 및 진행 메시지를 로그로 기록하고, 로그 신규 줄을 log 이벤트로 송신
    - 완료 시 done 이벤트 (download_url 포함), 실패 시 error 이벤트
    """
    job_id = uuid.uuid4().hex
    timeout_sec = int(os.getenv("PIPELINE_TIMEOUT", "1800"))  # 30분
    cmd = [sys.executable, "-u", MAIN_SCRIPT, "-p", prompt]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def generator() -> Iterator[str]:
        logger.info("=== /run/stream 시작 === job_id=%s", job_id)
        logger.info("입력 프롬프트 길이: %d", len(prompt))

        # 초기: 대기열 등록 & 현재 순번 안내
        with _queue_cv:
            _queue.append(job_id)
            pos = _queue.index(job_id)
        yield _sse_event({"type": "queue_position", "job_id": job_id, "position": pos}, event="queue")

        # 자신의 차례가 될 때까지 대기(반복적으로 포지션 업데이트 전송)
        while True:
            with _queue_cv:
                pos = _queue.index(job_id)
                running = (_running_job_id is not None)
                can_enter = (_queue[0] == job_id and _running_job_id is None)
            if can_enter:
                yield _sse_event({"type": "queue_enter", "job_id": job_id, "position": 0}, event="queue")
                break
            else:
                # 주기적 포지션 통지(1초)
                yield _sse_event({"type": "queue_wait", "job_id": job_id, "position": pos, "running": running}, event="queue")
                time.sleep(1.0)

        # 이제 큐의 실행 슬롯을 획득
        with enqueue_and_run(job_id):
            # 스트리밍 시작 알림
            yield _sse_event({"type": "progress", "code": 0, "message": "대기 중…", "job_id": job_id}, event="progress")

            start_ts = time.time()
            job_logger = _attach_job_file_logger(job_id)
            job_logger.info("[Queue] queue_entered position=0 job_id=%s", job_id)
            job_logger.info("[Cmd] %s", " ".join(cmd))

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            stderr_buf: List[str] = []
            stdout_buf: List[str] = []

            # 로그 파일 테일링을 위해 위치 추적
            log_path = LOG_DIR / f"{job_id}.log"
            # 표준에러 별도 수집
            def _stderr_thread():
                try:
                    assert proc.stderr is not None
                    for line in iter(proc.stderr.readline, ""):
                        if not line:
                            break
                        stderr_buf.append(line)
                        msg = line.rstrip()
                        logger.error(msg)
                        job_logger.error(msg)
                except Exception:
                    pass

            th_err = threading.Thread(target=_stderr_thread, daemon=True)
            th_err.start()

            last_code_sent = 0
            guessed_work: Optional[Path] = None
            last_fs_probe = 0.0
            last_log_size = 0

            try:
                assert proc.stdout is not None
                while True:
                    # stdout 비동기 수집(라인 단위)
                    line = proc.stdout.readline()
                    if line:
                        stdout_buf.append(line)
                        msg = line.rstrip()
                        logger.info(msg)
                        job_logger.info(msg)

                    now = time.time()

                    # 0) 로그 파일의 신규 줄을 log 이벤트로 푸시 (0.5s)
                    if now - last_fs_probe >= 0.5:
                        # 파일시스템 프로브도 같이 수행
                        last_fs_probe = now

                        # 로그 파일 증분 전송
                        try:
                            if log_path.exists():
                                with open(log_path, "r", encoding="utf-8") as f:
                                    f.seek(last_log_size)
                                    chunk = f.read()
                                    last_log_size = f.tell()
                                if chunk:
                                    # 여러 줄을 통째로 보내되, 프론트에서 줄 단위 split 가능
                                    yield _sse_event(
                                        {"type": "log", "job_id": job_id, "append": chunk},
                                        event="log",
                                    )
                        except Exception as e:
                            logger.warning("log tail error: %s", e)

                        # work-* 추정
                        if guessed_work is None:
                            guessed_work = _find_newest_work_dir(start_ts)

                        # 진행 코드 산출 + 변화 시 이벤트 전송
                        code = _progress_probe(guessed_work) if guessed_work else 0
                        label = None
                        if code >= 750 and last_code_sent < 750:
                            label = _progress_label(750)
                        elif code >= 400 and last_code_sent < 400:
                            label = _progress_label(400)
                        elif code >= 300 and last_code_sent < 300:
                            label = _progress_label(300)
                        elif code >= 250 and last_code_sent < 250:
                            label = _progress_label(250)

                        if label:
                            last_code_sent = code
                            evt = {"type": "progress", "job_id": job_id, "code": code, "message": label}
                            job_logger.info("[Progress] %s", json.dumps(evt, ensure_ascii=False))
                            yield _sse_event(evt, event="progress")

                    # 프로세스 종료 확인
                    if (not line) and (proc.poll() is not None):
                        break

                rc = proc.wait(timeout=timeout_sec)

            except subprocess.TimeoutExpired:
                logger.error("[Time] 서브프로세스 타임아웃. kill()")
                job_logger.error("[Time] 서브프로세스 타임아웃. kill()")
                proc.kill()
                rc = proc.wait()
            except GeneratorExit:
                try:
                    proc.kill()
                except Exception:
                    pass
                _detach_job_file_logger(job_logger)
                raise
            except Exception as e:
                logger.exception("SSE 스트리밍 중 예외: %s", e)
                job_logger.exception("SSE 스트리밍 중 예외: %s", e)
                try:
                    proc.kill()
                except Exception:
                    pass
                _detach_job_file_logger(job_logger)
                yield _sse_event({"type": "error", "job_id": job_id, "message": f"Internal error: {e}"}, event="error")
                return

            # 완료 처리
            out_all = "".join(stdout_buf)
            ok = (rc == 0)

            if ok:
                # stdout → ttf 경로 파싱 → 폴백
                ttf_path = _parse_ttf_from_stdout(out_all) or _fallback_latest_ttf()
                if ttf_path and ttf_path.exists():
                    dist_path = ensure_ascii_copy(ttf_path)
                    try:
                        rel = dist_path.resolve().relative_to(DEFAULT_OUT_DIR.resolve())
                    except ValueError:
                        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
                        stem_safe = ascii_sanitize(dist_path.stem, "output")
                        target = (DEFAULT_OUT_DIR / f"{stem_safe}.ttf").resolve()

                        if target.exists():
                            i = 1
                            while True:
                                alt = (DEFAULT_OUT_DIR / f"{stem_safe}-{i}.ttf").resolve()
                                if not alt.exists():
                                    target = alt
                                    break
                                i += 1

                        try:
                            shutil.copy2(dist_path, target)
                            logger.info("Copied TTF into output/: %s -> %s", dist_path, target)
                            dist_path = target
                            rel = dist_path.resolve().relative_to(DEFAULT_OUT_DIR.resolve())
                        except Exception as e:
                            logger.error("Failed to place TTF under /static directory: %s", e)
                            job_logger.error("Failed to place TTF under /static directory: %s", e)
                            _detach_job_file_logger(job_logger)
                            yield _sse_event(
                                {"type": "error", "job_id": job_id, "message": "Failed to place TTF under /static directory."},
                                event="error",
                            )
                            return

                    download_url = request.url_for("static", path=rel.as_posix()) if request else str(rel.as_posix())
                    safe_download_url = ascii_sanitize(str(download_url), str(download_url))
                    safe_filename = ascii_sanitize(dist_path.name, "output.ttf")

                    if last_code_sent < 800:
                        evt = {"type": "progress", "job_id": job_id, "code": 800, "message": "미리보기 렌더링"}
                        job_logger.info("[Progress] %s", json.dumps(evt, ensure_ascii=False))
                        yield _sse_event(evt, event="progress")

                    job_logger.info("[Done] filename=%s url=%s", safe_filename, safe_download_url)
                    _detach_job_file_logger(job_logger)

                    yield _sse_event(
                        {
                            "type": "done",
                            "job_id": job_id,
                            "code": 1000,
                            "message": "TTF 파일 생성완료!",
                            "download_url": safe_download_url,
                            "filename": safe_filename,
                        },
                        event="done",
                    )
                    return
                else:
                    _detach_job_file_logger(job_logger)
                    yield _sse_event(
                        {"type": "error", "job_id": job_id, "message": "Pipeline finished but TTF not found."},
                        event="error",
                    )
                    return
            else:
                stderr_tail = "".join(stderr_buf[-50:]) if stderr_buf else ""
                _detach_job_file_logger(job_logger)
                yield _sse_event(
                    {
                        "type": "error",
                        "job_id": job_id,
                        "message": "Pipeline failed",
                        "returncode": rc,
                        "stderr_tail": stderr_tail,
                    },
                    event="error",
                )
                return

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Job-Id": job_id,
    }
    return StreamingResponse(generator(), media_type="text/event-stream", headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# 추가: 로그 전용 SSE 스트림 (프론트가 로그만 보고 싶을 때)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/logs/{job_id}/stream")
def logs_stream(job_id: str = FPath(..., min_length=10, max_length=64)):
    """
    GET /logs/{job_id}/stream
    - 해당 잡의 로그 파일을 테일링하여 log 이벤트로 전송
    - 잡이 끝났더라도 파일이 존재하면 그 시점 이후 append를 계속 감시(타임아웃 없음)
    """
    log_path = LOG_DIR / f"{job_id}.log"

    def gen() -> Iterator[str]:
        yield _sse_event({"type": "log_open", "job_id": job_id}, event="log")
        last_size = 0
        while True:
            try:
                if log_path.exists():
                    with open(log_path, "r", encoding="utf-8") as f:
                        f.seek(last_size)
                        chunk = f.read()
                        last_size = f.tell()
                    if chunk:
                        yield _sse_event({"type": "log", "job_id": job_id, "append": chunk}, event="log")
                time.sleep(0.5)
            except GeneratorExit:
                break
            except Exception as e:
                yield _sse_event({"type": "error", "message": f"log stream error: {e}"}, event="error")
                time.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# 추가: 상태 폴링 엔드포인트 (대기열/진행/다운로드 링크 추정)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/status/{job_id}")
def status(job_id: str, request: Request):
    """대기열 정보 + 최신 work 디렉토리 기반 진행도 + 다운로드 가능 여부를 반환"""
    snap = queue_snapshot()
    # 큐 상태
    try:
        position = snap["queue"].index(job_id)
    except ValueError:
        position = -1
    running = (snap["running"] == job_id)

    # 진행도 추정
    # 최근 work-* 로만 추정(정확성이 SSE보다 떨어질 수 있음)
    work = _find_newest_work_dir(time.time() - 3600)  # 최근 1시간 범위
    code = _progress_probe(work) if work else 0
    label = _progress_label(code)

    # 최신 TTF 유무
    ttf = _fallback_latest_ttf()
    download_url = None
    filename = None
    if ttf and ttf.exists():
        dist_path = ensure_ascii_copy(ttf)
        try:
            rel = dist_path.resolve().relative_to(DEFAULT_OUT_DIR.resolve())
        except ValueError:
            rel = None
        if rel:
            download_url = request.url_for("static", path=rel.as_posix())
            filename = dist_path.name

    return {
        "queue": {"running": snap["running"], "position": position, "in_queue": position >= 0, "is_running": running},
        "progress": {"code": code, "label": label},
        "download": {"available": bool(download_url), "filename": filename, "url": download_url},
    }
