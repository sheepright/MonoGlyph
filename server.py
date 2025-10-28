#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import subprocess
import shutil
import time
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="MonoGlyph Font Generator", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 메인 스크립트
MAIN_SCRIPT = "main.py"


# ===== 요청 모델 =====
class GenerateRequest(BaseModel):
    prompt: str


# ===== SSE 이벤트 생성 =====
def sse_event(data: Dict[str, Any], event: str = "message") -> str:
    """SSE 형식의 이벤트 문자열 생성"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


# ===== work 디렉토리에서 최신 TTF 찾기 =====
def find_latest_ttf(work_dir: Path) -> Optional[Path]:
    """work 디렉토리 내에서 TTF 파일 찾기"""
    output_dir = work_dir / "output"
    if output_dir.exists():
        ttf_files = list(output_dir.glob("*.ttf"))
        if ttf_files:
            return ttf_files[0]
    return None


# ===== 진행 상태 감지 =====
def detect_progress(work_dir: Path) -> Tuple[int, str]:
    """
    work 디렉토리의 파일 상태를 보고 진행 단계 추정
    반환: (진행률 0-100, 메시지)
    """
    if not work_dir or not work_dir.exists():
        return 0, "초기화 중..."
    
    font_imgs = work_dir / "font_imgs"
    font_data = work_dir / "font_data"
    result = work_dir / "result"
    output = work_dir / "output"
    
    # 1단계: GPT 이미지 생성
    if font_imgs.exists() and any(font_imgs.rglob("*.png")):
        if not font_data.exists() or not any(font_data.rglob("*.png")):
            return 25, "스타일 이미지 생성 중..."
    
    # 2단계: 전처리
    if font_data.exists() and any(font_data.rglob("*.png")):
        if not result.exists() or not any(result.rglob("*.png")):
            return 40, "이미지 전처리 중..."
    
    # 3단계: MX 추론
    if result.exists() and any(result.rglob("*.png")):
        if not output.exists() or not any(output.rglob("*.ttf")):
            return 70, "전체 글자 생성 중..."
    
    # 4단계: TTF 생성
    if output.exists() and any(output.rglob("*.ttf")):
        return 95, "폰트 파일 생성 완료"
    
    return 10, "작업 준비 중..."


# ===== 최신 work 디렉토리 찾기 =====
def find_latest_work_dir(since_time: float) -> Optional[Path]:
    """since_time 이후에 생성된 가장 최신 work-* 디렉토리 찾기"""
    work_dirs = [p for p in Path(".").glob("work-*") if p.is_dir()]
    if not work_dirs:
        return None
    
    recent = [p for p in work_dirs if p.stat().st_mtime >= since_time - 2.0]
    if not recent:
        return None
    
    return max(recent, key=lambda x: x.stat().st_mtime)


# ===== DB 저장 함수 (주석 처리) =====
# def save_to_database(work_dir: Path, prompt: str):
#     """
#     NoSQL DB에 결과 저장
#     - work_dir 내의 run_summary.json 읽기
#     - font_imgs 내의 이미지들을 base64로 인코딩
#     - TTF 파일 경로
#     - 프롬프트 정보
#     """
#     import base64
#     
#     summary_path = work_dir / "run_summary.json"
#     if not summary_path.exists():
#         return
#     
#     with open(summary_path, "r", encoding="utf-8") as f:
#         summary = json.load(f)
#     
#     # 이미지 수집 (base64 인코딩)
#     images = []
#     font_imgs_dir = work_dir / "font_imgs"
#     if font_imgs_dir.exists():
#         for img_path in font_imgs_dir.rglob("*.png"):
#             with open(img_path, "rb") as img_file:
#                 img_b64 = base64.b64encode(img_file.read()).decode()
#                 images.append({
#                     "filename": img_path.name,
#                     "data": img_b64
#                 })
#     
#     # TTF 파일 경로
#     ttf_path = find_latest_ttf(work_dir)
#     
#     # DB에 저장할 데이터 구조
#     db_data = {
#         "prompt": prompt,
#         "work_dir": str(work_dir),
#         "summary": summary,
#         "images": images,
#         "ttf_path": str(ttf_path) if ttf_path else None,
#         "created_at": summary.get("runs", {}).get("started_at"),
#     }
#     
#     # TODO: NoSQL DB에 저장
#     # db.collection("fonts").insert_one(db_data)
#     
#     print(f"[DB] 저장 준비 완료: {len(images)}개 이미지, TTF: {ttf_path}")


# ===== 메인 엔드포인트: 폰트 생성 (SSE 스트리밍) =====
@app.post("/generate")
async def generate_font(request: GenerateRequest):
    """
    프롬프트를 받아 main.py 실행 후 진행 상황을 SSE로 스트리밍
    """
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="프롬프트가 비어있습니다.")
    
    def event_generator() -> Iterator[str]:
        start_time = time.time()
        work_dir = None
        
        # 초기 상태 전송
        yield sse_event({"progress": 0, "message": "작업 시작 중...", "stage": "init"}, event="progress")
        
        # main.py 실행
        cmd = [sys.executable, "-u", MAIN_SCRIPT, "-p", prompt]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )
            
            last_progress = 0
            last_check = time.time()
            
            # 프로세스 실행 중 진행 상황 모니터링
            while proc.poll() is None:
                now = time.time()
                
                # 0.5초마다 진행 상황 체크
                if now - last_check >= 0.5:
                    last_check = now
                    
                    # work 디렉토리 찾기
                    if work_dir is None:
                        work_dir = find_latest_work_dir(start_time)
                    
                    # 진행 상황 감지
                    if work_dir:
                        progress, message = detect_progress(work_dir)
                        
                        # 진행률이 변경되었을 때만 전송
                        if progress != last_progress:
                            last_progress = progress
                            
                            # 단계별 이벤트 전송
                            stage = "init"
                            if progress >= 25:
                                stage = "gpt_api"
                            if progress >= 40:
                                stage = "preprocessing"
                            if progress >= 70:
                                stage = "inference"
                            if progress >= 95:
                                stage = "fontforge"
                            
                            yield sse_event({
                                "progress": progress,
                                "message": message,
                                "stage": stage
                            }, event="progress")
                
                time.sleep(0.3)
            
            # 프로세스 종료 대기
            returncode = proc.wait()
            
            if returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                yield sse_event({
                    "error": "폰트 생성 실패",
                    "details": stderr[-500:] if stderr else "알 수 없는 오류"
                }, event="error")
                return
            
            # 성공: TTF 파일 찾기
            if work_dir is None:
                work_dir = find_latest_work_dir(start_time)
            
            if work_dir:
                ttf_path = find_latest_ttf(work_dir)
                
                if ttf_path and ttf_path.exists():
                    # 완료 이벤트 전송
                    yield sse_event({
                        "progress": 100,
                        "message": "폰트 생성 완료!",
                        "stage": "complete",
                        "work_dir": str(work_dir.name),
                        "filename": ttf_path.name
                    }, event="complete")
                    
                    # DB 저장 (주석 처리)
                    # save_to_database(work_dir, prompt)
                    
                else:
                    yield sse_event({
                        "error": "TTF 파일을 찾을 수 없습니다."
                    }, event="error")
            else:
                yield sse_event({
                    "error": "작업 디렉토리를 찾을 수 없습니다."
                }, event="error")
        
        except Exception as e:
            yield sse_event({
                "error": f"서버 오류: {str(e)}"
            }, event="error")
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ===== TTF 파일 다운로드 엔드포인트 =====
@app.get("/download/{work_dir}/{filename}")
async def download_font(work_dir: str, filename: str):
    """
    생성된 TTF 파일 다운로드
    """
    file_path = Path(work_dir) / "output" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="font/ttf"
    )


# ===== 헬스체크 =====
@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "ok", "service": "MonoGlyph Font Generator"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
