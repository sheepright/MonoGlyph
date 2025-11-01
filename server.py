#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import subprocess
import shutil
import time
import requests
import boto3
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Tuple, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

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

# AWS S3 설정 (환경변수에서 읽기)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")
API_ENDPOINT = os.getenv("API_ENDPOINT")

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    region_name=S3_REGION,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)


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


# ===== S3 업로드 함수 =====
def upload_images_to_s3(work_dir: Path) -> List[Dict[str, str]]:
    """
    font_imgs 디렉토리의 이미지를 S3에 업로드하고 public URL 반환
    S3 구조: work-*/font_imgs/*.png
    """
    uploaded_images = []
    font_imgs_dir = work_dir / "font_imgs"
    
    if not font_imgs_dir.exists():
        print(f"[S3] font_imgs 디렉토리가 없습니다: {font_imgs_dir}")
        return uploaded_images
    
    for img_path in sorted(font_imgs_dir.rglob("*.png")):
        try:
            # S3 키 생성 (work-*/font_imgs/filename.png)
            s3_key = f"{work_dir.name}/font_imgs/{img_path.name}"
            
            # S3에 업로드 (public-read 권한)
            s3_client.upload_file(
                str(img_path),
                S3_BUCKET_NAME,
                s3_key,
                # ExtraArgs={'ACL': 'public-read', 'ContentType': 'image/png'}
                ExtraArgs={'ContentType': 'image/png'}
            )
            
            # Public URL 생성
            public_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
            
            uploaded_images.append({
                "filename": img_path.name,
                "url": public_url,
                "s3_key": s3_key
            })
            
            print(f"[S3] 이미지 업로드 완료: {img_path.name} -> {public_url}")
            
        except Exception as e:
            print(f"[S3] 이미지 업로드 실패 ({img_path.name}): {str(e)}")
    
    return uploaded_images


def upload_ttf_to_s3(work_dir: Path, ttf_path: Path) -> Optional[str]:
    """
    TTF 파일을 S3에 업로드하고 public URL 반환
    S3 구조: work-*/output/*.ttf
    """
    if not ttf_path or not ttf_path.exists():
        print(f"[S3] TTF 파일이 없습니다: {ttf_path}")
        return None
    
    try:
        # S3 키 생성 (work-*/output/filename.ttf)
        s3_key = f"{work_dir.name}/output/{ttf_path.name}"
        
        # S3에 업로드 (public-read 권한)
        s3_client.upload_file(
            str(ttf_path),
            S3_BUCKET_NAME,
            s3_key,
            # ExtraArgs={'ACL': 'public-read', 'ContentType': 'font/ttf'}
            ExtraArgs={'ContentType': 'font/ttf'}
        )
        
        # Public URL 생성
        public_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        
        print(f"[S3] TTF 업로드 완료: {ttf_path.name} -> {public_url}")
        
        return public_url
        
    except Exception as e:
        print(f"[S3] TTF 업로드 실패 ({ttf_path.name}): {str(e)}")
        return None


# ===== DB 저장 함수 =====
def save_to_database(work_dir: Path, prompt: str):
    """
    run_summary.json을 읽고 S3에 파일 업로드 후 URL을 추가하여 API로 전송
    - font_imgs 이미지들을 S3에 업로드
    - TTF 파일을 S3에 업로드
    - font_output 경로를 S3 public URL로 변경
    - gpt_api_images의 file_path를 S3 URL로 변경
    - status 항목 삭제
    - summary는 font_output만 남기고 나머지 삭제
    """
    summary_path = work_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"[DB] run_summary.json이 없습니다: {summary_path}")
        return
    
    try:
        # run_summary.json 읽기
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # S3에 이미지 업로드
        uploaded_images = upload_images_to_s3(work_dir)
        
        # TTF 파일 찾기 및 S3 업로드
        ttf_path = find_latest_ttf(work_dir)
        ttf_public_url = None
        if ttf_path:
            ttf_public_url = upload_ttf_to_s3(work_dir, ttf_path)
        
        # status 항목 삭제
        if "status" in data:
            del data["status"]
        
        # summary 항목 정리 (ttf_output을 S3 URL로 변경)
        if "summary" in data:
            data["summary"] = {
                "ttf_output": ttf_public_url if ttf_public_url else None
            }
        else:
            data["summary"] = {
                "ttf_output": ttf_public_url if ttf_public_url else None
            }
        
        # gpt_api_images의 file_path를 S3 URL로 변경
        if "gpt_api_images" in data and isinstance(data["gpt_api_images"], list):
            # uploaded_images를 filename으로 매핑
            url_map = {img["filename"]: img["url"] for img in uploaded_images}
            
            for img_item in data["gpt_api_images"]:
                if "file_path" in img_item:
                    # 파일명 추출 (예: "font_imgs\\...\\가.png" -> "가.png")
                    filename = Path(img_item["file_path"]).name
                    # S3 URL로 변경
                    if filename in url_map:
                        img_item["file_path"] = url_map[filename]
        
        # 추가 데이터
        data["images"] = uploaded_images
        data["prompt"] = prompt
        data["work_dir"] = str(work_dir.name)
        data["ttf_filename"] = ttf_path.name if ttf_path else None
        
        # API로 POST 요청
        api_url = f"{API_ENDPOINT}/api/runs/ingest"
        response = requests.post(
            api_url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"[DB] API 저장 성공: {api_url}")
            print(f"[DB] 이미지 {len(uploaded_images)}개, TTF URL: {ttf_public_url}")
        else:
            print(f"[DB] API 저장 실패 ({response.status_code}): {response.text}")
    
    except Exception as e:
        print(f"[DB] 저장 중 오류 발생: {str(e)}")


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
                    
                    # DB 저장
                    save_to_database(work_dir, prompt)
                    
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
