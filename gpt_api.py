#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import json
import time
import base64
import argparse
from pathlib import Path
from typing import List
import requests

# ===== 기본 설정 =====
DEFAULT_MODEL = "gpt-image-1"
GENERATE_ENDPOINT = "https://api.openai.com/v1/images/generations"

DEFAULT_OUTDIR   = "font_imgs"     # 저장 폴더 기본값 (유지)
DEFAULT_SIZE     = "1024x1024"     # 해상도 기본값 (유지)
DEFAULT_QUALITY  = "low"           # 퀄리티 기본값 (유지: 미지원 시 standard로 폴백)
DEFAULT_DELAY    = 0.0             # 요청 간 대기(초), 레이트리밋 회피용

Common_Prompt = "추가적으로 이미지의 배경은 모두 흰색으로 글씨는 검은색으로 만들어줘야해 그리고 폰트들을 생성할 때 단어들의 특색을 잘 살려줘"

# ===== 14개 프롬프트 템플릿 =====
CHARS_14 = list("가나다라마바사아자차카타파하")
PROMPT_TEMPLATES = [
    f"{{q}}, 해당 스타일의 폰트로 작성한 한글 '{ch}' 이미지를 만들어줘, {{Common_Prompt}}"
    for ch in CHARS_14
]

# ===== 14개 결과 파일명 =====
OUTPUT_FILENAMES = [
    "가.png", "나.png", "다.png", "라.png",
    "마.png", "바.png", "사.png", "아.png",
    "자.png", "차.png", "카.png", "타.png",
    "파.png", "하.png",
]


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_safe_dirname(name: str) -> str:
    safe = "".join(c if (c.isalnum() or c in " _-.") else "_" for c in name).strip()
    return (safe[:50] or "prompt")


def save_b64_png(b64: str, out_path: Path) -> None:
    img_bytes = base64.b64decode(b64)
    out_path.write_bytes(img_bytes)


def load_api_key() -> str:
    """
    .env 또는 환경변수에서 API 키를 로드한다.
    우선순위: .env의 API_KEY -> .env의 OPENAI_API_KEY -> 환경변수들
    """
    # .env 로딩 (있으면)
    try:
        from dotenv import load_dotenv
        # 프로젝트 루트(.env)와 실행 경로(.env) 순서로 시도
        here = Path(__file__).resolve().parent
        for candidate in [here / ".env", Path.cwd() / ".env"]:
            if candidate.exists():
                load_dotenv(dotenv_path=candidate, override=False)
                break
        else:
            # .env가 없어도 조용히 패스
            load_dotenv(override=False)
    except Exception:
        # python-dotenv 미설치/에러 시에도 환경변수만으로 진행
        pass

    # 키 후보
    for key_name in ("API_KEY", "OPENAI_API_KEY"):
        val = os.getenv(key_name)
        if val and val.strip():
            return val.strip()

    # 실패 시 에러
    raise SystemExit(
        "[ERROR] API 키를 찾지 못했습니다.\n"
        " - .env 파일에 다음 중 하나를 넣어주세요:\n"
        "     API_KEY=sk-...\n"
        "     OPENAI_API_KEY=sk-...\n"
        " - 또는 환경변수(API_KEY/OPENAI_API_KEY)를 설정하세요."
    )


def post_generate_once(api_key: str,
                       prompt: str,
                       size: str,
                       quality: str,
                       background: str = None,
                       timeout: int = 90) -> dict:
    """
    단일 이미지 생성 요청. quality='low'가 400이면 자동으로 'standard'로 재시도.
    성공 시 {'data': [{'b64_json': ...}]} 형태의 dict 반환.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "n": 1,                 # 한 번에 1장만 생성
        "size": size,           # 예: 1024x1024
        "quality": quality,     # 'low' 미지원 계정/리전 존재 → 아래 폴백
    }
    if background:
        payload["background"] = background  # 예: "transparent" (미지원일 수 있음)

    r = requests.post(GENERATE_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    if r.status_code == 200:
        return r.json()

    # quality=low 미지원 계정/리전일 가능성 → standard로 폴백
    if r.status_code == 400 and "quality" in r.text.lower() and quality == "low":
        payload["quality"] = "standard"
        r2 = requests.post(GENERATE_ENDPOINT, headers=headers, json=payload, timeout=timeout)
        if r2.status_code == 200:
            return r2.json()
        raise requests.HTTPError(f"Image API 호출 실패(standard 폴백도 실패): {r2.status_code}: {r2.text}")

    raise requests.HTTPError(f"Image API 호출 실패: {r.status_code}: {r.text}")


def build_final_prompts(user_prompt: str) -> List[str]:
    q = user_prompt.strip()
    return [tpl.format(q=q, Common_Prompt=Common_Prompt) for tpl in PROMPT_TEMPLATES]


def main():
    parser = argparse.ArgumentParser(description="프롬프트 1개로 14장 생성 후 저장하고 자동 종료")
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="사용자 프롬프트(미지정 시 실행 중 1회 입력)")
    parser.add_argument("-o", "--out_dir", type=str, default=DEFAULT_OUTDIR,
                        help=f"저장 폴더 (기본: ./{DEFAULT_OUTDIR})")
    parser.add_argument("-s", "--size", type=str, default=DEFAULT_SIZE,
                        choices=["1024x1024", "2048x2048"],
                        help=f"출력이미지 해상도 (기본: {DEFAULT_SIZE})")
    parser.add_argument("--quality", type=str, default=DEFAULT_QUALITY,
                        choices=["low", "standard", "hd"],
                        help=f"생성 품질 (기본: {DEFAULT_QUALITY}, low 미지원 시 자동 standard 폴백)")
    args = parser.parse_args()

    # API 키 로드 (.env/환경변수)
    api_key = load_api_key()

    # 사용자 프롬프트
    user_prompt = args.prompt
    if not user_prompt:
        try:
            user_prompt = input("프롬프트를 입력하세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n입력이 취소되었습니다.")
            sys.exit(1)
    if not user_prompt:
        print("[ERROR] 빈 프롬프트입니다.")
        sys.exit(1)

    # 템플릿/파일명 수 검증
    if len(PROMPT_TEMPLATES) != 14 or len(OUTPUT_FILENAMES) != 14:
        print("[ERROR] 템플릿 또는 파일명 수가 14개가 아닙니다. 코드를 확인하세요.")
        sys.exit(1)

    prompt_dirname = make_safe_dirname(user_prompt)
    out_dir = Path(args.out_dir) / prompt_dirname   # output/프롬프트명
    ensure_outdir(out_dir)

    background = "transparent" if args.transparent else None
    final_prompts = build_final_prompts(user_prompt)

    # 14장 생성 루프
    for idx, (sub_prompt, fname) in enumerate(zip(final_prompts, OUTPUT_FILENAMES), start=1):
        if args.delay > 0 and idx > 1:
            time.sleep(args.delay)
        try:
            resp = post_generate_once(
                api_key=api_key,
                prompt=sub_prompt,
                size=args.size,
                quality=args.quality,
                background=background,
            )
        except Exception as e:
            print(f"[ERROR] #{idx} 생성 실패: {e}")
            continue

        data = resp.get("data", [])
        if not data:
            print(f"[WARN] #{idx} 응답에 data가 없습니다. 원문: {json.dumps(resp, ensure_ascii=False)[:400]}")
            continue

        b64 = data[0].get("b64_json")
        if not b64:
            print(f"[WARN] #{idx} b64_json이 없습니다. 항목: {data[0]}")
            continue

        out_path = out_dir / fname
        try:
            save_b64_png(b64, out_path)
            print(f"[OK] 저장 완료 #{idx:02d}: {out_path.resolve()}")
        except Exception as e:
            print(f"[ERROR] 저장 실패 #{idx:02d} ({out_path}): {e}")

    print("모든 생성 요청이 완료되었습니다. 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
