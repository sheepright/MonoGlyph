#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import shutil
import subprocess
import sys
import unicodedata
import locale
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

# ===== 사용자 스크립트 파일명 =====
GEN_SCRIPT = "gpt_api.py"             # 1) 이미지 14장 생성
PRE_SCRIPT = "pre-processing.py"      # 2) 전처리
MX_SCRIPT  = "inference_MX.py"        # 3) MX 추론
FF_SCRIPT  = "font_switch.py"         # 4) FontForge 스크립트

# FontForge CLI (PATH 필요)
FONTFORGE_BIN = "fontforge"

# 작업 하위 폴더명 (각 스크립트 기본값과 일치)
DIR_IMGS   = "font_imgs"
DIR_DATA   = "font_data"
DIR_RESULT = "result"
DIR_OUT    = "output"
DIR_LOGS   = "logs"


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iso6_now() -> str:
    """DATETIME(6)와 유사한 포맷(마이크로초) + 로컬 타임존 오프셋 포함"""
    return datetime.now().astimezone().isoformat(timespec="microseconds")


def dur_sec(start_iso: str, end_iso: str) -> float:
    s = datetime.fromisoformat(start_iso)
    e = datetime.fromisoformat(end_iso)
    return round((e - s).total_seconds(), 3)


def run_and_log(cmd: list, cwd: Path, stdout_log: Path, stderr_log: Path, check: bool = True) -> int:
    """
    서브프로세스 실행 + stdout/stderr를 파일에만 기록(콘솔 미표시).
    ✅ 수정 사항:
      - 바이너리로 읽고, UTF-8 → CP949 → UTF-16(LE/BE) 순으로 디코딩 시도
      - 자식 프로세스에 UTF-8 강제 환경변수 주입(Python 계열 안전)
    """
    cwd.mkdir(parents=True, exist_ok=True)
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)

    # 자식 프로세스 환경: 가능한 UTF-8 강제
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    # 시스템 선호 인코딩도 후보에 포함
    sys_pref = locale.getpreferredencoding(False)

    def _decode(b: bytes) -> str:
        for enc in ("utf-8", "cp949", "utf-16", "utf-16-le", "utf-16-be", sys_pref, sys.getfilesystemencoding()):
            try:
                return b.decode(enc)
            except Exception:
                continue
        return b.decode("utf-8", errors="replace")

    with stdout_log.open("a", encoding="utf-8", errors="replace") as out, \
         stderr_log.open("a", encoding="utf-8", errors="replace") as err:

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,   # 바이너리
            stderr=subprocess.PIPE,   # 바이너리
            text=False,               # ✅ 텍스트 디코딩 금지 (직접 디코딩)
            env=env,
        )

        # 라인 단위로 읽되, 빈 바이트가 돌아오고 프로세스가 종료되면 루프 탈출
        while True:
            so = proc.stdout.readline() if proc.stdout else b""
            se = proc.stderr.readline() if proc.stderr else b""
            if so:
                out.write(_decode(so))
            if se:
                err.write(_decode(se))
            if not so and not se and proc.poll() is not None:
                break

        code = proc.wait()
        if check and code != 0:
            raise subprocess.CalledProcessError(code, cmd)

        return code


def ensure_exists(p: Path, desc: str):
    if not p.exists():
        raise FileNotFoundError(f"{desc}를(을) 찾을 수 없습니다: {p}")


def rmdir_if_exists(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def rmfile_if_exists(p: Path):
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="프롬프트 한 개로 end-to-end 파이프라인 실행 (기본값 고정)")
    parser.add_argument("--prompt", "-p", required=True, help="이미지 생성 및 최종 폰트명/파일명에 사용할 프롬프트")
    args = parser.parse_args()

    original_prompt = args.prompt.strip()
    if not original_prompt:
        print("[ERROR] 빈 프롬프트입니다.", file=sys.stderr)
        sys.exit(1)

    # 작업 폴더 생성
    work_dir = Path(f"work-{now_tag()}").resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 하위 폴더
    d_imgs   = work_dir / DIR_IMGS
    d_data   = work_dir / DIR_DATA
    d_result = work_dir / DIR_RESULT
    d_out    = work_dir / DIR_OUT
    d_logs   = work_dir / DIR_LOGS
    for d in (d_imgs, d_data, d_result, d_out, d_logs):
        d.mkdir(parents=True, exist_ok=True)

    # =========================
    # DB 스키마 형태 로그 준비
    # =========================
    run_id = f"run-{now_tag()}"
    db_prompts_row = {
        "id": None,  # DB에서 upsert 후 채울 값
        "name": original_prompt,
        "created_at": iso6_now(),
    }
    db_steps_rows: List[Dict[str, Any]] = []
    db_gpt_api_images_rows: List[Dict[str, Any]] = []
    runs_started_at = iso6_now()

    # ---------- MX 리소스 work 안으로 복사 ----------
    mx_script_path = Path(MX_SCRIPT).resolve()
    project_root   = mx_script_path.parent

    # cfgs/
    src_cfgs = project_root / "cfgs"
    dst_cfgs = work_dir / "cfgs"
    ensure_exists(src_cfgs, "cfgs 폴더")
    if dst_cfgs.exists():
        shutil.rmtree(dst_cfgs)
    shutil.copytree(src_cfgs, dst_cfgs)

    # 가중치
    src_weight = project_root / "MonoGlyph_60000.pth"
    dst_weight = work_dir / "MonoGlyph_60000.pth"
    ensure_exists(src_weight, "MonoGlyph_60000.pth 가중치 파일")
    shutil.copy2(str(src_weight), str(dst_weight))

    # kor_string.json
    kor_json_src = Path("kor_string.json").resolve()
    ensure_exists(kor_json_src, "kor_string.json")
    kor_json_dst = work_dir / "kor_string.json"
    kor_json_dst.write_text(kor_json_src.read_text(encoding="utf-8"), encoding="utf-8")

    # data/kor/source.ttf
    source_ttf_src = Path("data/kor/source.ttf").resolve()
    ensure_exists(source_ttf_src, "data/kor/source.ttf")
    (work_dir / "data" / "kor").mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_ttf_src), str(work_dir / "data" / "kor" / "source.ttf"))

    # ---------- 1) 이미지 생성 (프롬프트만 전달) ----------
    step_label = "이미지 생성 (GPT_API)"
    step_name = "gpt_api"
    step_started = iso6_now()

    gen_cmd = [sys.executable, str(Path(GEN_SCRIPT).resolve()), "--prompt", original_prompt]
    try:
        rc = run_and_log(gen_cmd, work_dir, d_logs / f"{step_label}.stdout.log", d_logs / f"{step_label}.stderr.log")
    except Exception as e:
        rc = -1
    step_ended = iso6_now()

    imgs_gpt = sorted((work_dir / DIR_IMGS).rglob("*.png"))
    db_steps_rows.append({
        "run_id": run_id,
        "step_name": step_name,
        "started_at": step_started,
        "ended_at": step_ended,
        "duration_sec": dur_sec(step_started, step_ended),
        "ok": (rc == 0),
        "image_count": len(imgs_gpt),
    })
    # gpt_api_images: 작업폴더 기준 상대경로(\ 사용)
    now_for_imgs = iso6_now()
    for p in imgs_gpt:
        rel = str(p.relative_to(work_dir)) if p.is_file() else str(p)
        db_gpt_api_images_rows.append({
            "id": None,
            "prompt_id": None,   # DB 적재 시 prompts.id로 채워 넣을 것
            "run_id": run_id,
            "file_path": rel.replace("/", "\\"),
            "created_at": now_for_imgs,
        })
    if rc != 0:
        # 실패 즉시 요약 저장
        db_runs_row = {
            "run_id": run_id,
            "prompt_id": None,
            "started_at": runs_started_at,
            "ended_at": iso6_now(),
            "total_duration_sec": dur_sec(runs_started_at, iso6_now()),
        }
        db_ready_summary = {
            "prompts": db_prompts_row,
            "runs": db_runs_row,
            "steps": db_steps_rows,
            "gpt_api_images": db_gpt_api_images_rows,
            "summary": {},
            "status": {"ok": False, "exit_code": rc, "message": "gpt_api failed"},
        }
        (work_dir / "run_summary.json").write_text(json.dumps(db_ready_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FATAL] gpt_api 실패. run_summary.json 저장됨: {work_dir/'run_summary.json'}", file=sys.stderr)
        sys.exit(1)

    # ---------- 2) 전처리 (기본값) ----------
    step_label = "이미지 전처리 (pre-processing)"
    step_name = "pre_processing"
    step_started = iso6_now()

    pre_cmd = [sys.executable, str(Path(PRE_SCRIPT).resolve())]
    try:
        rc = run_and_log(pre_cmd, work_dir, d_logs / f"{step_label}.stdout.log", d_logs / f"{step_label}.stderr.log")
    except Exception as e:
        rc = -1
    step_ended = iso6_now()

    imgs_pre = sorted((work_dir / DIR_DATA).rglob("*.png"))
    db_steps_rows.append({
        "run_id": run_id,
        "step_name": step_name,
        "started_at": step_started,
        "ended_at": step_ended,
        "duration_sec": dur_sec(step_started, step_ended),
        "ok": (rc == 0),
        "image_count": len(imgs_pre),
    })
    if rc != 0:
        db_runs_row = {
            "run_id": run_id,
            "prompt_id": None,
            "started_at": runs_started_at,
            "ended_at": iso6_now(),
            "total_duration_sec": dur_sec(runs_started_at, iso6_now()),
        }
        db_ready_summary = {
            "prompts": db_prompts_row,
            "runs": db_runs_row,
            "steps": db_steps_rows,
            "gpt_api_images": db_gpt_api_images_rows,
            "summary": {},
            "status": {"ok": False, "exit_code": rc, "message": "pre_processing failed"},
        }
        (work_dir / "run_summary.json").write_text(json.dumps(db_ready_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FATAL] pre-processing 실패. run_summary.json 저장됨: {work_dir/'run_summary.json'}", file=sys.stderr)
        sys.exit(1)

    # ---------- 3) MX 추론 (기본값) ----------
    step_label = "전체 글자 이미지 생성 (inference_MX)"
    step_name = "inference_MX"
    step_started = iso6_now()

    mx_cmd = [sys.executable, str(Path(MX_SCRIPT).resolve())]
    try:
        rc = run_and_log(mx_cmd, work_dir, d_logs / f"{step_label}.stdout.log", d_logs / f"{step_label}.stderr.log")
    except Exception as e:
        rc = -1
    step_ended = iso6_now()

    imgs_res = sorted((work_dir / DIR_RESULT).rglob("*.png"))
    db_steps_rows.append({
        "run_id": run_id,
        "step_name": step_name,
        "started_at": step_started,
        "ended_at": step_ended,
        "duration_sec": dur_sec(step_started, step_ended),
        "ok": (rc == 0),
        "image_count": len(imgs_res),
    })
    if rc != 0:
        db_runs_row = {
            "run_id": run_id,
            "prompt_id": None,
            "started_at": runs_started_at,
            "ended_at": iso6_now(),
            "total_duration_sec": dur_sec(runs_started_at, iso6_now()),
        }
        db_ready_summary = {
            "prompts": db_prompts_row,
            "runs": db_runs_row,
            "steps": db_steps_rows,
            "gpt_api_images": db_gpt_api_images_rows,
            "summary": {},
            "status": {"ok": False, "exit_code": rc, "message": "inference_MX failed"},
        }
        (work_dir / "run_summary.json").write_text(json.dumps(db_ready_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[FATAL] inference_MX 실패. run_summary.json 저장됨: {work_dir/'run_summary.json'}", file=sys.stderr)
        sys.exit(1)

    # ---------- 4) FontForge → TTF 생성 (입력은 무조건 result/) ----------
    step_label = "TTF 파일 생성 (font_switch)"
    step_name = "fontforge_ttf"
    step_started = iso6_now()

    # 출력 파일명 고정
    out_ttf_path = (work_dir / DIR_OUT / "MonoGlyph.ttf")

    # 절대 POSIX 경로
    ff_input_dir  = (work_dir / DIR_RESULT).resolve().as_posix()  # result 고정
    ff_chars_json = (work_dir / "kor_string.json").resolve().as_posix()
    ff_out_ttf    = out_ttf_path.resolve().as_posix()

    # 폰트 표시용 이름: MonoGlyph_<timestamp>
    font_display_name = f"MonoGlyph_{now_tag()}"

    ff_cmd = [
        FONTFORGE_BIN, "-script", str(Path(FF_SCRIPT).resolve()),
        "--input_dir", ff_input_dir,
        "--chars_json", ff_chars_json,
        "--out_ttf", ff_out_ttf,
        "--fontname", font_display_name,
    ]
    try:
        rc = run_and_log(ff_cmd, work_dir, d_logs / f"{step_label}.stdout.log", d_logs / f"{step_label}.stderr.log")
    except Exception as e:
        rc = -1
    step_ended = iso6_now()

    db_steps_rows.append({
        "run_id": run_id,
        "step_name": step_name,
        "started_at": step_started,
        "ended_at": step_ended,
        "duration_sec": dur_sec(step_started, step_ended),
        "ok": (rc == 0),
        "image_count": 0,
    })

    # 폴백 시도 (output.ttf)
    if not out_ttf_path.exists():
        fallback_path = (work_dir / DIR_OUT / "output.ttf").resolve()
        ff_out_ttf_fb = fallback_path.as_posix()
        step_label_fb = "TTF 파일 생성 (font_switch)_fallback"
        step_name_fb  = "fontforge_ttf_fallback"
        step_started_fb = iso6_now()

        ff_cmd_fallback = [
            FONTFORGE_BIN, "-script", str(Path(FF_SCRIPT).resolve()),
            "--input_dir", ff_input_dir,
            "--chars_json", ff_chars_json,
            "--out_ttf", ff_out_ttf_fb,
            "--fontname", font_display_name,
        ]
        try:
            rc_fb = run_and_log(ff_cmd_fallback, work_dir,
                                d_logs / f"{step_label_fb}.stdout.log",
                                d_logs / f"{step_label_fb}.stderr.log")
        except Exception as e:
            rc_fb = -1
        step_ended_fb = iso6_now()

        db_steps_rows.append({
            "run_id": run_id,
            "step_name": step_name_fb,
            "started_at": step_started_fb,
            "ended_at": step_ended_fb,
            "duration_sec": dur_sec(step_started_fb, step_ended_fb),
            "ok": (rc_fb == 0),
            "image_count": 0,
        })

        if fallback_path.exists():
            out_ttf_path = fallback_path
        else:
            # 실패 요약 저장 후 종료
            db_runs_row = {
                "run_id": run_id,
                "prompt_id": None,
                "started_at": runs_started_at,
                "ended_at": iso6_now(),
                "total_duration_sec": dur_sec(runs_started_at, iso6_now()),
            }
            db_ready_summary = {
                "prompts": db_prompts_row,
                "runs": db_runs_row,
                "steps": db_steps_rows,
                "gpt_api_images": db_gpt_api_images_rows,
                "summary": {},
                "status": {"ok": False, "exit_code": rc_fb, "message": "fontforge failed"},
            }
            (work_dir / "run_summary.json").write_text(json.dumps(db_ready_summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[FATAL] FontForge가 .ttf를 생성하지 못했습니다. run_summary.json 저장됨: {work_dir/'run_summary.json'}", file=sys.stderr)
            sys.exit(1)

    # ---------- 성공 마무리 ----------
    runs_ended_at = iso6_now()
    db_runs_row = {
        "run_id": run_id,
        "prompt_id": None,  # DB 적재 시 prompts.id로 채울 것
        "started_at": runs_started_at,
        "ended_at": runs_ended_at,
        "total_duration_sec": dur_sec(runs_started_at, runs_ended_at),
    }

    summary_block = {
        "gpt_images": {
            "count": len(sorted((work_dir / DIR_IMGS).rglob("*.png"))),
        },
        "preprocessed_images": {
            "count": len(sorted((work_dir / DIR_DATA).rglob("*.png"))),
        },
        "result_images": {
            "count": len(sorted((work_dir / DIR_RESULT).rglob("*.png"))),
        },
        "ttf_output": str(out_ttf_path.resolve()),
    }
    status_block = {
        "ok": True,
        "exit_code": 0,
        "message": "success",
    }

    db_ready_summary = {
        "prompts": db_prompts_row,
        "runs": db_runs_row,
        "steps": db_steps_rows,
        "gpt_api_images": db_gpt_api_images_rows,
        "summary": summary_block,
        "status": status_block,
    }
    (work_dir / "run_summary.json").write_text(
        json.dumps(db_ready_summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # ---------- 정리(성공 시에만) ----------
    try:
        # MX 리소스 + 중간 산출물 제거
        rmdir_if_exists(work_dir / "cfgs")
        rmfile_if_exists(work_dir / "MonoGlyph_60000.pth")
        rmfile_if_exists(work_dir / "kor_string.json")
        rmdir_if_exists(work_dir / "data")

        # 남기는 것: font_imgs/, output/, logs/, run_summary.json
        # 지우는 것: font_data/, result/
        rmdir_if_exists(d_data)
        rmdir_if_exists(d_result)
    except Exception:
        pass

    print("\n================ DONE ================\n")
    print(f"[OK] 작업 디렉토리: {work_dir}")
    print(f"[OK] 최종 TTF:     {out_ttf_path}")
    print(f"[OK] 요약 JSON:    {work_dir / 'run_summary.json'}")
    print(f"[OK] 로그 폴더:    {d_logs}")
    print("=====================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] 파이프라인 실패: {e}", file=sys.stderr)
        sys.exit(1)
