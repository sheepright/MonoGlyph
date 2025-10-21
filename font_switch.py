#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Raster PNG(흑백 글자 이미지) → (potrace 또는 FontForge autoTrace) → 벡터 아웃라인 → TTF
- 파일명: '가.png'처럼 글자 1개가 파일명인 구조를 가정(하위 폴더 포함 가능)
- kor_string.json: "가나다..." 형태의 단일 JSON 문자열 (생성 대상 문자 집합)
- potrace가 설치되어 있으면 PBM→SVG→FontForge, 없으면 FontForge autoTrace로 처리
- 실행: fontforge -script font_switch.py --input_dir font_data --chars_json kor_string.json --out_ttf MonoGlyph.ttf --fontname MonoGlyph
"""

import sys, os, json, tempfile, subprocess, shutil
from pathlib import Path

# FontForge의 파이썬 환경에서 실행해야 import 가능
import fontforge
import psMat

import argparse

# --------- 설정 기본값 ---------
UPEM = 1000                 # Units Per EM
ASCENT = 800                # 상승 높이
DESCENT = UPEM - ASCENT     # 하강 높이
MARGIN = 50                 # 좌우/상하 여백(폰트 좌표계 기준)
ALLOW_MISSING_SKIP = True   # 이미지 없는 글자는 건너뛰기


def has_potrace() -> bool:
    return shutil.which("potrace") is not None


def png_to_svg_with_potrace(png_path: Path, tmp_dir: Path) -> Path:
    """
    PNG → PBM → potrace -s → SVG
    ImageMagick의 'convert'가 있으면 PBM 변환이 쉬운데, 없는 경우라도 FontForge 환경에서 PNG→PBM 실패시에는
    Pillow 등을 별도 사용해야 함. 여기선 'potrace'만 의존(최근 potrace는 PNG 직접 입력 안 되므로 PBM 권장).
    """
    pbm_path = tmp_dir / (png_path.stem + ".pbm")
    svg_path = tmp_dir / (png_path.stem + ".svg")

    # ImageMagick 'magick' 또는 'convert' 중 사용 가능한 명령 탐색
    magick = shutil.which("magick") or shutil.which("convert")
    if magick is None:
        # ImageMagick이 없다면, FontForge의 image→pbm은 제공되지 않으므로
        # Pillow를 쓰는 별도 파이썬이 필요. 간단히 실패 처리.
        raise RuntimeError("PBM 변환 도구가 없습니다. ImageMagick의 'magick' 또는 'convert'를 설치하세요.")

    # 1) PNG -> 모노 PBM
    #   -threshold로 바이너리화(200은 경험적 값, 입력이 흰배경/검정글씨인 전제)
    cmd_pbm = [magick, str(png_path), "-threshold", "200", str(pbm_path)]
    subprocess.run(cmd_pbm, check=True)

    # 2) PBM -> SVG (potrace)
    cmd_svg = ["potrace", "-s", "-o", str(svg_path), str(pbm_path)]
    subprocess.run(cmd_svg, check=True)

    return svg_path


def autotrace_inside_fontforge(glyph: "fontforge.glyph", png_path: Path) -> None:
    """
    FontForge 내부 autoTrace 경로: PNG를 배경 이미지로 붙인 뒤 autoTrace 수행
    """
    img = fontforge.image(str(png_path))
    glyph.addImage(img)     # 배경 이미지 추가
    glyph.autoTrace()       # 배경 이미지로부터 윤곽 추출
    # 배경 이미지는 제거(선택)
    for im in list(glyph.images):
        glyph.removeImage(im)


def fit_glyph_box(glyph: "fontforge.glyph"):
    """
    글리프를 UPEM 박스 내로 스케일/정렬:
    - 여백(MARGIN)을 두고 X/Y 방향으로 최대한 크게 배치
    - 좌측여백=MARGIN, 하단여백=MARGIN으로 정렬
    - advance width는 좌우 여백 + 실제 폭 + 여백(= MARGIN*2 + bbox_w)
    """
    # 윤곽이 없을 수 있음
    try:
        xmin, ymin, xmax, ymax = glyph.boundingBox()
    except Exception:
        # 빈 글리프
        glyph.width = UPEM
        return

    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)

    target_w = UPEM - 2*MARGIN
    target_h = UPEM - 2*MARGIN

    scale = min(target_w / bw, target_h / bh)
    m = psMat.scale(scale)
    glyph.transform(m)

    # 다시 bbox 계산
    xmin, ymin, xmax, ymax = glyph.boundingBox()
    dx = MARGIN - xmin
    dy = MARGIN - ymin
    glyph.transform(psMat.translate(dx, dy))

    # 정리
    try:
        glyph.removeOverlap()
    except Exception:
        pass
    glyph.correctDirection()
    glyph.simplify()
    glyph.round()

    # advance width: 좌우 여백 기준
    xmin, ymin, xmax, ymax = glyph.boundingBox()
    glyph.width = int(xmax + MARGIN)


def build_font(images_root: Path, chars: str, out_ttf: Path, fontname: str):
    """
    images_root 아래에 '가.png' 형태의 파일이 존재한다고 가정(하위 폴더 포함).
    """
    f = fontforge.font()
    f.encoding = "UnicodeFull"
    f.em = UPEM
    f.ascent = ASCENT
    f.descent = DESCENT

    # 네이밍
    f.fontname = fontname.replace(" ", "")
    f.familyname = fontname
    f.fullname = fontname

    # potrace 사용 가능?
    use_potrace = has_potrace()
    tmp_dir = Path(tempfile.mkdtemp(prefix="ff_vec_"))

    # 빠른 조회를 위해 파일 맵 구성 (글자→파일경로)
    # 우선 루트 바로 아래, 이어서 재귀적으로 찾음(뒤에서 찾은 걸로 덮어쓰지 않도록 주의)
    file_map = {}
    for p in list(images_root.glob("*.png")) + list(images_root.rglob("*.png")):
        try:
            ch = p.stem  # '가'
            if len(ch) == 1:
                file_map[ch] = p
        except Exception:
            continue

    missing = []
    imported = 0

    for ch in chars:
        # 해당 문자 이미지 찾기
        img_path = file_map.get(ch)
        if img_path is None:
            missing.append(ch)
            if ALLOW_MISSING_SKIP:
                continue
            else:
                raise FileNotFoundError(f"이미지를 찾지 못함: {ch}")

        # 글리프 만들기
        g = f.createChar(ord(ch))
        g.width = UPEM

        try:
            if use_potrace:
                # PNG -> SVG -> import (보다 안정적)
                svg_path = png_to_svg_with_potrace(img_path, tmp_dir)
                g.importOutlines(str(svg_path))
            else:
                # 내부 autoTrace 사용
                autotrace_inside_fontforge(g, img_path)

            fit_glyph_box(g)
            imported += 1
        except Exception as e:
            print(f"[WARN] {ch} 처리 실패: {e}")

    # 간단한 OS/명세 테이블(필요시 확장)
    f.os2_vendor = "KOR "
    f.os2_width = 5
    f.os2_weight = 400

    # 생성
    out_ttf.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Import OK: {imported}, Missing: {len(missing)}")
    if missing:
        print("[INFO] Missing chars:", "".join(missing)[:200], "..." if len(missing) > 200 else "")
    f.generate(str(out_ttf))
    f.close()
    print(f"[OK] Generated: {out_ttf.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="PNG 글자 이미지들을 FontForge로 묶어 TTF 생성")
    ap.add_argument("--input_dir", required=True, help="글자 PNG들이 있는 루트 폴더 (하위 폴더 포함 검색)")
    ap.add_argument("--chars_json", required=True, help='생성 대상 문자들이 들어있는 JSON 파일 (예: "가나다...")')
    ap.add_argument("--out_ttf", required=True, help="출력 TTF 파일 경로")
    ap.add_argument("--fontname", default="MyHangul", help="폰트 이름(패밀리/풀네임)")
    args = ap.parse_args()

    images_root = Path(args.input_dir)
    out_ttf = Path(args.out_ttf)

    with open(args.chars_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, str):
        raise ValueError("chars_json에는 \"가나다...\" 형태의 단일 문자열이 들어있어야 합니다.")
    chars = obj

    build_font(images_root, chars, out_ttf, args.fontname)


if __name__ == "__main__":
    main()