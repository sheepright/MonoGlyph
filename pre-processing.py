from PIL import Image, ImageOps
import os
from pathlib import Path

def preprocess_image(input_path, output_path, size=(128, 128)):
    # 이미지 열기
    img = Image.open(input_path)

    # RGBA 또는 RGB → Grayscale ('L')
    if img.mode in ("RGBA", "RGB"):
        # 알파 채널이 있을 경우 흰색 배경으로 덮어씌우기
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 알파 채널을 마스크로 사용
            img = background

        img = img.convert("L")  # Grayscale로 변환

    elif img.mode != "L":
        img = img.convert("L")  # 다른 모드도 'L'로 변환

    # 자동 대비 보정 (글자 강조, 배경 정리)
    img = ImageOps.autocontrast(img)

    # 리사이즈 (Pillow 9: ANTIALIAS 사용 가능)
    img = img.resize(size, Image.ANTIALIAS)

    # 저장
    img.save(output_path, format="PNG")


def preprocess_folder(input_dir="font_imgs", output_dir="font_data"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # input_dir 하위의 모든 PNG 파일을 재귀적으로 처리
    for img_path in input_dir.rglob("*.png"):
        # input_dir 기준 상대 경로 유지
        rel = img_path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        preprocess_image(img_path, out_path)
        print(f"[OK] Processed: {rel}")

# ✅ 입출력 경로를 명시한 실행
if __name__ == "__main__":
    preprocess_folder(input_dir="font_imgs", output_dir="font_data")
