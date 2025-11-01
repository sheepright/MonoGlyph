# MonoGlyph

**한이음 드림업 프로젝트** - LLM과 GAN 기반 딥러닝 기술을 활용한 AI 한글 폰트 자동 생성 서비스입니다.

사용자는 손글씨 없이도 원하는 스타일 설명만으로 한글 폰트를 쉽게 제작할 수 있으며, PC·모바일 웹 기반의 직관적인 UI를 통해 간편하게 이용할 수 있습니다. GPT API로 스타일 이미지를 생성하고, MX 모델로 일부 한글 글자를 생성한 후 TTF 폰트 파일로 변환하는 완전 자동화된 파이프라인을 제공합니다.

## 📋 목차

- [시스템 요구사항](#시스템-요구사항)
- [설치 가이드](#설치-가이드)
- [초기 설정](#초기-설정)
- [사용법](#사용법)
- [파이프라인 구조](#파이프라인-구조)
- [API 서버 실행](#api-서버-실행)
- [문제 해결](#문제-해결)

## 🖥️ 시스템 요구사항

### 필수 소프트웨어

- **Python 3.8+** (권장: 3.8-3.10)
- **CUDA 지원 GPU** (PyTorch CUDA 버전 필요)
- **FontForge** - TTF 폰트 생성용
- **ImageMagick** - 이미지 전처리용 (선택사항, potrace 사용 시)

### 운영체제 지원

- Windows 10/11
- macOS 10.15+

## 🚀 설치 가이드

### 1. 저장소 클론

```bash
git clone <repository-url>
cd MonoGlyph
```

### 2. Python 가상환경 생성 (권장)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. 기본 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. PyTorch 설치 (CUDA 버전)

CUDA 버전에 맞게 설치하세요:

```bash
# CUDA 11.8 예시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 예시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 5. FontForge 설치

#### Windows

1. [FontForge 공식 사이트](https://fontforge.org/en-US/downloads/)에서 Windows 버전 다운로드
2. 설치 후 시스템 PATH에 `fontforge` 명령어 추가

#### macOS

```bash
brew install fontforge
```

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install fontforge
```

### 6. ImageMagick 설치

potrace를 사용한 고품질 벡터화를 원하는 경우:

#### Windows

[ImageMagick 공식 사이트](https://imagemagick.org/script/download.php#windows)에서 다운로드

#### macOS

```bash
brew install imagemagick
brew install potrace
```

#### Ubuntu/Debian

```bash
sudo apt-get install imagemagick potrace
```

## ⚙️ 초기 설정

### 1. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 필요한 환경 변수를 추가하세요:

```env
# .env 파일

# OpenAI API 키 (필수)
API_KEY=sk-your-openai-api-key-here
# 또는
OPENAI_API_KEY=sk-your-openai-api-key-here

# AWS S3 설정 (API 서버 사용 시 필수)
AWS_ACCESS_KEY_ID=your-aws-access-key-id
AWS_SECRET_ACCESS_KEY=your-aws-secret-access-key
S3_BUCKET_NAME=your-s3-bucket-name
S3_REGION=ap-northeast-2

# API 엔드포인트 (선택사항)
API_ENDPOINT=https://your-api-endpoint.com
```

### 2. 모델 가중치 파일 준비

`MonoGlyph.pth` 파일을 프로젝트 루트에 배치하세요.

**참고**: 가중치 파일은 본 저장소에서 제공되지 않습니다. [MX-Font](https://github.com/clovaai/mxfont) 모델을 사용하여 60,000 에포크까지 학습한 가중치를 권장합니다. MX-Font 학습 방법은 해당 저장소의 문서를 참조하세요.

### 3. 폰트 소스 파일 확인

`data/kor/source.ttf` 파일이 존재하는지 확인하세요.

### 4. 생성 대상 문자 설정

`kor_string.json` 파일에서 생성할 한글 문자를 설정할 수 있습니다.

- (기본 2.780자의 자주 사용하는 한글 조합 설정)

## 🎯 사용법

### 1. 전체 파이프라인 실행 (권장)

```bash
python main.py -p "원하는 폰트 스타일 설명"
```

예시:

```bash
python main.py -p "붓글씨 스타일의 우아한 서예체"
python main.py -p "모던하고 깔끔한 산세리프체"
python main.py -p "손글씨 느낌의 따뜻한 폰트"
```

### 2. 개별 스크립트 실행

#### 2-1. 이미지 생성만 실행

```bash
python gpt_api.py -p "폰트 스타일 설명"
```

추가 옵션:

```bash
python gpt_api.py -p "붓글씨" --size 2048x2048 --quality hd --out_dir custom_output
```

#### 2-2. 이미지 전처리만 실행

```bash
python pre-processing.py
```

#### 2-3. MX 모델 추론만 실행

```bash
python inference_MX.py
```

#### 2-4. TTF 폰트 생성만 실행

```bash
fontforge -script font_switch.py --input_dir result --chars_json kor_string.json --out_ttf MyFont.ttf --fontname "MyCustomFont"
```

## 🔄 파이프라인 구조

MonoGlyph는 4단계 파이프라인으로 구성됩니다:

### 1단계: 스타일 이미지 생성 (`gpt_api.py`)

- OpenAI GPT API를 사용하여 14개 기본 한글 문자 이미지 생성
- 생성 문자: `가나다라마바사아자차카타파하`
- 출력: `font_imgs/` 폴더에 PNG 파일들

### 2단계: 이미지 전처리 (`pre-processing.py`)

- 생성된 이미지를 128x128 그레이스케일로 변환
- 대비 보정 및 정규화 수행
- 출력: `font_data/` 폴더에 전처리된 PNG 파일들

### 3단계: 전체 글자 생성 (`inference_MX.py`)

- MX 모델을 사용하여 `kor_string.json`에 정의된 모든 한글 문자 생성
- 14개 참조 이미지를 바탕으로 2.780개의 한글 문자 생성
- 출력: `result/` 폴더에 생성된 모든 글자 이미지

### 4단계: TTF 폰트 생성 (`font_switch.py`)

- FontForge를 사용하여 이미지들을 벡터화하고 TTF 파일로 패키징
- potrace 또는 FontForge 내장 autoTrace 사용
- 출력: TTF 폰트 파일

## 🌐 API 서버 실행

### 서버 시작

웹 인터페이스를 통해 폰트를 생성하려면:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

서버 실행 후 `http://localhost:8000`에서 웹 인터페이스 사용 가능합니다.

### API 엔드포인트

#### 1. 폰트 생성 (SSE 스트리밍)

```bash
POST /generate
Content-Type: application/json

{
  "prompt": "붓글씨 스타일의 우아한 서예체"
}
```

- 실시간 진행 상황을 Server-Sent Events(SSE)로 스트리밍
- 진행 단계: init → gpt_api → preprocessing → inference → fontforge → complete
- 완료 시 work 디렉토리명과 TTF 파일명 반환

#### 2. TTF 파일 다운로드

```bash
GET /download/{work_dir}/{filename}
```

예시: `GET /download/work-20241101_123456/MonoGlyph.ttf`

#### 3. 헬스체크

```bash
GET /health
```

### S3 통합 기능

API 서버는 생성된 폰트와 이미지를 AWS S3에 자동으로 업로드합니다:

#### S3 업로드 항목

1. **스타일 이미지**: `font_imgs/` 디렉토리의 모든 PNG 파일
2. **TTF 폰트 파일**: `output/` 디렉토리의 TTF 파일

#### S3 구조

```
s3://your-bucket/
└── work-YYYYMMDD_HHMMSS/
    ├── font_imgs/
    │   ├── 가.png
    │   ├── 나.png
    │   └── ...
    └── output/
        └── MonoGlyph.ttf
```

#### 데이터베이스 저장

폰트 생성 완료 후 `run_summary.json`을 기반으로 다음 정보를 API 엔드포인트로 전송합니다:

- 프롬프트 및 작업 디렉토리 정보
- S3에 업로드된 이미지 URL 목록
- TTF 파일의 S3 public URL
- 생성 시간 및 각 단계별 실행 정보

**참고**: S3 업로드 기능을 사용하려면 `.env` 파일에 AWS 자격 증명과 S3 버킷 정보를 설정해야 합니다.

## 📁 출력 구조

전체 파이프라인 실행 시 다음과 같은 구조로 결과가 생성됩니다:

```
work-YYYYMMDD_HHMMSS/
├── font_imgs/          # 1단계: GPT 생성 이미지
│   └── 프롬프트명/
│       ├── 가.png
│       ├── 나.png
│       └── ...
├── output/             # 4단계: 최종 TTF 파일
│   └── MonoGlyph.ttf
├── logs/               # 각 단계별 로그
│   ├── 이미지 생성 (GPT_API).stdout.log
│   ├── 이미지 전처리 (pre-processing).stdout.log
│   └── ...
└── run_summary.json    # 실행 요약 정보
```

## 🔧 문제 해결

### 자주 발생하는 문제들

#### 1. OpenAI API 오류

```
[ERROR] API 키를 찾지 못했습니다.
```

**해결방법**: `.env` 파일에 올바른 API 키가 설정되어 있는지 확인

#### 2. FontForge 명령어를 찾을 수 없음

```
fontforge: command not found
```

**해결방법**: FontForge가 올바르게 설치되고 PATH에 추가되었는지 확인

#### 3. 한글 인코딩 문제

**해결방법**:

- 시스템 로케일을 UTF-8로 설정
- Python 환경변수 `PYTHONIOENCODING=utf-8` 설정

### 로그 확인

각 단계별 상세 로그는 `logs/` 폴더에서 확인할 수 있습니다:

- `*.stdout.log`: 정상 출력 로그
- `*.stderr.log`: 오류 로그

## 📝 라이선스

이 프로젝트의 라이선스 정보는 `LICENSE` 파일을 참조하세요.

## 🤝 기여하기

버그 리포트나 기능 제안은 GitHub Issues를 통해 제출해 주세요.

---

**참고**: 이 시스템은 AI 모델을 사용하므로 생성 결과가 매번 다를 수 있습니다.
