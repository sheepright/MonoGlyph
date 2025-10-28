# MonoGlyph #

## server.py 실행 ##
uvicorn server:app --host 0.0.0.0 --port 8000 --proxy-headers --log-level info

## main.py 실행 ##
python main.py -p "프롬프트 값" 

- 생성시 가중치 파일과 GPT API Key 필요 
- requirements.txt 버전 및 호환 문제 확인 필요, 추가 설치 필요한 라이브러리 확인