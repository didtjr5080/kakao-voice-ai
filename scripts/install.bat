@echo off
echo 카카오톡 음성 AI 설치 시작

REM Python 버전 확인
python --version

REM 가상환경 생성
echo 가상환경 생성 중...
python -m venv venv

REM 가상환경 활성화
call venv\Scripts\activate

REM pip 업그레이드
pip install --upgrade pip

REM 패키지 설치
echo 패키지 설치 중...
pip install -r requirements.txt

echo.
echo 설치 완료!
echo.
echo 다음 단계:
echo 1. 가상환경 활성화: venv\Scripts\activate
echo 2. 학습: python train.py --file your_kakao.txt --user "양석"
echo 3. 실행: python app.py
pause
