#!/bin/bash
echo "🚀 카카오톡 음성 AI 설치 시작"

# Python 버전 확인
python --version || python3 --version

# 가상환경 생성
echo "📦 가상환경 생성 중..."
python -m venv venv || python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate

# pip 업그레이드
pip install --upgrade pip

# 패키지 설치
echo "📥 패키지 설치 중..."
pip install -r requirements.txt

echo ""
echo "✅ 설치 완료!"
echo ""
echo "🎯 다음 단계:"
echo "1. 가상환경 활성화: source venv/bin/activate"
echo "2. 학습: python train.py --file your_kakao.txt --user '양석'"
echo "3. 실행: python app.py"
