"""CLI 학습 스크립트"""

import argparse
from src.voice_ai import VoiceConversationAI

def main():
    parser = argparse.ArgumentParser(description="카카오톡 대화로 AI 학습")
    parser.add_argument("--file", required=True, help="카카오톡 대화 파일 경로")
    parser.add_argument("--user", required=True, help="학습할 사용자 이름")
    parser.add_argument("--output", default="./models/kakao-chatbot", help="모델 저장 경로")
    parser.add_argument("--epochs", type=int, default=3, help="학습 에폭 수")
    parser.add_argument("--context", type=int, default=1, help="컨텍스트 윈도우")
    parser.add_argument("--model", default="skt/kogpt2-base-v2", help="기본 모델")
    
    args = parser.parse_args()
    
    ai = VoiceConversationAI(model_name=args.model)
    
    model_path = ai.train_from_kakao(
        file_path=args.file,
        target_username=args.user,
        output_dir=args.output,
        epochs=args.epochs,
        context_window=args.context
    )
    
    print(f"\n✅ 학습 완료! 모델 저장: {model_path}")
    print(f"\n실행: python app.py")

if __name__ == "__main__":
    main()
