"""통합 음성 대화 AI"""

from .parser import KakaoTalkParser
from .model import LightweightChatbot
from .stt import LightweightSTT
from .tts import LightweightTTS
from typing import Tuple, List, Dict

class VoiceConversationAI:
    """음성 대화 AI 통합 클래스 (STT → Chatbot → TTS)"""
    
    def __init__(
        self,
        model_name: str = "skt/kogpt2-base-v2",
        stt_model: str = "tiny",
        tts_rate: int = 150
    ):
        self.parser = KakaoTalkParser()
        self.chatbot = LightweightChatbot(model_name=model_name)
        self.stt = LightweightSTT(model_size=stt_model)
        self.tts = LightweightTTS(rate=tts_rate)
        
        self.is_trained = False
    
    def train_from_kakao(
        self,
        file_path: str,
        target_username: str,
        output_dir: str = "./models/kakao-chatbot",
        epochs: int = 3,
        context_window: int = 1
    ) -> str:
        """카카오톡 파일로부터 학습"""
        
        print("\n" + "="*70)
        print("🎙️ 음성 대화 AI 학습 시작")
        print("="*70)
        
        print("\n📚 Step 1: 카카오톡 대화 파싱...")
        messages = self.parser.parse_file(file_path)
        
        if not messages:
            raise ValueError("파싱된 메시지가 없습니다.")
        
        stats = self.parser.get_user_stats(messages)
        print(f"\n   대화 참여자:")
        for user, count in stats.items():
            print(f"   - {user}: {count}개 메시지")
        
        print(f"\n📊 Step 2: '{target_username}' 학습 데이터 생성...")
        training_pairs = self.parser.create_training_pairs(
            messages, 
            target_username,
            context_window
        )
        
        if not training_pairs:
            raise ValueError(f"'{target_username}' 사용자의 응답을 찾을 수 없습니다.")
        
        print(f"   ✓ {len(training_pairs)}개 학습 쌍 생성")
        
        print("\n   📝 학습 데이터 샘플:")
        for i, (inp, out) in enumerate(training_pairs[:3], 1):
            print(f"   {i}. 입력: {inp}")
            print(f"      응답: {out}")
        
        print(f"\n🤖 Step 3: 모델 학습...")
        model_path = self.chatbot.train(
            training_pairs=training_pairs,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=1,
            use_lora=True
        )
        
        self.is_trained = True
        
        print(f"\n🎤 Step 4: 음성 시스템 로딩...")
        self.stt.load()
        self.tts.load()
        
        print("\n" + "="*70)
        print("✅ 모든 시스템 준비 완료!")
        print("="*70 + "\n")
        
        return model_path
    
    def load_trained_model(self, model_path: str):
        """학습된 모델 로드"""
        print(f"📂 학습된 모델 로딩: {model_path}")
        self.chatbot.load_model(model_path, use_lora=True)
        self.stt.load()
        self.tts.load()
        self.is_trained = True
        print("✅ 모든 시스템 준비 완료!")
    
    def voice_chat(self, audio_path: str) -> Tuple[str, str, str]:
        """음성 입력 → AI 응답 (음성)"""
        
        if not self.is_trained:
            return "⚠️ 모델을 먼저 학습하거나 로드해주세요.", "", None
        
        user_text = self.stt.transcribe(audio_path)
        
        if user_text.startswith("[음성 인식 실패"):
            return user_text, "음성 인식에 실패했습니다.", None
        
        ai_response = self.chatbot.generate_response(user_text)
        ai_audio_path = self.tts.speak(ai_response)
        
        return user_text, ai_response, ai_audio_path
    
    def text_chat(self, user_text: str) -> Tuple[str, str]:
        """텍스트 입력 → AI 응답 (음성)"""
        
        if not self.is_trained:
            return "⚠️ 모델을 먼저 학습하거나 로드해주세요.", None
        
        ai_response = self.chatbot.generate_response(user_text)
        ai_audio_path = self.tts.speak(ai_response)
        
        return ai_response, ai_audio_path
