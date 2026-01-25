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
        print(">>> Voice AI Training Started")
        print("="*70)
        
        print("\n[Step 1] Parsing KakaoTalk messages...")
        messages = self.parser.parse_file(file_path)
        
        if not messages:
            raise ValueError("No messages parsed")
        
        stats = self.parser.get_user_stats(messages)
        print(f"\nParticipants:")
        for user, count in stats.items():
            print(f"  - {user}: {count} messages")
        
        print(f"\n[Step 2] Creating training data for '{target_username}'...")
        training_pairs = self.parser.create_training_pairs(
            messages, 
            target_username,
            context_window
        )
        
        if not training_pairs:
            raise ValueError(f"No responses found for user '{target_username}'")
        
        print(f"  Created {len(training_pairs)} training pairs")
        
        print("\n  Sample data:")
        for i, (inp, out) in enumerate(training_pairs[:3], 1):
            print(f"  {i}. Input: {inp}")
            print(f"     Output: {out}")
        
        print(f"\n[Step 3] Training model...")
        model_path = self.chatbot.train(
            training_pairs=training_pairs,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=1,
            use_lora=True
        )
        
        self.is_trained = True
        
        print(f"\n[Step 4] Loading voice systems...")
        self.stt.load()
        self.tts.load()
        
        print("\n" + "="*70)
        print(">>> All systems ready!")
        print("="*70 + "\n")
        
        return model_path
    
    def load_trained_model(self, model_path: str):
        """학습된 모델 로드"""
        print(f"Loading trained model: {model_path}")
        self.chatbot.load_model(model_path, use_lora=True)
        self.stt.load()
        self.tts.load()
        self.is_trained = True
        print("All systems ready!")
    
    def voice_chat(self, audio_path: str) -> Tuple[str, str, str]:
        """음성 입력 → AI 응답 (음성)"""
        
        if not self.is_trained:
            return "Model not loaded. Please train or load first.", "", None
        
        user_text = self.stt.transcribe(audio_path)
        
        if user_text.startswith("["):
            return user_text, "Speech recognition failed.", None
        
        ai_response = self.chatbot.generate_response(user_text)
        ai_audio_path = self.tts.speak(ai_response)
        
        return user_text, ai_response, ai_audio_path
    
    def text_chat(self, user_text: str) -> Tuple[str, str]:
        """텍스트 입력 → AI 응답 (음성)"""
        
        if not self.is_trained:
            return "Model not loaded. Please train or load first.", None
        
        ai_response = self.chatbot.generate_response(user_text)
        ai_audio_path = self.tts.speak(ai_response)
        
        return ai_response, ai_audio_path
