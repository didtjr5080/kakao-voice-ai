"""경량 음성 합성 (pyttsx3)"""

import pyttsx3
import tempfile
import os
from typing import Optional

class LightweightTTS:
    """경량 음성 합성 엔진 (오프라인, VRAM 0MB)"""
    
    def __init__(self, rate: int = 150, volume: float = 1.0):
        """
        Args:
            rate: 말하기 속도 (기본 150)
            volume: 볼륨 (0.0 ~ 1.0)
        """
        self.engine = None
        self.rate = rate
        self.volume = volume
    
    def load(self):
        """TTS 엔진 로드"""
        if self.engine is None:
            print("🔊 TTS 엔진 로딩...")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.rate)
            self.engine.setProperty('volume', self.volume)
            
            # 한국어 음성 찾기
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if 'korean' in voice.name.lower() or 'korea' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
            
            print("✅ TTS 준비 완료")
    
    def speak(self, text: str, save_path: Optional[str] = None) -> str:
        """텍스트를 음성으로 변환"""
        if self.engine is None:
            self.load()
        
        if save_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            save_path = temp_file.name
            temp_file.close()
        
        try:
            self.engine.save_to_file(text, save_path)
            self.engine.runAndWait()
            
            print(f"🔊 음성 생성: {text[:30]}...")
            return save_path
            
        except Exception as e:
            print(f"❌ TTS 실패: {e}")
            return None
