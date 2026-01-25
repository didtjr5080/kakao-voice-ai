"""경량 음성 인식 (Whisper Tiny)"""

import torch
import whisper
from typing import Optional

class LightweightSTT:
    """경량 음성 인식 엔진 (VRAM ~400MB)"""
    
    def __init__(self, model_size: str = "tiny"):
        """
        Args:
            model_size: tiny, base, small 중 선택 (tiny 권장)
        """
        self.model_size = model_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        """Whisper 모델 로드"""
        if self.model is None:
            print(f"🎤 Whisper {self.model_size} 모델 로딩...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"✅ STT 준비 완료 (Device: {self.device})")
            
            if self.device == "cuda":
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"💾 VRAM 사용량: {vram:.2f} GB")
    
    def transcribe(self, audio_path: str, language: str = "ko") -> str:
        """음성 파일을 텍스트로 변환"""
        if self.model is None:
            self.load()
        
        try:
            result = self.model.transcribe(
                audio_path,
                language=language,
                fp16=(self.device == "cuda")
            )
            
            text = result["text"].strip()
            print(f"🎤 인식: {text}")
            return text
            
        except Exception as e:
            error_msg = f"[음성 인식 실패: {str(e)}]"
            print(f"❌ {error_msg}")
            return error_msg
