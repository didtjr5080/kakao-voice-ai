# 🏗️ 시스템 아키텍처

## 전체 구조
음성 입력 → STT → 챗봇 → TTS → 음성 출력

## 메모리 사용량
- Whisper Tiny: 400MB VRAM
- KoGPT-2 (4bit): 600MB VRAM
- 총합: ~1GB VRAM

## 경량화 기법
- 4bit 양자화
- LoRA 파인튜닝
- Gradient Checkpointing
