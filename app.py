"""Gradio 웹 인터페이스"""

import gradio as gr
import os
import torch
from src.voice_ai import VoiceConversationAI

# 전역 AI 인스턴스
voice_ai = VoiceConversationAI()

def get_vram_usage():
    """VRAM 사용량 확인"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"VRAM 사용: {allocated:.2f} GB (캐시: {cached:.2f} GB)"
    return "GPU 없음 (CPU 모드)"

def train_model(file, username, epochs, context_window):
    """학습 핸들러"""
    if file is None:
        return "⚠️ 카카오톡 파일을 업로드해주세요."
    if not username:
        return "⚠️ 사용자 이름을 입력해주세요."
    
    try:
        model_path = voice_ai.train_from_kakao(
            file.name,
            username.strip(),
            epochs=int(epochs),
            context_window=int(context_window)
        )
        
        vram_info = get_vram_usage()
        
        return f"""✅ '{username}' 학습 완료!

📍 모델 저장: {model_path}
💾 {vram_info}

이제 '음성 대화' 탭에서 대화를 시작할 수 있습니다!
"""
    except Exception as e:
        return f"❌ 학습 실패:\n{str(e)}"

def load_model(model_path):
    """모델 로드 핸들러"""
    try:
        voice_ai.load_trained_model(model_path)
        vram_info = get_vram_usage()
        return f"✅ 모델 로드 완료!\n💾 {vram_info}"
    except Exception as e:
        return f"❌ 로드 실패: {str(e)}"

def voice_conversation(audio, history):
    """음성 대화 핸들러"""
    if audio is None:
        return history, None, ""
    
    user_text, ai_text, ai_audio = voice_ai.voice_chat(audio)
    
    history.append([f"🎤 {user_text}", f"🔊 {ai_text}"])
    
    return history, ai_audio, user_text

def text_conversation(message, history):
    """텍스트 대화 핸들러"""
    if not message:
        return history, None
    
    ai_text, ai_audio = voice_ai.text_chat(message)
    
    history.append([message, f"🔊 {ai_text}"])
    
    return history, ai_audio

# Gradio 앱 구성
with gr.Blocks(title="경량 음성 대화 AI", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🎙️ 카카오톡 학습 음성 대화 AI
    ### VRAM 1.5GB 경량 버전 - 로컬 실행 최적화
    
    마이크로 말하면 AI가 듣고 생각하고 답합니다!
    """)
    
    with gr.Tab("🎙️ 음성 대화"):
        gr.Markdown("## 🎤 마이크로 AI와 대화하세요")
        
        chatbot = gr.Chatbot(label="대화 기록", height=400)
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    source="microphone",
                    type="filepath",
                    label="🎤 마이크 (녹음 시작)"
                )
                recognized_text = gr.Textbox(
                    label="인식된 음성",
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="🔊 AI 응답 음성",
                    autoplay=True
                )
        
        audio_input.change(
            voice_conversation,
            [audio_input, chatbot],
            [chatbot, audio_output, recognized_text]
        )
    
    with gr.Tab("💬 텍스트 테스트"):
        gr.Markdown("### 텍스트로 빠르게 테스트")
        
        test_chatbot = gr.Chatbot(label="대화", height=400)
        
        with gr.Row():
            text_input = gr.Textbox(
                label="메시지",
                placeholder="메시지를 입력하세요...",
                scale=4
            )
            send_btn = gr.Button("전송", scale=1, variant="primary")
        
        test_audio = gr.Audio(label="🔊 AI 응답 음성", autoplay=True)
        
        send_btn.click(
            text_conversation,
            [text_input, test_chatbot],
            [test_chatbot, test_audio]
        ).then(lambda: "", None, text_input)
        
        text_input.submit(
            text_conversation,
            [text_input, test_chatbot],
            [test_chatbot, test_audio]
        ).then(lambda: "", None, text_input)
    
    with gr.Tab("📚 모델 학습"):
        gr.Markdown("""
        ### 카카오톡 대화로 AI 학습하기
        
        **카카오톡 대화 내보내기:**
        1. 카카오톡 앱 → 대화방 열기
        2. 우측 상단 `≡` → **대화 내보내기**
        3. **텍스트만** 선택 → 저장
        4. 아래에 업로드
        """)
        
        with gr.Row():
            with gr.Column():
                kakao_file = gr.File(
                    label="📂 카카오톡 대화 파일 (.txt)",
                    file_types=[".txt"]
                )
                target_user = gr.Textbox(
                    label="🎯 학습할 사용자 이름",
                    placeholder="예: 양석"
                )
            
            with gr.Column():
                epochs_slider = gr.Slider(
                    minimum=1, maximum=10, value=3, step=1,
                    label="학습 에폭 수"
                )
                context_slider = gr.Slider(
                    minimum=1, maximum=5, value=1, step=1,
                    label="컨텍스트 윈도우"
                )
        
        train_btn = gr.Button("🚀 학습 시작", variant="primary", size="lg")
        train_status = gr.Textbox(label="학습 상태", lines=10, interactive=False)
        
        train_btn.click(
            train_model,
            [kakao_file, target_user, epochs_slider, context_slider],
            train_status
        )
    
    with gr.Tab("⚙️ 설정"):
        gr.Markdown("### 기존 모델 불러오기")
        
        model_path_input = gr.Textbox(
            label="모델 경로",
            value="./models/kakao-chatbot"
        )
        load_btn = gr.Button("📂 모델 로드", variant="secondary")
        load_status = gr.Textbox(label="상태", lines=5)
        
        load_btn.click(load_model, model_path_input, load_status)
        
        gr.Markdown(f"""
        ---
        ### 📊 시스템 정보
        
        - **챗봇**: KoGPT-2 (4bit 양자화 + LoRA)
        - **STT**: Whisper Tiny (~400MB VRAM)
        - **TTS**: pyttsx3 (오프라인, CPU)
        - **현재**: {get_vram_usage()}
        
        ### 💾 메모리 사용량 목표
        - 학습 시: 최대 1.5GB VRAM
        - 추론 시: 최대 1GB VRAM
        """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
