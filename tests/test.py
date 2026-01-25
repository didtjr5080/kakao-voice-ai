import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.model import LightweightChatbot

print("="*60)
print("Chat Test with 양석 Model")
print("="*60)

print("\nLoading model...")
chatbot = LightweightChatbot(load_in_4bit=False)  # CPU 모드
chatbot.load_model("./models/kakao_model", use_lora=True)

print("\n=== Testing Responses ===\n")

test_inputs = [
    "안녕",
    "뭐해?",
    "ㅋㅋㅋ",
    "오늘 뭐 먹을까?",
    "?"
]

for inp in test_inputs:
    print(f"User: {inp}")
    response = chatbot.generate_response(inp)
    print(f"AI:   {response}")
    print("-" * 60)

print("\n✅ Test Complete!")