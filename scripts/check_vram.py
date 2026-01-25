"""VRAM 체크 유틸리티"""

import torch

def check_vram():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ CUDA 사용 가능")
        print(f"   GPU 개수: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            total_vram = props.total_memory / 1024**3
            print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   총 VRAM: {total_vram:.2f} GB")
            
            if total_vram < 1.5:
                print(f"   ⚠️ VRAM이 1.5GB 미만입니다. CPU 모드를 권장합니다.")
            else:
                print(f"   ✅ 충분한 VRAM")
    else:
        print("❌ CUDA 사용 불가 (CPU 모드로 실행됩니다)")

if __name__ == "__main__":
    check_vram()
