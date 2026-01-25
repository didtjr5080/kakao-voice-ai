"""카카오톡 대화 파일 파서"""

import re
from typing import List, Dict, Tuple

class KakaoTalkParser:
    """카카오톡 대화 파일을 파싱하여 학습 데이터 생성"""
    
    def __init__(self):
        self.message_pattern = re.compile(r'\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)')
        self.date_pattern = re.compile(r'-{10,}.*-{10,}')
        self.system_keywords = [
            '님이.*초대했습니다', '님이 방장이 되어', 
            '님이 나갔습니다', '님이 들어왔습니다',
            '메시지가 삭제되었습니다', '저장한 날짜', '카카오톡 대화'
        ]
    
    def is_system_message(self, text: str) -> bool:
        """시스템 메시지 여부 확인"""
        for keyword in self.system_keywords:
            if re.search(keyword, text):
                return True
        return False
    
    def parse_file(self, file_path: str, encoding: str = 'utf-8') -> List[Dict[str, str]]:
        """카카오톡 파일 파싱"""
        messages = []
        
        with open(file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            if not line or self.date_pattern.match(line):
                continue
            
            if self.is_system_message(line):
                continue
            
            match = self.message_pattern.match(line)
            if match:
                username, timestamp, message = match.groups()
                message = message.strip()
                
                if message and message not in ['사진', '동영상', '이모티콘', '음성메시지', '파일']:
                    messages.append({
                        'username': username.strip(),
                        'timestamp': timestamp.strip(),
                        'message': message
                    })
        
        print(f"Parsed {len(messages)} messages")
        return messages
    
    def create_training_pairs(
        self, 
        messages: List[Dict], 
        target_username: str,
        context_window: int = 1
    ) -> List[Tuple[str, str]]:
        """학습용 (입력, 출력) 쌍 생성"""
        training_pairs = []
        
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]
            
            if next_msg['username'] == target_username:
                context_start = max(0, i - context_window + 1)
                context_messages = messages[context_start:i+1]
                
                if len(context_messages) == 1:
                    input_text = context_messages[0]['message']
                else:
                    input_text = ' '.join([m['message'] for m in context_messages])
                
                output_text = next_msg['message']
                training_pairs.append((input_text, output_text))
        
        print(f"Created {len(training_pairs)} training pairs")
        return training_pairs
    
    def get_user_stats(self, messages: List[Dict]) -> Dict[str, int]:
        """사용자별 메시지 수 통계"""
        stats = {}
        for msg in messages:
            username = msg['username']
            stats[username] = stats.get(username, 0) + 1
        return stats
