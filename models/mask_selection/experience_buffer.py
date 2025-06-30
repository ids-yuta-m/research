import random
from collections import deque

class ExperienceBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)
    
    def append(self, experience):
        """経験を追加"""
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """バッファからランダムにバッチサイズ分の経験をサンプリング"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)