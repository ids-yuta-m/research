import torch
import torch.nn as nn
import torch.nn.functional as F

class DecisionNet(nn.Module):
    def __init__(self, num_classes=16):
        super(DecisionNet, self).__init__()
        # 入力層でのnormalizationを追加
        self.input_norm = nn.InstanceNorm2d(16)
        
        self.features = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        
        # 重みの初期化を改善
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DecisionNetWithEpsilonGreedy(DecisionNet):
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.99):
        super().__init__(num_classes)
        self.num_classes = num_classes
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
    def forward_with_exploration(self, x, training=True):
        device = x.device
        batch_size = x.size(0)
        
        # 通常の順伝播を実行
        original_probs = super().forward(x)
        
        if not training or torch.rand(1).item() > self.epsilon:
            # 活用: モデルの予測をそのまま使用
            return original_probs
        else:
            # 探索: ランダムな確率分布を生成
            random_probs = torch.zeros_like(original_probs)
            for i in range(batch_size):
                # 各バッチに対してランダムな確率分布を生成
                random_dist = torch.zeros(self.num_classes, device=device)
                random_dist[torch.randint(0, self.num_classes, (1,))] = 1.0
                random_probs[i] = random_dist
            return random_probs
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

class DecisionNetWithExpReplay(DecisionNetWithEpsilonGreedy):
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.99, buffer_size=30000):
        super().__init__(num_classes, epsilon_start, epsilon_end, epsilon_decay)
        # ExperienceBufferはimportされる
        from models.mask_selection.experience_buffer import ExperienceBuffer
        self.experience_buffer = ExperienceBuffer(buffer_size)