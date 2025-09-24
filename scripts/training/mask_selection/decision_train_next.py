import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.data import Dataset
import scipy.io as scio
import time
import datetime
import os
import numpy as np
from torch.autograd import Variable
#import matplotlib.pyplot as plt
import json
import re
#from tqdm import tqdm

class Imgdataset(Dataset):
    def __init__(self, path):
        super(Imgdataset, self).__init__()
        self.data = []
        if os.path.exists(path):
            groung_truth_path = os.path.join(path, 'Train_frames_8x8')

            if os.path.exists(groung_truth_path):
                groung_truth_files = os.listdir(groung_truth_path)
                self.data = [{'groung_truth': os.path.join(groung_truth_path, fname), 'id': os.path.splitext(fname)[0]}
                             for fname in groung_truth_files]
            else:
                raise FileNotFoundError('Ground truth path does not exist!')
        else:
            raise FileNotFoundError('Provided path does not exist!')

    def __getitem__(self, index):
        groung_truth = self.data[index]["groung_truth"]
        vid_id = self.data[index]["id"]

        gt = scio.loadmat(groung_truth)
        if "patch" in gt:
            gt = torch.from_numpy(gt['patch'])
        elif "patch_save_gray" in gt:
            gt = torch.from_numpy(gt['patch_save_gray'])
        elif "p1" in gt:
            gt = torch.from_numpy(gt['p1'] / 255)
        elif "p2" in gt:
            gt = torch.from_numpy(gt['p2'] / 255)
        elif "p3" in gt:
            gt = torch.from_numpy(gt['p3'] / 255)

        return gt, str(vid_id)

    def __len__(self):
        return len(self.data)
    
class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

class Unet(nn.Module):

    def __init__(self,in_ch, out_ch):
        super(Unet, self).__init__()
                
        self.dconv_down1 = double_conv(in_ch, 32)
        self.dconv_down2 = double_conv(32, 64)

        self.maxpool = nn.MaxPool2d(2)
    
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.dconv_up = double_conv(32 + 32, 32)
        
        self.conv_last = nn.Conv2d(32, out_ch, 1)
        self.afn_last = nn.Tanh()
        
        
    def forward(self, x):
        inputs = x
        
        # エンコーダーパス
        conv1 = self.dconv_down1(x)      # 8x8 -> 8x8
        x = self.maxpool(conv1)          # 8x8 -> 4x4
        
        # ボトルネック
        conv2 = self.dconv_down2(x)      # 4x4 -> 4x4
        
        # デコーダーパス
        x = self.upsample(conv2)         # 4x4 -> 8x8
        x = torch.cat([x, conv1], dim=1) # スキップ接続
        x = self.dconv_up(x)
        
        # 出力
        x = self.conv_last(x)
        x = self.afn_last(x)
        out = x + inputs
        
        return out

class BinarizeHadamardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        tmp_zero = torch.zeros(weight.shape).to(weight.device)
        tmp_one = torch.ones(weight.shape).to(weight.device)
        weight_b = torch.where(weight>0, tmp_one, tmp_zero)
        output = input * weight_b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        tmp_zero = torch.zeros(weight.shape).to(weight.device)
        tmp_one = torch.ones(weight.shape).to(weight.device)
        weight_b = torch.where(weight>0, tmp_one, tmp_zero)
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * weight_b
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output * input
        return grad_input, grad_weight

class LearnableMask(nn.Module):
    def __init__(self, t=16, s=8):
        super().__init__()
        self.t = t
        self.s = s
        self.weight = nn.Parameter(torch.Tensor(t, s, s))
        self.reset_parameters()

    def reset_parameters(self):
        self.stdv = torch.sqrt(torch.tensor(1.5 / (self.s * self.s * self.t)))
        self.weight.data.uniform_(-self.stdv, self.stdv)

    def forward(self, input):
        return BinarizeHadamardFunction.apply(input, self.weight)

    def get_binary_mask(self):
        with torch.no_grad():
            return torch.where(self.weight > 0, 
                             torch.ones_like(self.weight), 
                             torch.zeros_like(self.weight))

#再構成モデル　DMM
class ADMM_net(nn.Module):

    def __init__(self, t=16, s=8):
        super(ADMM_net, self).__init__()
        self.mask = LearnableMask(t=t, s=s)
        self.unet1 = Unet(16, 16)
        self.unet2 = Unet(16, 16)
        self.unet3 = Unet(16, 16)
        self.unet4 = Unet(16, 16)
        self.unet5 = Unet(16, 16)
        # self.unet6 = Unet(16, 16)
        # self.unet7 = Unet(16, 16)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        # self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        # Generate measurement using learnable mask
        maskt = self.mask(x)
        y = torch.sum(maskt, dim=1)
        
        # Get binary mask for reconstruction
        binary_mask = self.mask.get_binary_mask()
        Phi = binary_mask.expand([x.shape[0], 16, 8, 8])
        Phi_s = torch.sum(binary_mask, dim=0).expand([x.shape[0], 8, 8])

        # 9 stages of reconstruction
        x_list = []
        theta = self.At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        theta = self.unet1(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        theta = self.unet2(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        theta = self.unet3(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        theta = self.unet4(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        theta = self.unet5(x1)
        b = b- (x-theta)
        x_list.append(theta)
        # yb = self.A(theta+b,Phi)
        # x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        # x1 = x-b
        # theta = self.unet6(x1)
        # b = b- (x-theta)
        # x_list.append(theta)
        # yb = self.A(theta+b,Phi)
        # x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        # x1 = x-b
        # theta = self.unet7(x1)
        # b = b- (x-theta)
        # x_list.append(theta)
        

        return x_list
    
    def A(self, x,Phi):
        temp = x*Phi
        y = torch.sum(temp,1)
        return y

    def At(self, y,Phi):
        temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1], 1,1)
        x = temp*Phi
        return x
    
class DecisionNet(nn.Module):
    def __init__(self, num_classes=16):
        super(DecisionNet, self).__init__()
        # 入力層でのnormalizationを追加
        self.input_norm = nn.InstanceNorm2d(16)
        
        self.features = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # BatchNormを後ろに移動
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
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.001, epsilon_decay=0.995):
        super().__init__(num_classes)
        self.num_classes = num_classes  # 明示的に num_classes を保存
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

class DecisionNetWithExpReplay(DecisionNetWithEpsilonGreedy):
    def __init__(self, num_classes=16, epsilon_start=1.0, epsilon_end=0.01, 
                 epsilon_decay=0.995, buffer_size=30000):
        super().__init__(num_classes, epsilon_start, epsilon_end, epsilon_decay)
        self.experience_buffer = ExperienceBuffer(buffer_size)

def train_decision_net(recon_nets, train_loader, num_epochs=10, lr=1e-3, 
                      replay_batch_size=32, min_experiences=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    decision_net = DecisionNetWithExpReplay().to(device)
    optimizer = torch.optim.AdamW(decision_net.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=1, eta_min=1e-7
    )
    
    reference_net = DecisionNetWithExpReplay().to(device)
    reference_state = {
        k: v.clone() for k, v in decision_net.state_dict().items()
    }
    reference_net.load_state_dict(reference_state)
    reference_net.eval()
    
    psnr_results = np.load("./recon_results_next.npz")

        
    best_expected_psnr = float('-inf')
    best_model_state = None

    for epoch in range(num_epochs):
        decision_net.train()
        epoch_loss = 0.0
        batch_num = 0
        sum_expected_psnr = 0
        
        for batch_idx, (video_batch, ids) in enumerate(train_loader):
            gt = Variable(video_batch)
            gt = gt.to(device).float()
            gt_idx = None
            selected_recon_net = None
            prev_recon = None

            filename = ids[0]
            num_frames = gt.shape[1]
            for idx in range(int(num_frames/16)):
                gt_idx = gt[:, idx*16:(idx+1)*16, :, :]
                if idx == 0:
                    selected_recon_net = recon_nets[np.random.randint(0, 16)]
                    reconstructions = selected_recon_net(gt_idx)
                    prev_recon = reconstructions[-1]
                else:
                    filename_idx = filename + f"_{idx}"
                    batch_psnrs = torch.from_numpy(psnr_results[filename_idx]).float().requires_grad_(True).to(device)
                    prev_recon.requires_grad_(True)
                    # reference_netでの予測（探索なし）
                    with torch.no_grad():
                        reference_probs = reference_net(prev_recon)
                        reference_expected_psnrs = torch.sum(batch_psnrs * reference_probs, dim=1)
                    # 現在のネットワークでの予測(epsilon greedyあり)
                    current_probs = decision_net.forward_with_exploration(prev_recon, training=True)
                    current_expected_psnrs = torch.sum(batch_psnrs * current_probs, dim=1)

                    # improvementsを報酬として使用
                    improvements = current_expected_psnrs - reference_expected_psnrs
                    

                    experience = {
                        'state': prev_recon.cpu(),
                        'action_probs': current_probs[0].detach().cpu(),
                        'reward': improvements[0].item(),  # improvementsを報酬として使用
                        'psnr_values': batch_psnrs.cpu(),
                        'reference_psnr': reference_expected_psnrs[0].cpu()
                    }
                    decision_net.experience_buffer.append(experience)

                    selected_recon_net = recon_nets[torch.argmax(current_probs).item()]
                    reconstructions = selected_recon_net(gt_idx)
                    prev_recon = reconstructions[-1]

                    # Experience Replayによる学習
                    if len(decision_net.experience_buffer) >= min_experiences and epoch > 0:
                        experiences = decision_net.experience_buffer.sample(replay_batch_size)
                        
                        # 経験バッチの処理
                        replay_states = torch.stack([exp['state'] for exp in experiences]).to(device)
                        replay_psnrs = torch.stack([exp['psnr_values'] for exp in experiences]).to(device)
                        replay_reference_psnrs = torch.tensor([exp['reference_psnr'] for exp in experiences]).to(device)
                        
                        # リプレイデータでの予測
                        replay_states = replay_states.squeeze(1)
                        replay_current_probs = decision_net(replay_states)

                        replay_expected_psnrs = torch.sum(replay_psnrs * replay_current_probs, dim=1)
                        
                        # 元の損失関数の形式を維持
                        if epoch == 0:
                            loss = -torch.mean(current_expected_psnrs)
                        else:
                            improvements = replay_expected_psnrs - replay_reference_psnrs
                            loss = -torch.mean(improvements)

                        reference_state = {
                            k: v.clone() for k, v in decision_net.state_dict().items()
                        }
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(decision_net.parameters(), max_norm=1.0)
                        optimizer.step()


            reference_net.load_state_dict(reference_state)
            reference_net.eval()

            sum_expected_psnr += current_expected_psnrs.mean()
            batch_num += 1
            
        
        scheduler.step()
        current_epsilon = decision_net.update_epsilon()
        current_expected_psnr = sum_expected_psnr / batch_num
        
        print(f"Epoch {epoch}")
        print(f"Epsilon: {current_epsilon:.4f}")
        print(f"Expected PSNR: {current_expected_psnr}")
        #print(f"Buffer size: {len(decision_net.experience_buffer)}")
        
        # エポック終了時の評価
        with torch.no_grad():
            # evaluate_decision_net(decision_net, 
            #                    test_data_path="./data/valdata_gray_8x8_2", 
            #                    json_path="./top3_model_idx.json")
            
            if current_expected_psnr > best_expected_psnr:
                best_expected_psnr = current_expected_psnr
                best_model_state = {
                    'state_dict': decision_net.state_dict(),
                    'epoch': epoch,
                    'expected_psnr': current_expected_psnr
                }
        
        # if epoch % 6 == 0:
        #     lr = lr * 0.95
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        if epoch % 100 == 0 and epoch > 0:
            if best_model_state is not None:
                save_path = f'./select_models_next/select_model_{best_model_state["epoch"]}_{best_model_state["expected_psnr"]:.4f}.pth'
                torch.save(best_model_state, save_path)
    
    return decision_net


def define_recon_nets(recon_net_path_list):
    recon_nets = []

    for i in range(len(recon_net_path_list)):
        recon_nets.append(create_recon_net(recon_net_path_list[i]))
    return recon_nets

def create_recon_net(model_path):
    network = ADMM_net().cuda()
    network = torch.load(model_path, map_location=torch.device('cpu'), weights_only = False)
    network = network.cuda()

    stop_grads(network)

    return network

def stop_grads(network):
    # マスクのパラメータを凍結
    network.mask.requires_grad_(False)
    # 再構成部分のパラメータを更新可能に
    network.unet1.requires_grad_(False)
    network.unet2.requires_grad_(False)
    network.unet3.requires_grad_(False)
    network.unet4.requires_grad_(False)
    network.unet5.requires_grad_(False)
    # network.unet6.requires_grad_(False)
    # network.unet7.requires_grad_(False)
    network.gamma1.requires_grad_(False)
    network.gamma2.requires_grad_(False)
    network.gamma3.requires_grad_(False)
    network.gamma4.requires_grad_(False)
    network.gamma5.requires_grad_(False)
    # network.gamma6.requires_grad_(False)
    # network.gamma7.requires_grad_(False)



# 学習の実行
def main():
    num_candidates = 16
    recon_net_path_list = [filename for filename in os.listdir("./masks/16_models")]
    recon_nets = define_recon_nets(recon_net_path_list)

    dataset = Imgdataset("./data/")
    train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    decision_net = train_decision_net(recon_nets, train_loader, num_epochs=1500)


if __name__ == '__main__':
    main()