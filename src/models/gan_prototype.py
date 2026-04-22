import torch
import torch.nn as nn

# 1. 偽物を作る「Generator」
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh() # スペクトルデータの正規化範囲に合わせる
        )

    def forward(self, x):
        return self.main(x)

# 2. 本物を見破る「Discriminator」
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # 0(偽物)か1(本物)を出す
        )

    def forward(self, x):
        return self.main(x)
    

# src/models/gan_prototype.py の末尾に追記するイメージ
def compute_loss(discriminator, real_data, fake_data, criterion):
    """
    鑑定士（D）と偽造犯（G）の「悔しさ」を計算する
    """
    # 本物に対する判定結果
    real_preds = discriminator(real_data)
    real_loss = criterion(real_preds, torch.ones_like(real_preds))

    # 偽物に対する判定結果
    fake_preds = discriminator(fake_data)
    fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

    return real_loss + fake_loss


import torch.optim as optim

def get_optimizers(generator, discriminator, lr=0.0002):
    """
    偽造犯(G)と鑑定士(D)それぞれの「学習の歩幅」を決める
    """
    # G用の最適化ツール（ベータ値などはGANの論文で推奨される設定）
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # D用の最適化ツール
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    return g_optimizer, d_optimizer