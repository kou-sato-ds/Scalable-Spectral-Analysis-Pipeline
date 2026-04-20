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