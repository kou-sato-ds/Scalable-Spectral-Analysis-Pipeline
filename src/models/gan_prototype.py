import torch
import torch.nn as nn
import torch.optim as optim

# デバイスの設定（GPUが使えるなら使う）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 偽物を作る「Generator」
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh() # スペクトルデータの正規化範囲（-1〜1）に合わせる
        )

    def forward(self, x):
        return self.main(x)

# 2. 本物を見破る「Discriminator」
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2), # GANではLeakyReLUが定石
            nn.Linear(256, 1),
            nn.Sigmoid() # 0(偽物)か1(本物)を確率で出す
        )

    def forward(self, x):
        return self.main(x)

def compute_loss(discriminator, real_data, fake_data, criterion):
    """
    鑑定士（D）の損失（本物を見逃した悔しさ + 偽物に騙された悔しさ）を計算
    """
    # 本物に対する判定
    real_preds = discriminator(real_data)
    real_loss = criterion(real_preds, torch.ones_like(real_preds))

    # 偽物に対する判定
    fake_preds = discriminator(fake_data)
    fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

    return real_loss + fake_loss

def get_optimizers(generator, discriminator, lr=0.0002):
    """
    GとDそれぞれの最適化ツールを用意
    """
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    return g_optimizer, d_optimizer

def train_gan(generator, discriminator, dataloader, epochs=10, z_dim=100, lr=0.0002):
    """
    GANの訓練メインループ
    """
    generator.to(device)
    discriminator.to(device)
    g_optimizer, d_optimizer = get_optimizers(generator, discriminator, lr)
    criterion = nn.BCELoss()

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        running_d_loss = 0.0
        running_g_loss = 0.0

        for i, real_data in enumerate(dataloader):
            # データの準備（本物データをデバイスへ転送）
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # --- 0. 準備：ひらめきの素（ノイズ）から偽物を作る ---
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_data = generator(noise)

            # --- 1. 鑑定士(D)の修行 ---
            d_optimizer.zero_grad()
            # Dの学習時はGの計算グラフを切り離す(.detach)
            d_loss = compute_loss(discriminator, real_data, fake_data.detach(), criterion)
            d_loss.backward()
            d_optimizer.step()

            # --- 2. 偽造犯(G)の修行 ---
            g_optimizer.zero_grad()
            # Gの目標は、Dに「本物(1)」と言わせること
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, torch.ones_like(outputs))
            g_loss.backward()
            g_optimizer.step()

            # ログ用
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        # エポックごとの進捗表示
        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {avg_d_loss:.4f}, Loss_G: {avg_g_loss:.4f}")

    print("Training finished!")