import torch
import torch.nn as nn
import torch.optim as optim

# デバイスの設定（GPUが使えるなら優先的に使用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 偽物を作る「Generator（偽造犯）」
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512), # 層を一段厚くして表現力を向上
            nn.ReLU(inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh() # スペクトルデータの正規化範囲（-1〜1）に合わせる
        )

    def forward(self, x):
        return self.main(x)

# 2. 本物を見破る「Discriminator（鑑定士）」
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid() # 0(偽物)か1(本物)を確率で出力
        )

    def forward(self, x):
        return self.main(x)

def compute_loss(discriminator, real_data, fake_data, criterion):
    """
    鑑定士（D）の損失を計算
    """
    # 本物に対する判定（目標は1）
    real_preds = discriminator(real_data)
    real_loss = criterion(real_preds, torch.ones_like(real_preds))

    # 偽物に対する判定（目標は0）
    fake_preds = discriminator(fake_data)
    fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))

    return real_loss + fake_loss

def get_optimizers(generator, discriminator, lr=0.0002):
    """
    GとDそれぞれの最適化アルゴリズム（Adam）を設定
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

        for i, data in enumerate(dataloader):
            # --- 重要：DataLoaderのリスト/タプルからテンソルを取り出す ---
            # TensorDatasetを使用している場合、dataはリスト形式で返るため [0] で取得
            real_data = data[0].to(device) if isinstance(data, (list, tuple)) else data.to(device)
            batch_size = real_data.size(0)

            # --- 0. 準備：ノイズから偽物を作成 ---
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
            # Gの目標は、Dに「本物(1)」と判定させること
            outputs = discriminator(fake_data)
            g_loss = criterion(outputs, torch.ones_like(outputs))
            g_loss.backward()
            g_optimizer.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        # エポックごとの進捗表示
        avg_d_loss = running_d_loss / len(dataloader)
        avg_g_loss = running_g_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss_D: {avg_d_loss:.4f}, Loss_G: {avg_g_loss:.4f}")

    print("Training finished!")

# --- 火入れ式（動作確認用スクリプト） ---
def test_run():
    z_dim = 100
    out_dim = 512 # 実データの波形長に合わせた設定
   
    gen = Generator(z_dim, out_dim)
    disc = Discriminator(out_dim)

    # 模擬データ（5サンプル）
    dummy_real = torch.randn(5, out_dim)
    # 本番同様にTensorDataset形式の擬似ローダーを作成
    from torch.utils.data import DataLoader, TensorDataset
    dummy_loader = DataLoader(TensorDataset(dummy_real), batch_size=2)

    print("火入れ式を開始します...")
    train_gan(gen, disc, dummy_loader, epochs=1)
    print("無事に1エポック完走！命が宿りました。")

if __name__ == "__main__":
    test_run()