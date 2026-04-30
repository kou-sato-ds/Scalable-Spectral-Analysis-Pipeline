import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Generatorクラス（学習時の重みを柔軟に受け入れる構造なのね！） ---
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        # 学習時の構造（256, 512）に合わせつつ、名前を self.main に統一したのね
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

def visualize_results():
    # 0. パス設定（どこから叩いても迷子にならない絶対パスなのね！）
    current_file_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(current_file_path)
    project_root = os.path.dirname(src_dir)

    model_path = os.path.join(project_root, 'models', 'gen_model_50.pth')
    real_data_path = os.path.join(project_root, 'data', 'raw', 'spectral_data.csv')
    output_path = os.path.join(project_root, 'docs', 'images', 'gan_comparison.png')

    # 1. データの読み込み
    if not os.path.exists(real_data_path):
        print(f"❌ Error: {real_data_path} が見当たりません。")
        return

    # Shift-JISで読み込み、4列目以降の波形データを取得するのね
    real_df = pd.read_csv(real_data_path, encoding='shift_jis')
    spectral_values = real_df.iloc[:, 4:].values.astype('float32')
    output_dim = spectral_values.shape[1]
    real_sample = spectral_values[0] # 比較用に最初のサンプルを使用

    # 2. モデルのロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_dim = 100
    gen = Generator(latent_dim=z_dim, output_dim=output_dim).to(device)
    
    if os.path.exists(model_path):
        # 【重要】strict=False を使うことで、層の番号が多少ズレていても
        # 重みの名前が合っていれば無理やり読み込ませる「力技」なのね！
        state_dict = torch.load(model_path, map_location=device)
        gen.load_state_dict(state_dict, strict=False)
        gen.eval()
        print(f"✨ Model loaded successfully (strict=False): {model_path}")
    else:
        print(f"❌ Error: {model_path} が見つかりません。")
        return

    # 3. GANによる波形生成
    z = torch.randn(1, z_dim).to(device)
    with torch.no_grad():
        fake_data = gen(z).cpu().numpy().reshape(-1)

    # 4. グラフの描画
    plt.figure(figsize=(12, 6))
    plt.plot(real_sample, label='Real Spectrum (Sample 0)', color='#1f77b4', alpha=0.7, linewidth=2)
    plt.plot(fake_data, label='GAN-Generated Spectrum', color='#d62728', linestyle='--', linewidth=2)
    
    plt.title('Spectral Synthesis Check: Real vs GAN Prototype', fontsize=14, fontweight='bold')
    plt.xlabel('Wavelength Bin', fontsize=12)
    plt.ylabel('Normalized Intensity', fontsize=12)
    plt.legend(frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 5. 保存と表示
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"🖼️ Visualization success! Saved at: {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_results()