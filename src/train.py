import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from src.models.gan_prototype import Generator, Discriminator, train_gan

def main():
    # 0. 保存先の準備（ここが重要！）
    # プロジェクト直下の 'models' フォルダを絶対パスで指定するのね
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "models")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting process in: {base_dir}")
    print(f"Models will be saved to: {save_dir}")

    # 1. 本物のデータを読み込む
    # データパスも確実に通るように絶対パスで構築するのね
    data_path = os.path.join(base_dir, 'data', 'raw', 'spectral_data.csv')
    df = pd.read_csv(data_path, encoding='shift_jis')

    # 2. 波形データ部分だけを抽出
    spectral_data = df.iloc[:, 4:].values.astype('float32')
    real_data_tensor = torch.from_numpy(spectral_data)

    # 3. DataLoaderを作成
    batch_size = 64
    dataloader = DataLoader(
        TensorDataset(real_data_tensor),
        batch_size=batch_size,
        shuffle=True
    )

    # 4. モデルの初期化
    z_dim = 100
    out_dim = spectral_data.shape[1]
    gen = Generator(z_dim, out_dim)
    disc = Discriminator(out_dim)

    # 5. 修行（学習）開始！
    print("Starting training on cpu...")
    train_gan(gen, disc, dataloader, epochs=50)

    # 6. 【ここが勝利の鍵！】学習終わったモデルを確実に保存する
    model_path = os.path.join(save_dir, "gen_model_50.pth")
    torch.save(gen.state_dict(), model_path)
    
    if os.path.exists(model_path):
        print("-" * 30)
        print(f"✨ SUCCESS! ✨")
        print(f"Model saved at: {model_path}")
        print("-" * 30)
    else:
        print("❌ Error: Failed to save the model.")

if __name__ == "__main__":
    main()