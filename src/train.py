import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.models.gan_prototype import Generator, Discriminator, train_gan

# 1. 本物のデータを読み込む
df = pd.read_csv('data/raw/spectral_data.csv', encoding='shift_jis')

# 2. 【ブラッシュアップ】波形データ部分だけを抽出
# 先頭4列（IDや樹種など）を除外し、5列目以降の「波形数値」のみを取得します
# また、念のため数値を float32 に変換します
spectral_data = df.iloc[:, 4:].values.astype('float32')
real_data_tensor = torch.from_numpy(spectral_data)

# 3. データを小分けにして運ぶ「DataLoader」を作成
batch_size = 64
dataloader = DataLoader(
    TensorDataset(real_data_tensor),
    batch_size=batch_size,
    shuffle=True
)

# 4. エンジン（モデル）を起動
z_dim = 100
out_dim = spectral_data.shape[1]  # 正しい波形の次元数（列数）を自動取得
gen = Generator(z_dim, out_dim)
disc = Discriminator(out_dim)

# 5. 本番の修行（学習）開始！
train_gan(gen, disc, dataloader, epochs=50)