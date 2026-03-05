# プログラム詳細解説 / Detailed Code Explanation

## 目次 (Table of Contents)
1. [プロジェクト概要](#1-プロジェクト概要)
2. [ファイル構成](#2-ファイル構成)
3. [config.py — 設定ファイル](#3-configpy--設定ファイル)
4. [model.py — モデル定義](#4-modelpy--モデル定義)
5. [DG.py — カスタムデータジェネレーター](#5-dgpy--カスタムデータジェネレーター)
6. [utils.py — ユーティリティ関数](#6-utilspy--ユーティリティ関数)
7. [augment.py — データ拡張補助関数](#7-augmentpy--データ拡張補助関数)
8. [losses.py — 損失関数](#8-lossespy--損失関数)
9. [train.py — 学習・評価メインスクリプト](#9-trainpy--学習評価メインスクリプト)
10. [データの流れ（全体パイプライン）](#10-データの流れ全体パイプライン)
11. [学習・評価の実行方法](#11-学習評価の実行方法)

---

## 1. プロジェクト概要

このプロジェクトは、**Sentinel-1 (SAR) と Sentinel-2 (光学) 衛星画像の時系列データを組み合わせて、建物の高さマップを10m解像度で推定する**ディープラーニングモデルを実装したものです。

IEEE IGARSS 2023 の論文「A CNN Regression Model to Estimate Buildings Height Maps Using Sentinel-1 SAR and Sentinel-2 MSI Time Series」に対応するコードです。

### タスクの概要

| 項目 | 内容 |
|------|------|
| 入力 | Sentinel-1 SAR 画像 (3チャネル) + Sentinel-2 マルチスペクトル画像 (5チャネル) |
| 出力 | 建物高さマップ (1チャネル、ピクセルごとの高さ推定値) |
| 解像度 | 10m 空間解像度 |
| モデル | Multimodal Building Height Regression Network (MBHR-Net) |
| 評価指標 | RMSE (3.73m)、IoU (0.95)、R² (0.61) |
| 対象地域 | オランダの10都市 |

### ネットワーク構成の概要

```
Sentinel-1 (SAR) 画像
        ↓
  SAR エンコーダー (ResNet50ベース)
        ↓ [f1, f2, f3, f4]
        ↓                ← 要素ごとに加算 (Add)
  OPT エンコーダー (ResNet50ベース)
        ↑ [f1, f2, f3, f4]
Sentinel-2 (光学) 画像
        ↓
    デコーダー (U-Net ライク)
        ↓
  建物高さマップ (128×128×1)
```

---

## 2. ファイル構成

```
Building-Height-Estimation_SEN12_IGARSS23/
├── train.py              # メインの学習・評価スクリプト
├── code/
│   ├── config.py         # ハイパーパラメータ・パス設定
│   ├── model.py          # ResNet50エンコーダー、デコーダー、モデル定義
│   ├── DG.py             # カスタムデータジェネレーター
│   ├── utils.py          # データ読み込み・前処理関数
│   ├── augment.py        # データ前処理補助関数（utils.pyと類似）
│   └── losses.py         # カスタム損失関数（SSIM損失、エッジ損失など）
└── img/
    ├── Network_Architecture.JPG  # ネットワーク構成図
    └── Data_samples.JPG          # データサンプル画像
```

---

## 3. config.py — 設定ファイル

このファイルは、モデルの学習・推論に必要な**全てのハイパーパラメータとパス設定**を一元管理しています。

```python
from pathlib import Path

# --- ディレクトリパス ---
ROOT_PATH = Path("ROOT_PATH")           # プロジェクトルートパス
DATA_PATH = ROOT_PATH / 'DATA_PATH'    # データルートパス
S1_PATH   = DATA_PATH / 'S1'           # Sentinel-1 画像フォルダ
S2_PATH   = DATA_PATH / "S2"           # Sentinel-2 画像フォルダ
LABEL10_PATH = DATA_PATH / "LABEL_10"  # 建物高さラベルフォルダ（10m解像度）
WEIGHT_PATH = ROOT_PATH / 'WEIGHT_PATH'# モデル重みの保存先

# --- 画像・モデルサイズ ---
IMG_HEIGHT, IMG_WIDTH = 128, 128  # 入力画像サイズ (ピクセル)
s1_ch, s2_ch = 3, 5              # S1: 3チャネル (VV, VH, 派生), S2: 5チャネル (B4,B3,B2,B8,B10)
model_patch = 128                 # モデルのパッチサイズ

# --- 学習パラメータ ---
splits = 0.2            # 検証データの割合 (20%)
train_batchSize = 4     # 学習バッチサイズ
val_batchSize = 1       # 検証バッチサイズ
S2_MAX = 3000           # Sentinel-2 の正規化最大値
lr = 0.0001             # 初期学習率
maxDepthVal = 176.0     # 建物高さの最大値 (メートル)

# --- ラベルファイル名 ---
LABEL_fname = 'Filter_LABELS.csv'   # フィルタリング済みラベルファイル名

# --- 損失関数の重み ---
ssim_loss_weight = 0.4   # SSIM損失の重み
mse_loss_weight  = 0.6   # MSE損失の重み
l1_loss_weight   = 0.1   # L1損失の重み (参考値)
edge_loss_weight = 0.9   # エッジ損失の重み (参考値)
```

### ポイント
- **`IMG_HEIGHT, IMG_WIDTH = 128, 128`**: 全ての入力画像は 128×128 ピクセルにリサイズされます。
- **`s1_ch=3`**: SAR 画像は VV 偏波・VH 偏波の2バンドを読み込み、3チャネルのテンソルとして扱います（3チャネル目は0埋め）。
- **`s2_ch=5`**: 光学画像はバンド1, 2, 3, 7, 10の5バンドを使用します。
- **`maxDepthVal=176.0`**: SSIM 損失計算時のスケール係数（データセット内の最大建物高さに対応）。

---

## 4. model.py — モデル定義

最も重要なファイルで、**ネットワークの全構成要素**を定義しています。

### 4-1. Squeeze-and-Excitation (SE) ブロック

#### `squeeze_excite_block(input_tensor, shape, ratio=16)`
チャネル方向の注意機構 (Attention) を実装しています。

```
入力テンソル
    ↓ GlobalAveragePooling2D  # 空間情報を圧縮してチャネルごとのスカラーへ
    ↓ Reshape(1, 1, C)
    ↓ Dense(C // ratio, relu) # チャネル数を 1/16 に圧縮（情報の絞り込み）
    ↓ Dense(C, sigmoid)       # 元のチャネル数に復元（0〜1のゲートを生成）
    ↓ multiply(入力, ゲート)  # 重要なチャネルを強調、不要なチャネルを抑制
```

**参考論文**: [Squeeze and Excitation Networks (arXiv:1709.01507)](https://arxiv.org/abs/1709.01507)

#### `spatial_squeeze_excite_block(input_tensor)`
空間方向の注意機構を実装しています。

```
入力テンソル
    ↓ Conv2D(1, 1×1, sigmoid)  # 各位置の重要度マップを1チャネルで生成
    ↓ multiply(入力, 重要度マップ) # 重要な空間位置を強調
```

#### `channel_spatial_squeeze_excite(input_tensor, shape, ratio=16)`
チャネル SE と空間 SE を**両方適用して足し合わせる**複合注意機構です。デコーダーのスキップ接続後に使用されます。

**参考論文**: [Concurrent Spatial and Channel Squeeze & Excitation (arXiv:1803.02579)](https://arxiv.org/abs/1803.02579)

---

### 4-2. ResNet50 の基本ブロック

#### `identity_block(input_tensor, kernel_size, filters, stage, block)`
残差ブロック（ショートカット接続あり・次元変換なし）。

```
入力
├── メインパス:
│   Conv2D(1×1) → BN → ReLU
│   Conv2D(3×3) → BN → ReLU
│   Conv2D(1×1) → BN
└── ショートカット: 入力をそのまま加算
    ↓ Add → ReLU
```

#### `conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2))`
ショートカット側にも畳み込み層を持つ残差ブロック（次元変換あり）。ダウンサンプリング時に使用。

```
入力
├── メインパス:
│   Conv2D(1×1, stride=2) → BN → ReLU
│   Conv2D(3×3)           → BN → ReLU
│   Conv2D(1×1)           → BN
└── ショートカット:
    Conv2D(1×1, stride=2) → BN
    ↓ Add → ReLU
```

---

### 4-3. ResNet50 エンコーダー

#### `get_resnet50_encoder1(...)` と `get_resnet50_encoder2(...)`
どちらも **ResNet50 のエンコーダー部分**を構築します。`encoder1` は S1 (SAR) 画像用、`encoder2` は S2 (光学) 画像用です。レイヤー名の衝突を避けるため、`encoder2` では `conv21`, `stage=22, 23, 24` のように番号がずらしてあります。

**出力する特徴マップ (スキップ接続用)**:
| 特徴マップ | サイズ (入力128×128の場合) | フィルター数 |
|-----------|--------------------------|------------|
| f1 | 64 × 64 | 64 |
| f2 | 32 × 32 | 256 |
| f3 | 16 × 16 | 512 |
| f4 | 8 × 8 | 1024 |
| f5 | 4 × 4 | 2048 |

`encoder2` は学習済み ImageNet 重みを自動ダウンロードして初期化します（転移学習）。

---

### 4-4. `Resnet50_UNet(n_classes, in_img, in_inf, l1_skip_conn=True)`

**標準的なマルチモーダル U-Net モデル**（`subclass=False` のとき使用）。

```
S1 入力 → Encoder1 → [f11, f12, f13, f14, f15]
                                ↓ Add（要素ごとに加算）
S2 入力 → Encoder2 → [f21, f22, f23, f24, f25]
                → [f1, f2, f3, f4]
                        ↓
                   デコーダー（U-Net）
                        ↓
              建物高さマップ (128×128×1)
```

**デコーダー部分の詳細**:
1. `f4` から開始 (8×8×1024)
2. Conv2D(512) + BN + UpSampling(×2) → 16×16
3. f3 と concatenate → SE Attention → Conv2D(256) + BN + UpSampling(×2) → 32×32
4. f2 と concatenate → SE Attention → Conv2D(128) + BN + UpSampling(×2) → 64×64
5. f1 と concatenate → SE Attention → Conv2D(64)  + BN + UpSampling(×2) → 128×128
6. Conv2D(1, 1×1, relu) → 建物高さマップ (128×128×1)

---

## 5. DG.py — カスタムデータジェネレーター

`tf.keras.utils.Sequence` を継承した**カスタムデータジェネレーター**を定義しています。大量のデータをメモリに一度に読み込まず、バッチごとにディスクから読み込むことでメモリを節約します。

### `Cust_DatasetGenerator`（メインで使用）

```python
class Cust_DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, label_files, batch_size=64):
        # label_files: ラベルファイル名のリスト
        # batch_size: バッチサイズ

    def __len__(self):
        # データセットのバッチ数を返す（= サンプル数 // バッチサイズ）

    def __getitem__(self, idx):
        # バッチを生成して返す
        # ランダムにサンプルを選択し、ランダムな時系列インデックス(0〜5)から
        # S1画像・S2画像・ラベルを読み込む
```

**バッチ生成の流れ**:
1. ラベルファイル名からランダムにバッチ分のサンプルを選択
2. ファイル名を `_` で分割して都市名・パッチIDを取得
3. ランダムな時系列インデックス `tmp` (0〜5の整数) を選択  
   → これにより **時系列データのランダムサンプリング**が実現される（データ拡張の一種）
4. S1画像・S2画像・ラベルを読み込んで配列に追加
5. 返り値: `([batch_s1, batch_s2], batch_label)`

### `Cust_DatasetGenerator2`（未使用・2つの時系列を同時に読み込む拡張版）

同様の構造ですが、1サンプルに対して異なる時系列インデックスの画像ペアを2セット読み込みます。コントラスト学習などの拡張手法向けに設計されています。

---

## 6. utils.py — ユーティリティ関数

### `load_data(DATA_PATH, fname, split)`
CSVファイルからラベルファイル名一覧を読み込み、学習・検証データに分割します。

```python
# 例: Filter_LABELS.csv
# city1_patch001_label_v1.tif
# city1_patch002_label_v1.tif
# ...

# split=0.2 なら全体の20%をランダムに検証データとして選択
# random.seed(30) により再現性を確保
```

### `scale_img(matrix)` — S1 画像の正規化

Sentinel-1 SAR 画像を 0〜1 の範囲に正規化します。

| チャネル | 物理量 | 最小値 | 最大値 |
|---------|--------|--------|--------|
| ch0 (VV) | VV偏波後方散乱強度 [dB] | -23 dB | 0 dB |
| ch1 (VH) | VH偏波後方散乱強度 [dB] | -28 dB | -5 dB |
| ch2 | 派生特徴 | データの実測最小値 | データの実測最大値 |

### `GRD_toRGB_S1(S1_PATH, fname)` — S1 画像読み込み

GeoTIFF 形式の Sentinel-1 画像を読み込み、前処理します。

```
1. rasterio でバンド1(VV)、バンド2(VH) を読み込み
2. チャネル軸を末尾に移動 (H,W,C 形式へ変換)
3. VV, VH を 128×128 にリサイズ
4. 3チャネルのゼロ配列に VV (ch0), VH (ch1) をセット (ch2=0)
5. scale_img() で 0〜1 に正規化
```

### `scale_imgS2(matrix, max_vis)` — S2 画像の正規化

Sentinel-2 画像の全5チャネルを `[0, max_vis]` の範囲から 0〜1 に正規化します（`max_vis=3000` を使用）。

### `GRD_toRGB_S2(S2_PATH, fname, max_vis)` — S2 画像読み込み

GeoTIFF 形式の Sentinel-2 画像を読み込み、前処理します。

```
使用バンド: Band1(B4=赤), Band2(B3=緑), Band3(B2=青), Band7(B8=近赤外), Band10(B10=SWIR)
1. rasterio でバンド 1,2,3,7,10 を読み込み
2. チャネル軸を末尾に移動
3. 各バンドを 128×128 にリサイズ
4. scale_imgS2() で 0〜1 に正規化
```

---

## 7. augment.py — データ拡張補助関数

`utils.py` の `scale_img`、`GRD_toRGB_S1`、`scale_imgS2`、`GRD_toRGB_S2` と**ほぼ同一の実装**を含んでいます。リファクタリング前の段階で別ファイルとして残っているものと思われます。

---

## 8. losses.py — 損失関数

### `SSIM_loss_graph(target, pred)`
**SSIM + MSE の複合損失**を計算します。

```python
ssim_loss = mean(1 - SSIM(target, pred, max_val=176.0, filter_size=7))
mse_loss  = MSE(target, pred)
loss = 0.4 * ssim_loss + 0.6 * mse_loss
```

- **SSIM (Structural Similarity Index)**: 輝度・コントラスト・構造の3成分で類似度を測る知覚的な画質指標。1に近いほど類似。
- `1 - SSIM` を損失とすることで、構造的な類似性を最大化するよう学習させます。

### `depth_loss_function(y_true, y_pred, config)`
深さ推定タスク向けの複合損失（試験的実装）。

```python
l_depth = mean(|y_pred - y_true|)         # L1損失
l_edges = mean(|∇y_pred - ∇y_true|)       # エッジ一致損失（勾配の差）
l_ssim  = clip((1 - SSIM(y_true, y_pred)) * 0.5, 0, 1)  # SSIM損失

loss = ssim_weight * l_ssim + edge_weight * mean(l_edges) + mean(l_depth)
```

### `mrcnn_mask_edge_loss_graph(...)`
エッジ検出フィルター（Sobel、Prewitt、Kayyali、Roberts、Laplacian など複数のカーネルをサポート）を使った**精密なエッジ一致損失**。現在のメイン学習では使用されていませんが、境界の精度向上が期待できる発展的な損失関数です。

---

## 9. train.py — 学習・評価メインスクリプト

### エンコーダー構築（ファイル先頭部分）

`train.py` の先頭でグローバルに SAR エンコーダーと光学エンコーダーを構築しています。

```python
# --- SAR エンコーダー (sar_encoder1) ---
sar_input = Input(shape=(128, 128, 3))
# ResNet50 の Stage1〜Stage4 に相当する層を手動で構築
# 最終的に [f1, f2, f3, f4] の4つの特徴マップを出力
sar_encoder1 = keras.Model(sar_input, [f1, f2, f3, f4], name="sar_encoder1")
# ImageNet 学習済み重みをロード（転移学習）
sar_encoder1.load_weights(weights_path, by_name=True, skip_mismatch=True)

# --- 光学エンコーダー (opt_encoder1) ---
# SAR エンコーダーと同じ構造（ただしチャネル数は s2_ch=5）
# 同様に ImageNet 重みで初期化
opt_encoder1 = keras.Model(opt_input, [f1, f2, f3, f4], name="opt_encoder1")
```

### デコーダー構築

```python
# 4つのスキップ接続入力を受け取り、128×128×1 の高さマップを出力
f1 = Input(shape=(64, 64, 64))    # sar/opt encoder の出力1
f2 = Input(shape=(32, 32, 256))   # sar/opt encoder の出力2
f3 = Input(shape=(16, 16, 512))   # sar/opt encoder の出力3
f4 = Input(shape=(8, 8, 1024))    # sar/opt encoder の出力4

# f4 → アップサンプリング+畳み込み → f3 と結合 → SE Attention
# → アップサンプリング+畳み込み → f2 と結合 → SE Attention
# → アップサンプリング+畳み込み → f1 と結合 → SE Attention
# → アップサンプリング+畳み込み(1×1, relu) → 高さマップ

decoder1 = keras.Model([f1, f2, f3, f4], outputs, name="decoder1")
```

### `Combined_HE_model` クラス（サブクラスモデル）

`keras.Model` のサブクラスとして実装されたカスタムモデルで、**カスタム学習・評価ループ**を持ちます。

```python
class Combined_HE_model(keras.Model):
    def __init__(self, sar_encoder1, opt_encoder1, decoder1, **kwargs):
        self.alpha = 0.4          # MSE 損失の重み
        self.beta  = 0.6          # コサイン類似度損失の重み
        self.maxDepthVal = 176.0  # 建物高さの最大値
        # 損失関数: MSE, Huber, MAPE, CosineSimilarity を定義
```

#### `train_step(data)`（1バッチの学習ステップ）

```python
[s1_img, s2_img], label = data   # S1画像, S2画像, 正解ラベル

# 1. 各エンコーダーで特徴抽出
[o11, o12, o13, o14] = sar_encoder1(s1_img)
[o21, o22, o23, o24] = opt_encoder1(s2_img)

# 2. 対応する特徴マップを要素ごとに加算（モダリティ融合）
o1 = Add()([o11, o21])   # 64×64×64
o2 = Add()([o12, o22])   # 32×32×256
o3 = Add()([o13, o23])   # 16×16×512
o4 = Add()([o14, o24])   # 8×8×1024

# 3. デコーダーで高さマップを生成
he_out = decoder1([o1, o2, o3, o4])

# 4. 損失計算
mse_loss = MSE(label, he_out)
ss_loss  = CosineSimilarity(label, he_out)
total_loss = 0.4 * mse_loss + 0.6 * ss_loss

# 5. 勾配計算と重み更新
grads = tape.gradient(total_loss, trainable_weights)
optimizer.apply_gradients(grads)
```

#### `test_step(data)`（1バッチの検証ステップ）
`train_step` と同様ですが、`training=False` で推論し、重み更新を行いません。

### `train_fusion(...)` — 学習関数

```python
def train_fusion(n_classes, S1, S2, train_y, val_y, WEIGHT_FNAME, subclass=False):
    optimizer = Adam(lr=0.0001)
    earlystopper = EarlyStopping(patience=20)       # 20エポック改善なしで停止
    scheduler = ReduceLROnPlateau(patience=10, factor=0.1)  # 学習率を1/10に減少

    if subclass:
        # Combined_HE_model（カスタム学習ループ）を使用
        model = Combined_HE_model(sar_encoder1, opt_encoder1, decoder1)
        model.fit(..., epochs=50)
        model.save_weights(WEIGHT_PATH / WEIGHT_FNAME)
    else:
        # 標準的な Resnet50_UNet を使用（MSE損失）
        model = Resnet50_UNet(n_classes, S1, S2)
        model.compile(optimizer, loss=mse)
        model.fit(..., epochs=50)
```

### `evaluate_fusion(...)` — 評価関数

```python
def evaluate_fusion(weight_file, S1, S2, val_y):
    model = Combined_HE_model(sar_encoder1, opt_encoder1, decoder1)
    model.load_weights(WEIGHT_PATH / weight_file)
    
    for fname in val_y:
        # ラベルファイル名から S1/S2 ファイル名を構築（時系列インデックスは固定値6）
        s1img = GRD_toRGB_S1(S1_PATH, S1_name)
        s2img = GRD_toRGB_S2(S2_PATH, S2_name, S2_MAX)
        
        # 推論
        [o11..o14] = model.sar_encoder1(in_s1img)
        [o21..o24] = model.opt_encoder1(in_s2img)
        o1..o4 = Add()([oX1, oX2])
        pred_mask = model.decoder1([o1, o2, o3, o4])
        
        # MSE を計算して蓄積
        MSE.append(mean_squared_error(labelimg, pred_mask))
    
    print('Average MSE:', sum(MSE) / len(MSE))
```

---

## 10. データの流れ（全体パイプライン）

```
Filter_LABELS.csv
    ↓ load_data()
    ↓ 80%: train_y, 20%: val_y
    ↓
Cust_DatasetGenerator
    ↓ バッチごとに:
    │   ラベル名 → S1ファイル名 / S2ファイル名 を生成
    │   時系列インデックス (0〜5) をランダム選択
    │   GRD_toRGB_S1() → S1画像 (128×128×3, float32, 0〜1)
    │   GRD_toRGB_S2() → S2画像 (128×128×5, float32, 0〜1)
    │   ラベル (128×128×1, float32)
    ↓
Combined_HE_model.train_step()
    ├── sar_encoder1(S1画像) → [f1,f2,f3,f4]
    ├── opt_encoder1(S2画像) → [f1,f2,f3,f4]
    ├── 特徴マップを要素ごとに加算
    └── decoder1([o1,o2,o3,o4]) → 予測高さマップ (128×128×1)
    ↓
損失計算: 0.4×MSE + 0.6×CosineSimilarity
    ↓
Adam オプティマイザーで重み更新
    ↓
検証データで test_step() を実行
    ↓
EarlyStopping / ReduceLROnPlateau で収束判定
    ↓
重みを .h5 ファイルに保存
```

---

## 11. 学習・評価の実行方法

```bash
# 学習
python train.py train --weight 'my_model.h5'

# 評価
python train.py evaluate --weight 'my_model.h5'
```

### 注意事項

1. **パスの設定**: `code/config.py` の `ROOT_PATH`、`DATA_PATH` を実際の環境に合わせて変更してください。
2. **データ形式**: 
   - S1画像: GeoTIFF形式、最低2バンド (VV, VH)
   - S2画像: GeoTIFF形式、最低10バンド（バンド1,2,3,7,10を使用）
   - ラベル: GeoTIFF形式、1バンド（建物高さ [m]）
3. **ラベルCSV**: `Filter_LABELS.csv` に有効なラベルファイル名（`.tif`）を列挙してください。ファイル名の形式は `cityname_patchid_..._v1.tif` を想定しています。
4. **GPU環境**: `train.py` では GPU 0, 1, 2 を使用するよう設定されています（`CUDA_VISIBLE_DEVICES="0, 1, 2"`）。
5. **転移学習**: ResNet50の ImageNet 学習済み重みは自動的にダウンロードされます。

### モデル選択

| `subclass` パラメータ | モデル | 損失関数 | 特徴 |
|----------------------|--------|---------|------|
| `True` (デフォルト) | `Combined_HE_model` | 0.4×MSE + 0.6×CosineSimilarity | カスタム学習ループ |
| `False` | `Resnet50_UNet` | MSE のみ | 標準的な Keras API |
