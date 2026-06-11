# Phân Tích Kiến Trúc Mô Hình MS2I-Style (sketch-ms2i-style.ipynb)

> **Tên mô hình:** MS2I — *Multi-Scale Sketch-to-Image with Style Disentanglement*
> **File:** `notebook/sketch-ms2i-style.ipynb`

---

## 1. Tổng Quan

MS2I là một mô hình GAN (Generative Adversarial Network) được thiết kế để chuyển đổi ảnh phác thảo (sketch) thành ảnh thực tế có màu sắc, với khả năng **kiểm soát màu sắc rõ ràng từ người dùng** thông qua một vector màu one-hot 11 lớp. Mô hình kết hợp ba đặc điểm kỹ thuật mới:

1. **Reparameterizable Convolution** (Multi-branch → Single-branch sau khi fuse)
2. **Monarch Attention** (attention hiệu quả cho inference)
3. **Style-Modulated Generation** (điều khiển màu sắc theo phong cách StyleGAN)

---

## 2. Kiến Trúc Tổng Thể

```
Input: Sketch (3×256×256) + Color Vec (11,) + Noise z (128,)
                │
          ┌─────▼──────┐
          │ StyleMapping│  ← Color + z → Style latent (256,)
          └─────────────┘
                │ style
     ┌──────────▼──────────┐
     │   Generator: MS2I    │
     │  ┌────────────────┐  │
     │  │   Stem Conv    │  │  7×7, stride=4 → (B, 32, 64, 64)
     │  └───────┬────────┘  │
     │          ▼           │
     │  ┌───────────────┐   │
     │  │ Encoder Stage │   │  3×(Block + DownSample)
     │  └───────┬───────┘   │
     │          ▼           │
     │  ┌───────────────┐   │
     │  │  Bottleneck   │   │  Block (4 RepTransformerBlocks, dim=256)
     │  └───────┬───────┘   │
     │          ▼           │
     │  ┌───────────────┐   │
     │  │ Decoder Stage │   │  3×(UpSample + Skip + Block + StyledRefinement)
     │  └───────┬───────┘   │
     │          ▼           │
     │  ┌───────────────┐   │
     │  │ StyleAwareHead│   │  RepConv3 + StyledRefinement + 1×1 Conv → RGB
     │  └───────────────┘   │
     └─────────────────────-┘
                │
          Output (3×256×256)
```

---

## 3. Kiến Trúc Chi Tiết Từng Lớp

### 3.1. StyleMapping — Mạng Ánh Xạ Style

```python
class StyleMapping(nn.Module):
    # Input: color_vec (11,) + z (128,) → style (256,)
```

| Lớp | Chi tiết |
|-----|----------|
| Input | Nối `color_vec` (11-dim one-hot) + `z` (128-dim noise, L2-normalized) = 139 dim |
| Hidden layers | 3 lớp Linear(256) + LeakyReLU(0.2) + Dropout(0.1) |
| Output | Linear(256) → style latent vector |

**Vai trò:** Chuyển đổi thông tin màu sắc và nhiễu ngẫu nhiên thành một **style latent vector** 256 chiều, vector này điều khiển toàn bộ quá trình colorization trong decoder.

---

### 3.2. Stem — Lớp Đầu Vào

```python
self.stem = nn.Conv2d(3, dims[0], kernel_size=7, stride=4, padding=3)
# 3×256×256 → 32×64×64
```

- Conv 7×7, stride=4 để tạo receptive field lớn ngay từ đầu, giảm kích thước ảnh xuống 4 lần.

---

### 3.3. Building Block: RepConv3 (Re-parameterizable 3×3 Conv)

```python
class RepConv3(nn.Module):
    # Training: 5 branches song song
    # Inference: 1 conv duy nhất (sau khi fuse)
```

**Cấu trúc khi training (5 nhánh):**

| Nhánh | Kernel | Mục đích |
|-------|--------|----------|
| `conv_3x3` | 3×3 | Spatial features chính |
| `conv_1x1` | 1×1 (padding=1) | Channel mixing |
| `conv_1x3` | 1×3 (padding=0,1) | Horizontal edges |
| `conv_3x1` | 3×1 (padding=1,0) | Vertical edges |
| `conv_1x1_branch → conv_3x3_branch` | 1×1 → 3×3 | Non-linear composition |

**Fusion (khi inference):** Tất cả 5 nhánh được gộp lại thành **1 conv 3×3 duy nhất** bằng cách padding và cộng trọng số.

---

### 3.4. Building Block: RepConv5 và RepConv7

Tương tự RepConv3 nhưng với kernel lớn hơn:

**RepConv5** (10 nhánh): conv_5×5, 3×3, 1×1, 1×5, 5×1, 1×3, 3×1, 3×5, 5×3, 1×1→5×5

**RepConv7** (9 nhánh): conv_7×7, 7×1, 1×7, 7×5, 5×7, 5×5, 1×5, 5×1, 1×1→7×7

Tất cả được fuse thành 1 conv duy nhất sau training.

---

### 3.5. DownSample và UpSample

```python
class DownSample(nn.Module):
    # Conv 1×1 (C → C//2) → PixelUnshuffle(factor=2)
    # Output: (B, C*2, H/2, W/2)

class UpSample(nn.Module):
    # Conv 1×1 (C → C*2) → PixelShuffle(factor=2)
    # Output: (B, C//2, H*2, W*2)
```

**Tại sao dùng PixelShuffle?** Tránh checkerboard artifact so với deconvolution thông thường, đồng thời giữ thông tin tốt hơn.

---

### 3.6. LayerNorm (BiasFree & WithBias)

```python
class LayerNorm(nn.Module):
    # Hỗ trợ 2 mode: 'BiasFree' và 'WithBias'
    # Input/Output: (B, C, H, W) — normalize trên chiều channel cuối
```

Chuẩn hóa trên từng token (pixel position), không phải batch-level, phù hợp với transformer architecture.

---

### 3.7. RepAttn — Re-parameterizable Attention Block ⭐

```python
class RepAttn(nn.Module):
    # Training:  Scaled Dot-Product Attention (standard)
    # Inference: MonarchAttention (xấp xỉ Butterfly-Monarch)
```

| Phase | Mechanism |
|-------|-----------|
| **Training** | `q @ k.T * scale → softmax → @ v` (standard self-attention) |
| **Inference** | `MonarchAttention(block_size=16, num_steps=2)` — hiệu quả hơn |

**Cấu trúc:**
- `qkv`: Conv 1×1 chiếu input thành Q, K, V
- Multi-head split: `rearrange(q, 'b (head c) h w → b head c (h w)')`
- Áp dụng attention function tương ứng theo phase
- `proj`: Conv 1×1 chiếu output về chiều gốc

---

### 3.8. RepFFN — Feed-Forward Network với RepConv

```python
class RepFFN(nn.Module):
    # project_in: RepConv3(dim, hidden, groups=1)  — Channel mixing
    # dwconv:     RepConv3(hidden, hidden*2, groups=hidden)  — Depthwise
    # project_out: Conv 1×1  — Chiều về lại dim
    # Activation: Gated Linear Unit (GLU): gelu(x1) * x2
```

Sử dụng **Gated Linear Unit** thay vì ReLU thông thường — đã được chứng minh hiệu quả hơn trong các vision transformer.

---

### 3.9. RepTransformerBlock

```python
class RepTransformerBlock(nn.Module):
    # x = x + RepAttn(LayerNorm1(x))   # Pre-norm attention
    # x = x + RepFFN(LayerNorm2(x))    # Pre-norm FFN
```

Block tiêu chuẩn theo dạng **Pre-norm Transformer**, giống ViT nhưng dùng RepConv thay cho MLP thông thường.

---

### 3.10. ModulatedConv2d — StyleGAN-like Modulated Convolution ⭐

```python
class ModulatedConv2d(nn.Module):
    # affine: Linear(style_dim, in_channels) — tính modulation
    # weight được điều chỉnh theo style: weight * affine(style)
    # Sau đó demodulate (chuẩn hóa trọng số)
```

**Cơ chế hoạt động:**
```
modulation = affine(style)      # (B, C_in)
weight = conv_weight * modulation  # per-sample weight modulation
weight = demodulate(weight)        # tránh amplitude explosion
output = grouped_conv(x, weight, groups=B)  # per-sample grouped conv
```

---

### 3.11. StyledRefinement — Style Injection vào Decoder ⭐

```python
class StyledRefinement(nn.Module):
    # residual = ModulatedConv2d(LayerNorm(x), style)
    # output = x + strength * GELU(residual)
```

| Parameter | Giá trị | Ý nghĩa |
|-----------|---------|---------|
| `strength` (stage 0) | 0.15 | Màu sắc ảnh hưởng ít ở decoder stage đầu (gần bottleneck) |
| `strength` (stage 1) | 0.35 | Ảnh hưởng trung bình |
| `strength` (stage 2) | 0.65 | Ảnh hưởng mạnh ở stage gần output (high-res features) |

---

### 3.12. StyleAwareHead — Output Head ⭐

```python
class StyleAwareHead(nn.Module):
    # pre:    RepConv3(in_ch, hidden, 1)
    # style:  StyledRefinement(hidden, style_dim, strength=1.0)
    # to_rgb: Conv 1×1(hidden → 3)
```

Layer cuối cùng trước output, áp dụng style một lần nữa với `strength=1.0` (mạnh nhất) để đảm bảo màu sắc final được in rõ lên ảnh RGB.

---

### 3.13. MS2I Generator — Tổng Hợp

```
Encoder (3 stages):
  Stage 0: Stem → Block(1 RepTransformerBlock, dim=32)  → DownSample(32)
  Stage 1:       → Block(2 RepTransformerBlock, dim=64)  → DownSample(64)
  Stage 2:       → Block(2 RepTransformerBlock, dim=128) → DownSample(128)

Bottleneck:
  Block(4 RepTransformerBlock, dim=256)

Decoder (3 stages, ngược lại):
  Stage 0: UpSample(256) → SkipConnection + Block(2, dim=128) → StyledRefinement(strength=0.15)
  Stage 1: UpSample(128) → SkipConnection + Block(2, dim=64)  → StyledRefinement(strength=0.35)
  Stage 2: UpSample(64)  → SkipConnection + Block(1, dim=32)  → StyledRefinement(strength=0.65)

Output:
  Bilinear Upsample → StyleAwareHead(strength=1.0) → RGB (3×256×256)
```

**Config mặc định:**
- `dims = [32, 64, 128, 256]`
- `num_blocks = [1, 2, 2, 4]`
- `num_heads = [1, 2, 4, 8]`
- `z_dim = 128`, `style_dim = 256`, `color_dim = 11`

---

### 3.14. Discriminator

```python
class Discriminator(nn.Module):
    # PatchGAN-style với Spectral Normalization
```

| Lớp | Chi tiết |
|-----|----------|
| Stem | Conv 4×4, stride=2 (với SN optionally) |
| Stage 1 | ConvBNLReLU(64→128, stride=2) — SN + BN + LeakyReLU(0.2) |
| Stage 2 | ConvBNLReLU(128→256, stride=2) |
| Stage 3 | ConvBNLReLU(256→512, stride=2) |
| Head | Conv 4×4, stride=1, padding=1 → 1 channel (patch scores) |

- **Spectral Normalization:** Giúp ổn định training GAN bằng cách giới hạn Lipschitz constant của D.
- **PatchGAN:** D phân biệt từng patch nhỏ của ảnh → học được texture cục bộ tốt hơn.

---

### 3.15. Loss Functions

```python
# Adversarial: MSE Loss (LSGAN objective — ổn định hơn BCE)
criterion_adv = nn.MSELoss()

# Perceptual Loss: VGG19 feature matching tại nhiều layer

# L1 Loss: Pixel-level reconstruction (tuỳ chọn)
criterion_l1 = nn.L1Loss()

# Tổng loss Generator:
L_G = λ_adv * L_adv + λ_perc * L_perc + λ_l1 * L_l1
```

**Label Smoothing:** `d_label_real=0.9`, `d_label_fake=0.1` để tránh D học quá tự tin.

---

## 4. Những Điểm Mới và Cải Tiến Đột Phá

### 4.1. ⭐ Re-parameterizable Multi-branch Convolution (RepConv)

**Điểm mới:** Mô hình sử dụng RepConv3/5/7 với **nhiều nhánh song song** khi training, sau đó **gộp lại thành 1 conv duy nhất** khi inference.

**Cơ chế:**
```
Training:  out = conv_3×3(x) + conv_1×1(x) + conv_1×3(x) + conv_3×1(x) + conv_3×3(conv_1×1(x))
Inference: out = reparam(x)  [một conv 3×3 duy nhất với weight = sum of all branches]
```

**Lý do áp dụng:**
- **Training:** Nhiều nhánh tạo ra **implicit ensemble** — mỗi nhánh capture một loại pattern khác nhau (isotropic features, vertical edges, horizontal edges...). Gradient flow tốt hơn.
- **Inference:** Chỉ dùng **1 conv duy nhất** → tốc độ ngang bằng với single conv thông thường, không tốn thêm FLOP.
- **Zero trade-off:** Không đánh đổi tốc độ inference để có chất lượng training tốt hơn.

---

### 4.2. ⭐ Monarch Attention — Re-parameterization Training↔Inference

**Điểm mới:** Training dùng standard SDPA (O(N²)), nhưng **inference được switch sang MonarchAttention** — một dạng xấp xỉ attention với cấu trúc Butterfly/Monarch matrix.

**Cơ chế Re-parameterization:**
```python
# Training: standard softmax attention
def common_attn(q, k, v):
    scale = (q.shape[-1]) ** -0.5
    attn = (q @ k.T) * scale
    return attn.softmax(-1) @ v

# Inference: switch sang MonarchAttention sau khi .fuse()
def fuse(self):
    self.attn_fn = self.monarch_attn
```

**Lý do áp dụng:**
- Standard attention có độ phức tạp **O(N²)** với N = H×W → rất chậm khi ảnh lớn
- MonarchAttention xấp xỉ attention bằng **structured sparse matrices** (Butterfly patterns) → gần **O(N log N)**
- Dùng SDPA khi training (gradient ổn định hơn) + MonarchAttention khi inference (nhanh hơn) → best of both worlds

---

### 4.3. ⭐ StyleGAN-like Modulated Convolution cho Color Control

**Điểm mới:** Thay vì concat màu sắc vào input (như pix2pix), mô hình dùng **weight modulation** để inject màu sắc vào **từng lớp decoder**.

**Cơ chế:**
```python
modulation = affine(style)                          # (B, C_in)
weight = conv_weight * modulation                   # per-sample
demod = rsqrt(weight.pow(2).sum() + eps)           # demodulate
weight = weight * demod
out = grouped_conv(x.reshape(1,B*C,H,W), weight, groups=B)
```

**Lý do áp dụng:**
- **Disentanglement:** Màu sắc và cấu trúc được giữ tách biệt. Encoder chỉ xử lý sketch (cấu trúc), style vector chỉ chứa thông tin màu.
- **Controllability:** Người dùng có thể chọn màu sắc ở inference time mà không cần retrain.
- **StyleGAN demodulation:** Tránh hiện tượng "blobby artifacts" khi modulation quá mạnh.

---

### 4.4. ⭐ Graduated Style Strength trong Decoder

**Điểm mới:** Mỗi stage decoder có một `strength` khác nhau, tăng dần từ bottleneck ra output:

```python
style_strengths = [0.15, 0.35, 0.65]  # stage 0 → 1 → 2 (gần output)
```

**Lý do áp dụng:**
- **Stage gần bottleneck (strength=0.15):** Feature map nhỏ, chứa thông tin semantic cao cấp. Màu sắc không nên override quá mạnh để giữ cấu trúc hình học.
- **Stage gần output (strength=0.65):** Feature map lớn, chứa texture và màu sắc chi tiết. Style cần ảnh hưởng mạnh để tô màu chính xác.
- Mô phỏng cách não người xử lý: nhận dạng cấu trúc trước, tô màu sau.

---

### 4.5. ⭐ Sketch Degradation Augmentation

**Điểm mới:** Trong quá trình training, sketch input được **cố tình làm hỏng** (degrade) với xác suất 70%:

```python
A.CoarseDropout(max_holes=20, max_height=40, fill=255, p=0.7)  # Xóa nét ngẫu nhiên
A.ElasticTransform(alpha=300, sigma=10, fill=255, p=0.49)      # Biến dạng nét
```

**Lý do áp dụng:**
- Sketch thực tế của người dùng thường **không hoàn hảo**: thiếu nét, run tay, biến dạng
- Train với sketch chuẩn (HED/Pencil/Canny) sẽ overfit và fail với sketch kém chất lượng
- Degradation buộc model học cách **nội suy và hoàn thiện** các nét bị thiếu → robust hơn với input thực tế

---

### 4.6. ⭐ 11-Class Color Conditioning (One-Hot)

**Điểm mới:** Điều kiện màu sắc được encode thành vector one-hot 11 lớp:

```python
COLOR_LABELS = ["black", "white", "gray", "red", "orange", "yellow",
                "green", "blue", "purple", "pink", "brown"]
```

**Lý do áp dụng:**
- **Dễ sử dụng:** Người dùng chỉ cần chọn 1 trong 11 màu thay vì cung cấp ảnh màu tham khảo
- **Training ổn định:** Bài toán classification đơn giản, không có ambiguity
- **Generalization:** 11 lớp màu bao phủ đủ không gian màu sắc thời trang (fashion domain)
- **Stochastic variety:** Vector `z` (128-dim noise) cho phép sinh ra nhiều biến thể trong cùng 1 màu (khác tone, pattern, texture)

---

### 4.7. ⭐ Multi-Method Sketch Training (HED + Pencil + Canny)

**Điểm mới:** Training dùng đồng thời 3 loại sketch khác nhau với tỷ lệ cố định:

```python
sketch_ratios = {"hed": 0.5, "pencil": 0.3, "canny": 0.2}
```

**Lý do áp dụng:**
- Mỗi phương pháp extract sketch có đặc điểm riêng: HED (mịn, giống thực tế), Pencil (giống nét tay), Canny (sắc nét, nhiều nhiễu)
- Training với nhiều loại giúp model **tổng quát hóa** tốt hơn với mọi loại sketch đầu vào
- Tỷ lệ 50/30/20 (HED/Pencil/Canny) được chọn vì HED gần với sketch thực tế nhất

---

## 5. Sơ Đồ Luồng Dữ Liệu

```
Sketch(3,256,256)         Color(11,)        z(128,)
        │                     │                │
     Stem 7×7                 └──────────────┬─┘
  (3→32, stride=4)                           │
        │                            StyleMapping(MLP)
        │                                    │
        ▼                                    ▼
   Encoder[0]: Block(dim=32) ──────────── style(256,)
        │                                    │
    DownSample                               │
        │                                    │
   Encoder[1]: Block(dim=64)                 │
        │                                    │
    DownSample                               │
        │                                    │
   Encoder[2]: Block(dim=128)                │
        │                                    │
    DownSample                               │
        │                                    │
   Bottleneck: Block(dim=256)                │
        │                                    │
    UpSample                                 │
        │←──Skip─── Encoder[2]              │
   Decoder[0]: Block(dim=128)                │
        │                                    │
   StyledRefinement(str=0.15)  ←────────────┤
        │                                    │
    UpSample                                 │
        │←──Skip─── Encoder[1]              │
   Decoder[1]: Block(dim=64)                 │
        │                                    │
   StyledRefinement(str=0.35)  ←────────────┤
        │                                    │
    UpSample                                 │
        │←──Skip─── Encoder[0]              │
   Decoder[2]: Block(dim=32)                 │
        │                                    │
   StyledRefinement(str=0.65)  ←────────────┤
        │                                    │
   Bilinear Upsample → 256×256               │
        │                                    │
   StyleAwareHead(str=1.0)     ←────────────┘
        │
   Output RGB (3,256,256)
```

---

## 6. So Sánh với Các Phương Pháp Trước

| Tiêu chí | pix2pix | Pix2PixHD | MUNIT | **MS2I-Style** |
|----------|---------|-----------|-------|----------------|
| Kiểm soát màu | ❌ Không | ❌ Không | Một phần | ✅ 11-class explicit |
| Attention | ❌ | ❌ | ❌ | ✅ MonarchAttn |
| Re-param Conv | ❌ | ❌ | ❌ | ✅ RepConv3/5/7 |
| Style injection | ❌ | ❌ | Per-image | ✅ Per-layer graduated |
| Inference speed | Normal | Normal | Normal | ✅ Fused (nhanh hơn) |
| Multi-sketch | ❌ | ❌ | ❌ | ✅ HED+Pencil+Canny |
| Sketch robustness | ❌ | ❌ | ❌ | ✅ Degradation Aug |

---

## 7. Kết Luận

MS2I-Style là một mô hình tổng hợp ảnh sketch-to-image **có điều kiện màu sắc rõ ràng**, kết hợp nhiều cải tiến kỹ thuật:

- **Tốc độ inference nhanh:** Nhờ Re-parameterizable Convolutions và MonarchAttention fuse
- **Kiểm soát tốt hơn:** Color one-hot + noise z cho phép kiểm soát màu và diversity
- **Chất lượng cao hơn:** Style modulation có độ mạnh tăng dần qua decoder layers
- **Robust với sketch không hoàn hảo:** Nhờ Sketch Degradation Augmentation
- **Tổng quát hóa tốt:** Training với 3 loại sketch method khác nhau

Đây là sự kết hợp sáng tạo giữa **RepVGG-style reparameterization**, **StyleGAN-style modulation**, và **Transformer attention** trong một framework GAN đơn giản, thực tế và hiệu quả cho bài toán chuyển đổi sketch thời trang sang ảnh thực tế có màu sắc có kiểm soát.
