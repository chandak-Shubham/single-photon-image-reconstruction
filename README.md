# Single Photon Image Reconstruction

This repository contains my solution for the **Single Photon Challenge 2025**, a benchmark competition organized by researchers from UW-Madison, Portland State University, Purdue, CMU, and the US Naval Research Lab.

The objective is to reconstruct high-quality RGB images from extremely noisy single-photon camera (SPC) data using deep learning.

---

## Project Description

Single-photon cameras are capable of detecting individual photons, making them extremely sensitive in low-light conditions. However, each captured frame is a **binary image** — every pixel records either 0 or 1 depending on whether a photon was detected. A single frame carries almost no usable information; meaningful reconstruction requires aggregating information from many such frames.

The task is to take a burst of **1024 binary frames** (800×800 pixels, RGB) and produce a single clean, high-fidelity RGB image.

The model used is an **Attention Residual U-Net** trained end-to-end with a combined pixel, structural, and perceptual loss.

---

## Dataset Description

The dataset is provided by the [Single Photon Challenge](https://singlephotonchallenge.com/download).

* **Total training samples:** 1850
* **Input:** Bursts of 1024 binary single-photon frames per scene
* **Image resolution:** 800 × 800 pixels (after unpacking)
* **Channels:** RGB (3 channels)

Each scene folder contains two types of files:

| File Type | Description | Shape after loading |
|---|---|---|
| `.npy` | Bit-packed binary photon frames (noisy input) | (128, 800, 800, 3) after unpacking |
| `.png` | Clean ground truth RGB image | (800, 800, 3) |

---

## Data Loader

The data loader handles scene discovery, bit-unpacking of compressed frames, and tensor construction.

### Scene Discovery

Training and test scenes are discovered by scanning the dataset folder for `train_*` and `test_*` subfolders:

```python
def get_train_scenes(dataset_path):
    for folder in sorted(Path(dataset_path).glob("train_*")):
        scene_root = folder / "train"
        for scene in scene_root.iterdir():
            train_scenes.append(scene)
```

### Unpacking Compressed Frames

Raw photon data is stored as bit-packed `.npy` files. Each file is unpacked into binary frames:

```python
def unpack(arr):
    arr = arr[-128:]                  # take last 128 frames → (128, 800, 100, 3)
    arr = np.unpackbits(arr, axis=2)  # unpack bits → (128, 800, 800, 3)
    return arr
```

### Input Tensor Construction

The 128 unpacked binary frames are converted into a single flat tensor used as model input:

```
(128, 800, 800, 3)
→ Permute to (128, 3, 800, 800)
→ Flatten frames into channels → (384, 800, 800)
```

Each of the 384 channels represents one color channel from one of the 128 selected frames. This compact representation allows the model to process all frame information in a single forward pass.

### SPCDataset

`SPCDataset` supports both train and test modes:

* **Train mode:** Returns `(noisy_tensor, clean_tensor)` — noisy input (384, 800, 800) and clean RGB target (3, 800, 800)
* **Test mode:** Returns `noisy_tensor` only — no ground truth available

---

## Model Architecture

The model is an **Attention Residual U-Net** (`AttentionResUNet`) with 384 input channels and 3 output channels.

### Residual Block

Each encoder and decoder stage uses a residual block with GroupNorm:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip  = nn.Conv2d(in_c, out_c, 1)  # learned projection
```

GroupNorm is used instead of BatchNorm because it works correctly with small batch sizes (batch size 2 here).

### Attention Gate

At each decoder upsampling step, an attention gate selectively weights encoder skip features:

```python
class AttentionBlock(nn.Module):
    def __init__(self, g_c, x_c, inter_c):
        self.Wg  = nn.Conv2d(g_c, inter_c, 1)
        self.Wx  = nn.Conv2d(x_c, inter_c, 1)
        self.psi = nn.Sequential(nn.Conv2d(inter_c, 1, 1), nn.Sigmoid())
```

This lets the decoder focus on spatially relevant regions rather than using all encoder features equally.

### Full Architecture

```
Input (384, 800, 800)
│
Encoder:
  enc1: 384 → 128  + MaxPool
  enc2: 128 → 256  + MaxPool
  enc3: 256 → 512  + MaxPool
│
Bottleneck:
  512 → 1024 → 1024
│
Decoder:
  up1: ConvTranspose2d + AttentionGate + ResidualBlock → 512
  up2: ConvTranspose2d + AttentionGate + ResidualBlock → 256
  up3: ConvTranspose2d + AttentionGate + ResidualBlock → 128
│
Output Conv (128 → 3)
+ Global Residual (mean of input channels, repeated 3×)
│
Output (3, 800, 800) — clean RGB image
```

### Global Residual (Photon Mean Trick)

The mean across all 384 input channels approximates the underlying photon flux and is added directly to the output:

```python
input_mean = x.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
out = self.out(d3) + input_mean
```

This gives the network a meaningful starting estimate so it refines signal rather than learning from scratch.

---

## Image Preprocessing

All preprocessing is done in the data loader and training loop — no external transforms library is used.

| Step | Detail |
|---|---|
| Bit unpacking | `.npy` files unpacked via `np.unpackbits` |
| Frame selection | Last 128 of 1024 frames used |
| Channel stacking | 128 frames × 3 channels = 384 input channels |
| Normalization | Divide by 255.0 to scale to [0, 1] |
| Clean target | Loaded as PIL image, converted to tensor, normalized to [0, 1] |

---

## Loss Function

Training uses a weighted combination of three losses:

| Loss | Weight | Purpose |
|---|---|---|
| L1 (pixel) | 0.7 | Pixel-level accuracy |
| SSIM | 0.2 | Structural and perceptual similarity |
| VGG Perceptual | 0.1 | High-level feature similarity via VGG16 (relu3_3) |

```python
loss = 0.7 * l1 + 0.2 * (1 - ssim) + 0.1 * vgg_loss
```

VGG features are extracted from the first 16 layers of a pretrained VGG16 (frozen during training). Both output and target are normalized to ImageNet statistics before passing through VGG.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Model | AttentionResUNet |
| Input channels | 384 |
| Output channels | 3 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=50, eta_min=1e-6) |
| Epochs | 50 |
| Batch size | 2 |
| Mixed precision | torch.amp (float16) |
| Gradient clipping | 1.0 |
| Checkpointing | Every 10 epochs |
| TensorBoard logging | Batch loss, epoch loss, visual samples |

Optimizer:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

Scheduler:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
```

Mixed precision training with gradient scaling:

```python
scaler = torch.amp.GradScaler("cuda")
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

---

## Results

| Model | Avg PSNR | Avg SSIM |
|---|---|---|
| **Attention ResUNet (this work)** | **31.489** | **0.948** |

---

## Repository Structure

```
single-photon-challenge/
├── dataloader/
│   ├── dataset_loader.py      
│   └── unpack.py              
├── model/
│   └── attention_resunet.py   
├── train.py                   
├── test.py                    
└── README.md
```

---

## Libraries Used

```
torch
torchvision
torchmetrics
numpy
pillow
matplotlib
tensorboard
```

---

## Competition

**The Single Photon Challenge** is a benchmark organized by researchers from UW-Madison, Portland State University, Purdue University, CMU, and the US Naval Research Lab, co-located with ICCV 2025.

* Website: https://singlephotonchallenge.com
* Dataset: https://singlephotonchallenge.com/download

