<div align="center">

# 🔁 Exploring Recurrent Architectures for Sequence Learning — Experiment 6
### RNN, LSTM & GRU for Sequence Classification, Video Understanding, and Sequence-to-Sequence Learning

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

**Course:** CS3807 – Deep Learning Laboratory · **Degree:** B.Tech AI & Data Science
**Datasets:** [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — 9-channel raw inertial signals, 6 activity classes · [UCF101 subset](https://huggingface.co/datasets/sayakpaul/ucf101-subset) — 210 real clips, 5 action classes

📄 [**Read the full lab report (PDF)**](./docs/DL_LAB_6_Report.pdf) · 📓 [**Open the notebook**](./Deep_Learning_Lab_6.ipynb)

</div>

---

## 📌 Objective

Build a working understanding of recurrent sequence learning — Vanilla RNN, LSTM, GRU, Backpropagation Through Time, and the vanishing/exploding gradient problem — by training all three architectures under an identical protocol, then extending the same gated-recurrence idea to video action recognition and sequence-to-sequence learning.

```
Input(128×9) → Recurrent(32) → Dropout(0.2) → Dense(16, ReLU) → Dense(6, Softmax)
Video(10×1280 CNN features) → LSTM/GRU(32) → Dense(32, ReLU) → Dropout(0.3) → Dense(5, Softmax)
```

## ⚠️ A Note on Data Sourcing

`archive.ics.uci.edu` and the UCF101/CRCV host were both unreachable from the execution sandbox. Both datasets were still sourced as **genuine, unmodified data**: the HAR files came from a byte-identical GitHub mirror (`srvds/Human-Activity-Recognition`) of the official UCI release, and the video clips came from a pre-packaged, genuine UCF101 subset (`sayakpaul/ucf101-subset`) — 5 of its 10 real classes (**Basketball**, Archery, BenchPress, BabyCrawling, BandMarching) were used, since UCF101 has no literal "Walking" or "Running" category despite the manual's illustrative example list.

## 📑 Table of Contents

- [1. Imports & Setup](#1-imports--setup)
- [2. Preprocessing the UCI HAR Dataset](#2-preprocessing-the-uci-har-dataset)
- [3. Temporal Visualization & the BPTT Numerical Exercise](#3-temporal-visualization--the-bptt-numerical-exercise)
- [4. RNN / LSTM / GRU Architecture & Training](#4-rnn--lstm--gru-architecture--training)
- [5. Sequence Length Ablation Study](#5-sequence-length-ablation-study)
- [6. Video Understanding — CNN Feature Extraction](#6-video-understanding--cnn-feature-extraction)
- [7. CNN-LSTM / CNN-GRU Model](#7-cnn-lstm--cnn-gru-model)
- [8. Sequence to Sequence Learning](#8-sequence-to-sequence-learning)
- [9. Additional Exercise 1 — Recurrent Unit Sweep](#9-additional-exercise-1--recurrent-unit-sweep)
- [10. Additional Exercises 3 & 4 — Stacked & Bidirectional LSTM](#10-additional-exercises-3--4--stacked--bidirectional-lstm)
- [11. Additional Exercise 7 — Mismatched-Length Seq2Seq](#11-additional-exercise-7--mismatched-length-seq2seq)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Recommended Configuration](#-recommended-configuration)
- [References](#-references)

---

## 1. Imports & Setup

```python
import numpy as np, random, json, time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import cv2, matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

CLASS_NAMES = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
```

> **What's happening:** Loads TensorFlow/Keras for every model, scikit-learn for splits and evaluation metrics, and OpenCV for video frame sampling. A single `set_seed()` helper is called before every model build so RNN/LSTM/GRU and every exercise variant are trained from the same initialization, isolating the architecture as the only variable.

---

## 2. Preprocessing the UCI HAR Dataset

```python
def load_split(base_dir, split):
    channels = ['body_acc_x','body_acc_y','body_acc_z','body_gyro_x','body_gyro_y',
                'body_gyro_z','total_acc_x','total_acc_y','total_acc_z']
    arrs = [np.loadtxt(f'{base_dir}/{split}/Inertial Signals/{ch}_{split}.txt') for ch in channels]
    X = np.stack(arrs, axis=-1)                                   # (N, 128, 9)
    y = np.loadtxt(f'{base_dir}/{split}/y_{split}.txt').astype(int) - 1
    return X, y

# stratified 400-windows-per-class subset (2400 total), inside the manual's 1500-3000 range
for c in range(6):
    idx_c = np.where(y_train_full == c)[0]
    sel = rng.choice(idx_c, size=400, replace=False)

X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.15/0.85,
                                                    stratify=y_trainval, random_state=42)
mean, std = X_train.mean(axis=(0,1), keepdims=True), X_train.std(axis=(0,1), keepdims=True) + 1e-8
```

> **What's happening:** Loads all 9 raw inertial channels (body acceleration, body gyroscope, total acceleration, ×3 axes each) at their native 128-timestep window length, builds a stratified 2400-window subset (**exactly 400 windows per class**), then splits 70/15/15 → **1680 / 360 / 360** windows, stratified so every split stays perfectly balanced (280/60/60 per class). Normalization statistics are computed from the training split only, then applied to validation and test.

---

## 3. Temporal Visualization & the BPTT Numerical Exercise

```python
h_t = np.tanh(Wx * x_t + Wh * h_prev + b)   # h0=0, Wx=0.5, Wh=0.8, b=0.1
```

> **What's happening:** Plots `body_acc_x`, `body_gyro_x`, and `total_acc_x` across WALKING, SITTING, and LAYING — WALKING shows a strong quasi-periodic gait oscillation while the two static postures are nearly flat, all the classification signal lives in the temporal structure. The BPTT numerical exercise is then computed by hand and cross-checked against a program: **h₁ = 0.336376, h₂ = 0.616352, h₃ = 0.599958** — matched to 6 decimal places, confirming the recurrence was applied correctly.

<p align="center"><img src="readme_figs/plot01_temporal.png" width="700"></p>

---

## 4. RNN / LSTM / GRU Architecture & Training

```python
def build_model(layer_cls):
    return keras.Sequential([
        layers.Input(shape=(128, 9)),
        layer_cls(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(6, activation='softmax'),
    ])

for name, layer_cls in {'RNN': layers.SimpleRNN, 'LSTM': layers.LSTM, 'GRU': layers.GRU}.items():
    model = build_model(layer_cls)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32)
```

> **What's happening:** Identical architecture, optimizer, learning rate, batch size and epoch count for all three recurrent cells — only the layer type changes, isolating its effect. RNN plateaus at **72.50% test accuracy** (1,974 params); LSTM reaches **96.11%** (6,006 params) and GRU reaches **95.28%** (4,758 params) — a >20 point jump from the same input and budget, direct evidence of the vanishing-gradient limitation of Vanilla RNN over 128 timesteps. The confusion matrices tell an even sharper story: RNN's errors concentrate among the three *dynamic* walking classes, while LSTM/GRU's only residual error is the classic *static*-posture SITTING vs. STANDING pair reported throughout the HAR literature — a completely different failure mode once gradients stop vanishing.

<p align="center">
<img src="readme_figs/plot02_loss.png" width="800"><br>
<img src="readme_figs/plot03_accuracy.png" width="800"><br>
<img src="readme_figs/plot04_confusion.png" width="800"><br>
<img src="readme_figs/plot05_comparison.png" width="600">
</p>

---

## 5. Sequence Length Ablation Study

```python
for T in [32, 64, 128]:
    X_trunc = X_train[:, :T, :]              # truncate to the first T timesteps
    for name, layer_cls in RECURRENT.items():
        model = build_model(layer_cls, T)
        model.fit(X_trunc, y_train, epochs=30, batch_size=32, verbose=0)
```

> **What's happening:** Each architecture is retrained at three sequence lengths, all else unchanged. RNN and the gated architectures move in **opposite directions**: RNN's F1 *falls* monotonically as T grows (**80.3% → 74.2% → 72.6%**), while LSTM (**92.5% → 96.1%**) and GRU (**94.5% → 95.3%**) both *improve* with more context. This is the cleanest empirical signature of the BPTT gradient-flow argument in the whole experiment — more timesteps means strictly worse vanishing gradients for RNN, but more usable temporal signal for the gated cells.

<p align="center"><img src="readme_figs/plot06_seqlen.png" width="600"></p>

---

## 6. Video Understanding — CNN Feature Extraction

```python
base = keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False,
                                       weights='imagenet', pooling='avg')
base.trainable = False                                     # frozen — never fine-tuned

def sample_frames(path, n=10):
    idxs = np.linspace(0, total_frames-1, n).astype(int)    # 10 frames, uniformly spaced
    ...

feats = base.predict(preprocess(sample_frames(video_path)))  # (10, 1280) per clip
```

> **What's happening:** 10 frames sampled uniformly from each of 210 real UCF101 clips, resized to `224×224×3`, passed through a **frozen** MobileNetV2 (ImageNet weights, no fine-tuning at any point) to produce a **1280-d** feature vector per frame — so every video becomes a `(10, 1280)` tensor, mirroring the `(T, F)` shape the HAR sensor data used. Feature extraction for all 210 clips took **77.4s** on this 2-core CPU sandbox.

<p align="center"><img src="readme_figs/plot07_video_frames.png" width="800"></p>

---

## 7. CNN-LSTM / CNN-GRU Model

```python
def build_video_model(layer_cls, units=32):
    return keras.Sequential([
        layers.Input(shape=(10, 1280)),
        layer_cls(units),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax'),
    ])
# trained on the cached (10, 1280) feature sequences from Section 6, 25 epochs, batch size 8
```

> **What's happening:** The recurrent classifier trains entirely on the cached CNN features — CNN-LSTM reaches **97.78% test accuracy** (169,285 params) and CNN-GRU reaches **93.33%** (127,365 params) on held-out clips, with no fine-tuning of the CNN backbone at all. The only recurring confusion is Archery predicted as BenchPress (1-2 of 7 test clips) — both share a similar static, arms-raised silhouette across several sampled frames.

<p align="center">
<img src="readme_figs/plot08_video_curves.png" width="800"><br>
<img src="readme_figs/plot09_video_confusion.png" width="700">
</p>

---

## 8. Sequence to Sequence Learning

```python
enc_out, state_h, state_c = layers.LSTM(64, return_state=True)(encoder_embedding)
dec_out, _, _ = layers.LSTM(64, return_sequences=True, return_state=True)(
    decoder_embedding, initial_state=[state_h, state_c])
# trained with teacher forcing; evaluated with TRUE autoregressive greedy decoding:
def decode_greedy(x_seq):
    h, c = encoder_model.predict(x_seq[None, :])
    tok = SOS
    for _ in range(SEQ_LEN):
        pred, h, c = decoder_model.predict([tok, h, c])
        tok = np.argmax(pred)                 # feed the model's OWN prediction back in
```

> **What's happening:** An encoder-decoder LSTM learns to reverse 4-integer sequences (e.g. `[4,2,9,6] → [6,9,2,4]`). Trained with teacher forcing, but — critically — **evaluated with genuine autoregressive decoding**, where each predicted token is fed back in as the next input rather than the ground truth. A first pass at this reused the wrong embedding layer for single-step inference and silently collapsed to ~42% token accuracy despite 100% teacher-forced training accuracy; fixing the layer reference (`dec_embedding` instead of a fragile `model.get_layer(index=2)`) brought it to **100% token accuracy and 100% sequence accuracy** on the held-out test set.

---

## 9. Additional Exercise 1 — Recurrent Unit Sweep

```python
for units in [16, 32, 64]:
    model = build_model(layers.LSTM, units)
    model.fit(X_train, y_train, epochs=30, batch_size=32)
```

> **What's happening:** 16 units already matches 32 units' accuracy (**96.11% both**) at a third of the parameters (2,038 vs. 6,006); 64 units gives only a marginal further gain (**96.39%**) for more than 3× the parameter count — clear diminishing returns, the HAR task saturates recurrent capacity quickly at this dataset size.

---

## 10. Additional Exercises 3 & 4 — Stacked & Bidirectional LSTM

```python
stacked = keras.Sequential([layers.Input((128,9)), layers.LSTM(32, return_sequences=True),
                             layers.LSTM(32), layers.Dropout(0.2), ...])
bidir   = keras.Sequential([layers.Input((128,9)), layers.Bidirectional(layers.LSTM(32)),
                             layers.Dropout(0.2), ...])
```

> **What's happening:** A second stacked LSTM layer drops accuracy to **95.56%** (14,326 params), and a Bidirectional LSTM drops it to **95.83%** (11,894 params) — both slight regressions from the single-layer baseline's 96.11%. An informative negative result: the 1680-window training set isn't large enough to benefit from the added depth or backward context, so the simplest single-layer model wins on both accuracy and cost.

---

## 11. Additional Exercise 7 — Mismatched-Length Seq2Seq

```python
# input length 6, output length 4 — the reverse of the input with the first 2 tokens dropped
Y = X[:, ::-1][:, :4]
```

> **What's happening:** Confirms the encoder-decoder framework isn't tied to equal input/output lengths — the decoder's context comes entirely from the encoder's final state — reaching **100% token accuracy and 100% sequence accuracy** on this length-changing variant, same as the base task.

---

## 📊 Results

| Model | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Parameters |
|:---|:---:|:---:|:---:|:---:|:---:|
| RNN | 72.50% | 73.02% | 72.50% | 72.62% | 1,974 |
| LSTM | **96.11%** | 96.12% | 96.11% | 96.11% | 6,006 |
| GRU | 95.28% | 95.30% | 95.28% | 95.28% | 4,758 |
| CNN-LSTM (video) | **97.78%** | 98.00% | 97.14% | 97.41% | 169,285 |
| CNN-GRU (video) | 93.33% | 94.36% | 92.47% | 92.66% | 127,365 |

| Sequence Length | RNN F1 | LSTM F1 | GRU F1 |
|:---|:---:|:---:|:---:|
| 32 | **80.33%** | 92.49% | 94.45% |
| 64 | 74.23% | 93.65% | 95.26% |
| 128 | 72.62% | **96.11%** | **95.28%** |

| Task | Token Accuracy | Sequence Accuracy |
|:---|:---:|:---:|
| Seq2Seq (reversal) | 100% | 100% |
| Seq2Seq (mismatched length, Ex. 7) | 100% | 100% |

## 🔍 Key Findings

- **Gated units resolve the vanishing gradient problem in practice, not just in theory** — RNN plateaus at 72.5% while LSTM/GRU both clear 95%, from the same input, protocol, and training budget.
- **RNN and the gated architectures respond in *opposite* directions to longer sequences** — RNN's F1 falls as T grows from 32→128 while LSTM's and GRU's F1 rise, direct empirical confirmation of the BPTT gradient-flow argument.
- **Error modes shift entirely once gradients stop vanishing** — RNN confuses the three *dynamic* walking classes; LSTM/GRU's only residual error is the unrelated *static* SITTING vs. STANDING pair.
- **A frozen, off-the-shelf CNN is enough for video action recognition at this scale** — no fine-tuning of MobileNetV2 at all, yet CNN-LSTM reaches 97.8% test accuracy.
- **More capacity did not help this task** — 16 units matches 32 units; a second stacked layer and a Bidirectional LSTM both slightly *underperform* the simple single-layer baseline.

## ✅ Recommended Configuration

> Single-layer LSTM · 16–32 recurrent units · Adam optimizer · learning rate 1e-3 · batch size 32 for the HAR task; GRU as a lower-parameter alternative (79% of LSTM's params for 96.9% of its accuracy) when training/inference budget is tighter. CNN-LSTM over CNN-GRU for the video task, where the smaller 150-clip training set favors LSTM's extra cell-state capacity.

---

## 📚 References

1. Goodfellow, Bengio, Courville, *Deep Learning*, MIT Press, 2016.
2. Hochreiter & Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
3. Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," EMNLP, 2014.
4. Anguita et al., "A Public Domain Dataset for Human Activity Recognition Using Smartphones," ESANN, 2013.
5. Soomro, Zamir, Shah, "UCF101: A Dataset of 101 Human Actions Classes From Videos in The Wild," 2012.
6. TensorFlow and Keras Documentation
