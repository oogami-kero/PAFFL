Few-Shot Federated Learning with Client-Side DP and Private Transform
=============================================================

This repository extends the F2L few-shot federated learning framework with:

- Client-side user-level differential privacy on client model updates
- A client-private transform layer inspired by PrivateFL, kept local and excluded from aggregation/DP
- Rich console + file logging (ε per round, signal/noise norms, SNR, clip ratios, transform stats)

The implementation targets thesis-style experiments to compare against baselines and published methods.

Features
--------
- DP mechanism: per-client L2 clipping + Gaussian noise on model deltas, before upload
- Epsilon accounting: integer-order RDP across rounds, with per-round and cumulative ε@δ logging
- Private transform: linear layer applied to embeddings (default ResNet12 path), never aggregated or DP’d
- Adjustable via CLI flags: DP clip/noise/delta/orders; transform dim/init/lr/wd; logging level

Quickstart
----------
1) Environment (CUDA 11.8 + PyTorch 2.4.0)

- Conda (example):
  - conda create -n dp_ffsl python=3.10 -y
  - conda activate dp_ffsl
  - pip install --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0
  - pip install numpy<2 scikit-learn pillow prv-accountant torchtext==0.18.0

2) Datasets
- FC100 (CIFAR100-based): will auto-download into `--datadir` on first run
- miniImageNet: place `mini-imagenet-cache-train.pkl` and `mini-imagenet-cache-test.pkl` into `--datadir`
- 20newsgroup / huffpost: use `main_text.py` and ensure data loader-compatible layout under `data/`

3) Smoke test (CPU)

```
python main_image.py \
  --dataset FC100 --datadir ./data \
  --dp --dp_noise_multiplier 1.2 --dp_clip_norm 1.0 --dp_delta 1e-6 \
  --comm_round 1 --sample_fraction 0.2 \
  --epochs 1 --num_train_tasks 1 --num_test_tasks 1 --num_true_test_ratio 1 \
  --N 2 --K 1 --Q 1 \
  --transform --transform_dim 640 --transform_lr 1e-3 --transform_wd 1e-4 \
  --logdir ./logs --modeldir ./models --log_file_name dp_test \
  --device cpu
```

4) GPU example

```
python main_image.py ... --device cuda:0
```

DP Controls
-----------
- `--dp`: enable client-side DP on updates
- `--dp_clip_norm`: L2 clip norm (default 1.0)
- `--dp_noise_multiplier`: Gaussian noise multiplier σ (default 1.2)
- `--dp_delta`: target δ for ε computation (default 1e-6)
- `--dp_orders`: comma-separated integer RDP orders (default `2,3,4,5,6,8,10,12,16,32,64,128`)

Logging additions:
- Per-client: signal norm, clipped norm, clip coefficient, noise norm, SNR
- Per-round: average signal/noise/SNR, clip ratio, sample rate, ε per-round and cumulative

Accountant: integer-order RDP implementation in `dp_utils.py`. You can swap to PRV if desired (we installed `prv-accountant`), but the default logs use the provided RDP accountant.

Private Transform Controls
--------------------------
- `--transform`: enable private transform
- `--transform_type`: `linear` or `none` (default `linear`)
- `--transform_dim`: output dimension (default: backbone dim)
- `--transform_init`: `orthogonal` (default), `xavier_uniform`, `xavier_normal`, `identity`
- `--transform_lr`, `--transform_wd`: optimizer group for private transform parameters

The transform is applied after the transformer encoder in `ModelFed_Adp` for ResNet12. Parameters under `private_transform` are:
- Not aggregated
- Not included in DP update perturbation
- Logged for weight/grad norms after updates

What to Look For in Logs
------------------------
- DP client lines, e.g.:
  - `DP client 6 | signal=... | clipped=... | clip_coef=... | noise=... | snr=...`
- Round summary and ε:
  - `DP round 1 | sample_rate=... | epsilon_round=... | epsilon_total=... | delta=...`
- Transform layer:
  - `Transform | client=... | weight_norm=... | grad_norm=... | lr=... | wd=...`

Console mirrors the log file. Adjust with `--console_log_level`.

Suggested Experiments
---------------------
- Baseline: no DP (`--dp` off)
- Vary noise: `--dp_noise_multiplier {0.5, 1.0, 1.2, 2.0}` and observe accuracy/SNR and ε
- Transform ablation: toggle `--transform` and measure impact

Repository Structure (key changes)
----------------------------------
- `dp_utils.py`: DP mechanism + integer-order RDP accountant
- `main_image.py`, `main_text.py`: DP/transform CLI flags, aggregation hooks, logging
- `model.py`: `PrivateTransform` and integration into ResNet12 path for few-shot

References
----------
- F2L (base): https://github.com/SongW-SW/F2L
- PrivateFL (transform idea): https://github.com/BHui97/PrivateFL
- Paper: Yuchen Yang et al. "When Federated Learning Meets Private Data Transformation" (USENIX Security 2023)
  - https://www.usenix.org/system/files/usenixsecurity23-yang-yuchen.pdf

This project is for research/educational purposes and builds upon the above works.

Pushing to GitHub
-----------------
1) Create a new empty GitHub repo (e.g., `F2L-DP-Transform`).
2) From this project root:

```
git init
git add -A
git commit -m "Add client-side DP + private transform + logging"
git branch -M main
git remote add origin <your-new-repo-url>
git push -u origin main
```

Use SSH or HTTPS + personal access token per your setup.

If you prefer, I can run the git commands for you once you share the repo URL (and confirm SSH or HTTPS).

