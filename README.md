# F2L: Federated Few-shot Learning
Thank you for your interest in our work! </br>

This is the code for the paper *Federated Few-shot Learning*, published in SIGKDD 2023.  
  
![Alt text](./model_fed.png)

## Requirement:
```
torch==2.4.0
torchvision==0.19.0
torchtext==0.18.0
numpy
scikit-learn
tqdm
transformers
```


## Code Running:

First download the data file from [here](https://drive.google.com/file/d/1us-iQiY9YSDE9SOX9YohGmnbAyOghqMr/view?usp=sharing) and unzip it into the folder 'data'.  

To run the command for image datasets, i.e., 'miniImageNet' and 'FC100':
```
python main_image.py --dataset dataset_name
```

To run the command for text datasets, i.e., '20newsgroup' and 'huffpost':  
```
python main_text.py --dataset dataset_name
```
Note that the text model requires the GloVe embedding file named 'glove.42B.300d.zip', which should be put in the main folder. The download link is [here](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip).


## Metrics

The training scripts log both **pre-adaptation** and **post-adaptation** accuracies for each evaluation task:

* *Pre-adaptation* accuracy is computed by applying the global classifier to the query samples before any task-specific fine tuning.
* *Post-adaptation* accuracy is measured after fitting a task-specific classifier on the support set (logistic regression by default) and evaluating on the query set.

During training, the "global" accuracy reported for each round is the maximum post-adaptation accuracy across clients. Tracking the pre-adaptation metric alongside the post-adaptation one helps diagnose cases where the global accuracy appears stagnant even though the base model continues to improve.


## New Features

The repository now includes optional support for a **personalised transformation layer** and **differential privacy** with a privacy accountant.

* Enable the transformation layer with `--use_transform_layer 1`. Each client learns its own affine layer `T_k(x) = α ⊙ x + β` that is excluded from model aggregation.
* Differential privacy is controlled solely via `--dp_mode` (the deprecated `--use_dp` flag has been removed). DP-specific options are ignored when `--dp_mode off`:
  * `local` (default): apply DP-SGD on each client.
  * `server`: clip and noise client updates on the server.
  * `off`: disable differential privacy.
  * `--dp_clip`: clipping norm (default `1.0`)
  * `--dp_k_min`: floor multiplier for per-layer norms (default `0.2`)
  * `--dp_init_scale`: initial clip scaling (default `0.6`)
  * `--dp_clip_max`: maximum clip (default `0.8`)
  * `--no_client_clipping`: disable per-client clipping while retaining server-side DP
  * `--dp_target_mean_scale`: desired mean clipping scale (default `0.6`)
  * `--dp_depth_decay`: exponential clip target decay per network depth (default `1.0`, e.g. `--dp_depth_decay=0.85` so deeper blocks get progressively tighter clip targets)
  * `--dp_adapt_gain`: adaptation gain (default `0.1`)
  * `--dp_adapt_period`: rounds between clip adaptations (default `3`)
  * `--dp_bootstrap`: bootstrap clip from the first round's median norm (default `True`)
  * `--dp_noise`: fixed noise multiplier (default) **or** `--dp_noise_scale` to scale noise with the clip via `sigma = scale * clip`
  * `--dp_delta`: target delta for privacy accounting (default `1e-5`)
  * `--dp_accountant`: privacy accountant used to estimate ε (`rdp` default). Set to `prv` to use the PRV accountant, which requires installing [`prv_accountant`](https://github.com/microsoft/prv_accountant) (e.g. `pip install prv-accountant`).
  * `--print_eps`: output the current ε after each communication round when set to `1`
  * ε uses the minimum σ<sub>l</sub> across layers when noise multipliers differ
  * per-round noise statistics are written to `*_noise.csv`; `noise_std_rep` is the mean per-parameter noise standard deviation derived from the current layer clips and noise multipliers (after step scaling) and `noise_norm` is the total L2 norm of the sampled noise

Examples:

```
# Local DP
python main_image.py --dataset miniImageNet --dp_mode local --dp_noise 0.2
# Server DP
python main_image.py --dataset miniImageNet --dp_mode server --dp_noise 0.2
# No DP
python main_image.py --dataset miniImageNet --dp_mode off
# FedAvgM with server learning rate
python main_image.py --dataset miniImageNet --server_momentum 0.9 --server_lr 0.1
```
When `--print_eps 1`, the current ε and δ are printed after each round using the minimum noise-to-clip ratio across layers.

* Enable server momentum with `--server_momentum <m>` and set the server learning rate with `--server_lr <lr>` to use FedAvgM for faster convergence. A good starting point is `lr ≈ 1 - m`.
* Cap the server update norm with `--target_step <s>` to limit the L2 norm of the aggregated step after scaling. Values around `1.0` work well; tune by monitoring the logged `||u||` and set `s` either as an absolute norm or as a fraction of the typical signal norm.
* Configure learning rate schedules with `--lr_schedule` and `--lr_decay`. Only `cosine` is currently supported. A suggested starting point is:

    ```bash
    python main_image.py --dataset miniImageNet --lr 0.001 --lr_schedule cosine --lr_decay 0.1
    ```
  This applies a cosine schedule that anneals the learning rate for both DP and head optimizers to 10% of its initial value.
* Select DP-compatible optimizers via `--optimizer`. In addition to `sgd`, `adam`, and `amsgrad`, `adamw` is supported and becomes DP-AdamW when differential privacy is enabled.
* Enable mixed precision with `--use_amp` on CUDA devices to speed up training and reduce memory use. Convolutional layers run under autocast (dtype via `--amp_dtype`) while losses and backprop stay in full precision for stability.
* Training can stop early when global accuracy plateaus. Set `--convergence_patience` and `--convergence_delta` to monitor convergence and exit before reaching the full `--comm_round`.

### Choosing a noise multiplier for a target ε

The helper `find_noise_multiplier` in `dp_utils.py` searches for a noise multiplier that achieves a desired privacy budget:

```python
from dp_utils import find_noise_multiplier

sigma = find_noise_multiplier(
    num_steps=100,           # total number of DP steps
    target_eps=1.0,          # desired epsilon
    delta=1e-5,
    accountant='rdp',
    sampling_rate=0.01,      # client sampling rate q
)
```

Use the returned `sigma` as the value for `--dp_noise` when running the training scripts.


## Citation
Welcome to cite our work! </br>

> @inproceedings{wang2023federated,  
  title={Federated Few-shot Learning},  
  author={Wang, Song and Fu, Xingbo and Ding, Kaize and Chen, Chen and Chen, Huiyuan and Li, Jundong},  
  booktitle={SIGKDD},  
  year={2023}  
}
