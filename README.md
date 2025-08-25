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


## New Features

The repository now includes optional support for a **personalised transformation layer** and **differential privacy** with a privacy accountant.

* Enable the transformation layer with `--use_transform_layer 1`. Each client learns its own affine layer `T_k(x) = α ⊙ x + β` that is excluded from model aggregation.
* Differential privacy is controlled solely via `--dp_mode` (the deprecated `--use_dp` flag has been removed). DP-specific options are ignored when `--dp_mode off`:
  * `local` (default): apply DP-SGD on each client.
  * `server`: clip and noise client updates on the server.
  * `off`: disable differential privacy.
  * `--dp_clip`: clipping norm (default `1.0`)
  * `--dp_noise`: noise multiplier
  * `--dp_delta`: target delta for privacy accounting (default `1e-5`)
  * `--print_eps`: output the current ε after each communication round when set to `1`

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
When `--print_eps 1`, the current ε and δ are printed after each round.

* Enable server momentum with `--server_momentum <m>` and set the server learning rate with `--server_lr <lr>` to use FedAvgM for faster convergence. A good starting point is `lr ≈ 1 - m`.
* Select DP-compatible optimizers via `--optimizer`. In addition to `sgd`, `adam`, and `amsgrad`, `adamw` is supported and becomes DP-AdamW when differential privacy is enabled.
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
