# F2L: Federated Few-shot Learning
Thank you for your interest in our work! </br>

This is the code for the paper *Federated Few-shot Learning*, published in SIGKDD 2023.  
  
![Alt text](./model_fed.png)

## Requirement:
```
torch==1.11.0+cu113
torchvision==0.12.0+cu113  
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

## Differential Privacy Aggregation (Updated)

When central DP (`--dp_enable 1`) is enabled, the server now:

- Aggregates only the tensors selected by `--dp_target`; all other parameters remain frozen on the server between rounds.
- Clips each participating client update to `dp_clip_norm`, averages the clipped updates with equal weight `1/m` over the `m` sampled clients, and adds Gaussian noise with standard deviation `sigma * dp_clip_norm / m`.
- Tracks privacy loss with a subsampled Gaussian RDP accountant using sampling rate `q = m / N` (where `N` is the total number of clients) and client-level `δ = N^{-1.1}`. The per-round and cumulative ε values, along with `m`, `q`, `sigma`, the effective noise standard deviation, signal-to-noise ratio, and clipping rate, are logged every round.
- Seeds per-parameter noise with `--dp_seed` (when non-negative); use unique seeds across experiments to maintain independence of the added noise.

## Citation
Welcome to cite our work! </br>

> @inproceedings{wang2023federated,  
  title={Federated Few-shot Learning},  
  author={Wang, Song and Fu, Xingbo and Ding, Kaize and Chen, Chen and Chen, Huiyuan and Li, Jundong},  
  booktitle={SIGKDD},  
  year={2023}  
}
