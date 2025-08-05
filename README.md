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

The repository now includes optional support for a **personalised transformation layer** and **DP-SGD** training with a privacy accountant.

* Enable the transformation layer with `--use_transform_layer 1`. Each client learns its own affine layer `T_k(x) = α ⊙ x + β` that is excluded from model aggregation.
* Enable DP-SGD with `--use_dp 1`. The following arguments control the behaviour:
  * `--dp_clip`: clipping norm (default `1.0`)
  * `--dp_noise`: noise multiplier added after clipping
  * `--dp_delta`: target delta for privacy accounting (default `1e-5`)
  * `--print_eps`: output the current ε after each communication round when set to `1`

Example:

```
python main_image.py --dataset miniImageNet --use_transform_layer 1 --use_dp 1 --dp_clip 0.5 --dp_noise 0.2 --dp_delta 1e-5 --print_eps 1
```
When `--print_eps 1`, the final ε and δ are printed after training.


## Citation
Welcome to cite our work! </br>

> @inproceedings{wang2023federated,  
  title={Federated Few-shot Learning},  
  author={Wang, Song and Fu, Xingbo and Ding, Kaize and Chen, Chen and Chen, Huiyuan and Li, Jundong},  
  booktitle={SIGKDD},  
  year={2023}  
}
