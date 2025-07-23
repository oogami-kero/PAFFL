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

## Privacy option
This repository includes an optional **Delta-DP** mechanism which applies central differential privacy on client updates. To enable it, specify a clipping norm and noise multiplier:

```
--clip_norm 1.0 --noise_multiplier 0.5 --dp_delta 1e-5
```

During training each client update is clipped to `clip_norm` and Gaussian noise with standard deviation `clip_norm * noise_multiplier` is added before aggregation. The scripts print a few values of each client's delta before and after noise as well as an approximate privacy `epsilon`.

## Citation
Welcome to cite our work! </br>

> @inproceedings{wang2023federated,  
  title={Federated Few-shot Learning},  
  author={Wang, Song and Fu, Xingbo and Ding, Kaize and Chen, Chen and Chen, Huiyuan and Li, Jundong},  
  booktitle={SIGKDD},  
  year={2023}  
}
