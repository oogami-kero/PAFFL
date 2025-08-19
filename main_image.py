import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import logging
import os
import copy
import datetime
import random
import time

from PIL import Image

from model import *
from model import WordEmbed
from utils import *
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
import dp_utils
from dp_utils import remove_dp_hooks
import warnings
from data.class_mappings import fine_id_coarse_id, coarse_id_fine_id, coarse_split

warnings.filterwarnings('ignore')

from collections import defaultdict

fine_split=defaultdict(list)

for fine_id,sparse_id in fine_id_coarse_id.items():
    if sparse_id in coarse_split['train']:
        fine_split['train'].append(fine_id)
    elif sparse_id in coarse_split['valid']:
        fine_split['valid'].append(fine_id)  
    else:
        fine_split['test'].append(fine_id)  

#fine_split_train_map={class_:i for i,class_ in enumerate(fine_split['train'])}
        
#train_class2id={class_id: i for i, class_id in enumerate(fine_split['train'])}
        
        
import torchvision.transforms as transforms

#FC100
normalize_fc100 = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])

#miniImageNet
mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
normalize_mini = transforms.Normalize(mean=mean_pix,
                                 std=std_pix)


# transform_train = transforms.Compose([
#     transforms.RandomCrop(32),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize
# ])

def transform_train(normalize, crop_size=None, padding=None):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(crop_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize
    ])


# data prep for test set
def transform_test(normalize):
    return transforms.Compose([
        transforms.ToTensor(),
        normalize])


#transform_train=transform_test
def l2_normalize(x):
    norm = (x.pow(2).sum(1, keepdim=True)+1e-9).pow(1. / 2)
    out = x.div(norm+1e-9)
    return out


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float).cuda()
    neg_mask = 1. - pos_mask
    sim = _similarity(anchor, sample / temperature_matrix if temperature_matrix != None else sample) / tau
    exp_sim = torch.exp(sim) * (pos_mask + neg_mask)

    if not all_negative:
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True)+1e-9)
    else:
        log_prob = - torch.log(exp_sim.sum(dim=1, keepdim=True)+1e-9)

    loss = log_prob * pos_mask
    loss = loss.sum(dim=1) / pos_mask.sum(dim=1)

    return -loss.mean(), sim

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet12', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='FC100', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.01, 0.0005, 0.005)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='communication strategy: fedavg/fedprox')
    
    parser.add_argument('--method', type=str, default='new',
                        help='few-shot or normal')
    parser.add_argument('--mode', type=str, default='few-shot',
                        help='few-shot or normal')
    parser.add_argument('--N', type=int, default=5, help='number of ways')
    parser.add_argument('--K', type=int, default=5, help='number of shots')
    parser.add_argument('--Q', type=int, default=5, help='number of queries')   
    parser.add_argument('--num_train_tasks', type=int, default=50, help='number of meta-training tasks (5)')
    parser.add_argument('--num_test_tasks', type=int, default=10, help='number of meta-test tasks')
    parser.add_argument('--num_true_test_ratio', type=int, default=10, help='number of meta-test tasks (10)')
    parser.add_argument('--fine_tune_steps', type=int, default=5, help='number of meta-learning steps (5)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1, help='number of meta-learning lr (0.05)')
    parser.add_argument('--meta_lr', type=float, default=0.1/100, help='number of meta-learning lr (0.05)')
    parser.add_argument('--comm_round', type=int, default=5000, help='number of maximum communication roun')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    
    
    parser.add_argument("--bert_cache_dir", default=None, type=str,
                        help=("path to the cache_dir of transformers"))
    parser.add_argument("--pretrained_bert", default=None, type=str,
                        help=("path to the pre-trained bert embeddings."))
    parser.add_argument("--wv_path", type=str,
                        default="./",
                        help="path to word vector cache")
    parser.add_argument("--word_vector", type=str, default="wiki.en.vec",
                        help=("Name of pretrained word embeddings."))
    parser.add_argument("--finetune_ebd", type=bool, default=False)
    # induction networks configuration
    parser.add_argument("--induct_rnn_dim", type=int, default=128,
                        help=("Uni LSTM dim of induction network's encoder"))
    parser.add_argument("--induct_hidden_dim", type=int, default=100,
                        help=("tensor layer dim of induction network's relation"))
    parser.add_argument("--induct_iter", type=int, default=3,
                        help=("num of routings"))
    parser.add_argument("--induct_att_dim", type=int, default=64,
                        help=("attention projection dim of induction network"))
    
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=1,  #0.5
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')

    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100, help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1, help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_pool_file', type=str, default=None, help='the old model pool path to load')
    parser.add_argument('--load_model_round', type=int, default=None, help='how many rounds have executed for the loaded model')
    parser.add_argument('--load_first_net', type=int, default=1, help='whether load the first net as old net or not')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--loss', type=str, default='contrastive')
    parser.add_argument('--save_model',type=int,default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--use_transform_layer', type=int, default=0,
                        help='enable personalized transformation layer')
    parser.add_argument('--use_dp', type=int, default=0, help='enable DP-SGD')
    parser.add_argument('--dp_clip', type=float, default=1.0, help='DP-SGD clipping norm')
    parser.add_argument('--dp_noise', type=float, default=0.0, help='DP-SGD noise multiplier')
    parser.add_argument('--dp_delta', type=float, default=1e-5, help='target delta for DP accountant')
    parser.add_argument('--dp_mode', choices=['local', 'server', 'off'], default='server')
    parser.add_argument('--dp_accountant', choices=['rdp', 'prv'], default='rdp',
                        help='DP accountant to estimate the privacy budget')
    parser.add_argument('--print_eps', type=int, default=0, help='print final privacy budget')
    args = parser.parse_args()
    args.use_dp = int(args.dp_mode != 'off')
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100' or args.dataset=='FC100' :
        total_classes=60 #100
    elif args.dataset=='miniImageNet':
        total_classes=64
    elif args.dataset == '20newsgroup':
        total_classes=8
    elif args.dataset=='fewrel':
        total_classes=len([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                                 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                                 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                 76, 77, 78])
    elif args.dataset=='huffpost':
        total_classes=20

    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 26
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset == 'xray':
        n_classes = 2

    if args.mode=='few-shot':
        if args.dataset=='FC100':
            n_classes=args.N*4
        else:
            n_classes=args.N*4
        
    if args.mode=='few-shot' and args.method=='new':
        if args.dataset=='20newsgroup':
            ebd=WordEmbed(args.finetune_ebd)
        for net_i in range(n_parties):
            if args.dataset=='FC100' or args.dataset=='miniImageNet':
                net = ModelFed_Adp(args.model, args.out_dim, n_classes, total_classes, net_configs, args)
            else:
                net = LSTMAtt(WordEmbed(args.finetune_ebd), args.out_dim, n_classes, total_classes,args)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

            
    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


def train_net_few_shot_new(net_id, net, n_epoch, lr, args_optimizer, args, X_train_client, y_train_client, X_test, y_test,
                                        device='cpu', test_only=False, test_only_k=0):
    base_model = net
    base_model.train()
    gmodel = base_model
    model_template = copy.deepcopy(base_model)   # DP-free template

    client_sample_size = len(y_train_client)

    dp_named_params = [
        (n, p) for n, p in base_model.named_parameters()
        if 'transform_layer' not in n and 'few_classify' not in n and 'transformer' not in n and p.requires_grad
    ]
    dp_params = [p for _, p in dp_named_params]
    head_params = list(base_model.few_classify.parameters())
    tl_params = [p for n, p in base_model.named_parameters() if 'transform_layer' in n and p.requires_grad]

    if args_optimizer == 'adam':
        dp_optimizer = optim.Adam(dp_params, lr=lr, weight_decay=args.reg)
        head_optimizer = optim.Adam(head_params, lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        dp_optimizer = optim.Adam(
            dp_params,
            lr=lr,
            weight_decay=args.reg,
            amsgrad=True,
        )
        head_optimizer = optim.Adam(
            head_params,
            lr=lr,
            weight_decay=args.reg,
            amsgrad=True,
        )
    elif args_optimizer == 'sgd':
        dp_optimizer = optim.SGD(
            dp_params,
            lr=lr,
            momentum=0.9,
            weight_decay=args.reg,
        )
        head_optimizer = optim.SGD(
            head_params,
            lr=lr,
            momentum=0.9,
            weight_decay=args.reg,
        )

    orig_optimizer = dp_optimizer

    N, K, Q = get_n_k_q(args, mode='train', fewrel_multiplier=3)
    total_batch = N * K + N * Q
    sample_rate = total_batch / client_sample_size

    privacy_engine = None
    if args.dp_mode == 'local' and dp_params:
        noise_mult = getattr(args, 'dp_noise', 0.0)
        clip = getattr(args, 'dp_clip', 1.0)
        privacy_engine = PrivacyEngine(accountant='rdp')
        dummy_loader = DataLoader(
            TensorDataset(torch.zeros(client_sample_size, 1)),
            batch_size=total_batch,
            shuffle=True,
        )
        gmodel, dp_optimizer, _ = privacy_engine.make_private(
            module=base_model,
            optimizer=dp_optimizer,
            data_loader=dummy_loader,
            noise_multiplier=noise_mult,
            max_grad_norm=clip,
        )
        dp_optimizer.sample_rate = sample_rate
        dp_optimizer.expected_batch_size = total_batch
    dp_named_params = [
        (n, p) for n, p in gmodel.named_parameters()
        if 'transform_layer' not in n and 'few_classify' not in n and 'transformer' not in n and p.requires_grad
    ]
    if not hasattr(args, 'grad_norms_ma'):
        args.grad_norms_ma = {}
    for name, _ in dp_named_params:
        args.grad_norms_ma.setdefault(name, 0.0)
    grad_ma_decay = 0.9
    tl_optimizer = None
    if tl_params:
        tl_optimizer = optim.SGD(tl_params, lr=lr, momentum=0.9, weight_decay=args.reg)

    if args.dataset == 'FC100':
        X_transform_train = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize_fc100,
        ])
        X_transform_test = transform_test(normalize=normalize_fc100)
    else:
        X_transform_train = transforms.Compose([
            lambda x: Image.fromarray(x),
            transforms.RandomCrop(84, padding=8),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            lambda x: np.asarray(x),
            transforms.ToTensor(),
            normalize_mini,
        ])
        X_transform_test = transform_test(normalize=normalize_mini)

    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    result = None
    epsilon = None
    try:

        def train_epoch(epoch, mode='train'):
            nonlocal dp_optimizer, head_optimizer, tl_optimizer, gmodel, base_model
    
            if mode == 'train':
                N, K, Q = get_n_k_q(args, mode='train', fewrel_multiplier=3)
                gmodel.train()
                gmodel.zero_grad(set_to_none=True)  # clears param.grad and param.grad_sample
                dp_optimizer.zero_grad()
                head_optimizer.zero_grad()
                if tl_optimizer is not None:
                    tl_optimizer.zero_grad()
                X_transform = X_transform_train
            else:
                N, K, Q = get_n_k_q(args, mode='test', fewrel_multiplier=3)
                gmodel.eval()
                X_transform = X_transform_test
    
            if test_only==True:
                K=test_only_k
    
            support_batch = N * K
            query_batch = N * Q
            total_batch = support_batch + query_batch
            sample_rate = total_batch / client_sample_size
            if args.dp_mode == 'local':
                dp_optimizer.expected_batch_size = total_batch
                dp_optimizer.sample_rate = sample_rate
    
            support_labels = torch.zeros(N * K, dtype=torch.long)
            for i in range(N):
                support_labels[i * K:(i + 1) * K] = i
            query_labels = torch.zeros(N * Q, dtype=torch.long)
            for i in range(N):
                query_labels[i * Q:(i + 1) * Q] = i
            if args.device != 'cpu':
                support_labels = support_labels.cuda()
                query_labels = query_labels.cuda()
    
            if mode == 'train':
                if args.dataset=='FC100':
                    class_dict = fine_split['train']
                elif args.dataset=='miniImageNet':
                    class_dict=list(range(64))
                elif args.dataset=='20newsgroup':
                    class_dict=[1, 5, 10, 11, 13, 14, 16, 18]
                elif args.dataset=='fewrel':
                    class_dict = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                     76, 77, 78]
                elif args.dataset=='huffpost':
                    class_dict=list(range(20))
    
                X=X_train_client
                y=y_train_client
                #for i in class_dict:  
                    #class_dict[i] = class_dict[i][:avail_train_num_per_class]
            elif mode == 'test':
                if args.dataset=='FC100':
                    class_dict = fine_split['test']
                elif args.dataset=='miniImageNet':
                    class_dict=list(range(20))
                elif args.dataset=='20newsgroup':
                    class_dict=[0, 2, 3, 8, 9, 15, 19]
                elif args.dataset=='fewrel':
                    class_dict = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]
                elif args.dataset=='huffpost':
                    class_dict=list(range(25, 41))
    
                X=X_test
                y=y_test
    
            min_size=0
            while min_size<K+Q:
                X_class=[]
                classes = np.random.choice(class_dict, N, replace=False).tolist()
                for i in classes:
                    X_class.append(X[y==i])      
                min_size=min([one.shape[0] for one in X_class])
    
            X_total_sup=[]
            X_total_query=[]
            y_sup=[]
            y_query=[]
            transformed_class_list=[]
            for class_, X_class_i in zip(classes, X_class):
                sample_idx=np.random.choice(list(range(X_class_i.shape[0])), K+Q, replace=False).tolist()
                X_total_sup.append(X_class_i[sample_idx[:K]])
                X_total_query.append(X_class_i[sample_idx[K:]])
                if mode=='train':
                    if args.dataset=='FC100' or args.dataset=='20newsgroup' or args.dataset=='fewrel' or args.dataset=='huffpost':
                        transformed_class_list.append(fine_split_train_map[class_])
                        y_sup.append(torch.ones(K)*fine_split_train_map[class_])
                        y_query.append(torch.ones(Q) * fine_split_train_map[class_])
                    elif args.dataset=='miniImageNet':
                        transformed_class_list.append(class_)
                        y_sup.append(torch.ones(K)*class_)
                        y_query.append(torch.ones(Q) * class_)
    
    
    
    
                    y_total = torch.cat([torch.cat(y_sup, 0), torch.cat(y_query, 0)], 0).long().cuda()
            #y_total=torch.tensor(np.concatenate([np.concatenate(y_sup, 0),np.concatenate(y_query, 0)],0)).cuda()
            
            X_total_sup=np.concatenate(X_total_sup, 0)
            X_total_query=np.concatenate(X_total_query,0)
    
    
            if args.dataset=='FC100' or args.dataset=='miniImageNet':
                X_total_transformed_sup=[]
                X_total_transformed_query=[]
                for i in range(X_total_sup.shape[0]):
                    X_total_transformed_sup.append(X_transform(X_total_sup[i]))
                X_total_sup=torch.stack(X_total_transformed_sup,0).cuda()
    
                for i in range(X_total_query.shape[0]):
                    X_total_transformed_query.append(X_transform(X_total_query[i]))
                X_total_query=torch.stack(X_total_transformed_query,0).cuda()
            else:
                X_total_sup=torch.tensor(X_total_sup).cuda()
                X_total_query=torch.tensor(X_total_query).cuda()
    
    
    
    
            #net.load_state_dict(net_para_ori)
            #_,_,out_all=net_new(torch.cat([X_total_sup, X_total_query],0), all_classify=True)
    
                #print(out[:3])
            if mode == 'train':
                loss_all = 0
                if args.fine_tune_steps > 0:
                    gmodel_base = gmodel._module if hasattr(gmodel, '_module') else gmodel
                    net_new = copy.deepcopy(model_template)
                    net_new.load_state_dict(gmodel_base.state_dict())

                    for j in range(args.fine_tune_steps):
                        net_new.zero_grad()
                        X_out_sup, X_transformer_out_sup, out = net_new(X_total_sup)
                        losses = F.cross_entropy(out, support_labels, reduction='none')

                        params_to_update = []
                        for name, param in net_new.named_parameters():
                            if name in ('few_classify.weight', 'few_classify.bias') and param.requires_grad:
                                params_to_update.append(param)

                        losses.mean().backward()
                        with torch.no_grad():
                            for param in params_to_update:
                                if param.grad is None:
                                    continue
                                param.data.add_(-args.fine_tune_lr * param.grad)
    
                    X_out_query, _, out = net_new(X_total_query)
                    X_out_sup, X_transformer_out_sup, _ = net_new(X_total_sup)

                    X_transformer_out_sup = X_transformer_out_sup.reshape([N, K, -1]).transpose(0, 1)
                    for name, param in gmodel.named_parameters():
                        if 'transformer' in name:
                            param.requires_grad_(False)
                    X_out_all, x_all, out_all = gmodel(torch.cat([X_total_sup, X_total_query], 0), all_classify=True)
                    for name, param in gmodel.named_parameters():
                        if 'transformer' in name:
                            param.requires_grad_(True)
                    out_sup = X_out_all[:N * K].reshape([N, K, -1]).transpose(0, 1)
                    out_query = X_out_all[N * K:].reshape([N, Q, -1]).transpose(0, 1)
                    #############################
                    # Q=K here update for all-model
                    for j in range(Q):
                        contras_loss, similarity = InforNCE_Loss(X_transformer_out_sup[j], out_sup[(j+1) % Q],
                                                                 tau=0.5)
                        loss_all += contras_loss / Q * 0.1
                    loss_all += loss_ce(out_all, y_total)
                    loss_all.backward()
                    if args.dp_mode == 'local':
                        for name, param in dp_named_params:
                            if param.grad is None:
                                continue
                            grad_norm = param.grad.detach().norm(2).item()
                            prev = args.grad_norms_ma.get(name, grad_norm)
                            args.grad_norms_ma[name] = grad_ma_decay * prev + (1 - grad_ma_decay) * grad_norm
                    dp_optimizer.step()
                    head_optimizer.step()
                    if tl_optimizer is not None:
                        tl_optimizer.step()
                    ############################
    
                    for name, param in gmodel.named_parameters():
                        if 'transformer' in name:
                            param.requires_grad_(False)
                    if isinstance(gmodel, GradSampleModule):
                        gmodel.disable_hooks()
                    with torch.no_grad():
                        X_out_all, x_all, out_all = gmodel(torch.cat([X_total_sup, X_total_query], 0), all_classify=True)
                    if isinstance(gmodel, GradSampleModule):
                        gmodel.enable_hooks()
                    for name, param in gmodel.named_parameters():
                        if 'transformer' in name:
                            param.requires_grad_(True)
                    ###################################
                    # few_classify update
                    params_to_update = []
                    for name, param in net_new.named_parameters():
                        if name in ('few_classify.weight', 'few_classify.bias') or 'transformer' in name:
                            params_to_update.append((name, param))

                    #meta-update few-classifier on query
                    losses = F.cross_entropy(out, query_labels, reduction='none')
                    out_sup_on_N_class = out_all[N * K:, transformed_class_list]
                    out_sup_on_N_class /= out_sup_on_N_class.sum(-1, keepdim=True)
                    aux_loss = F.cross_entropy(out, out_sup_on_N_class.detach(), reduction='none') * 0.1
                    losses = losses + aux_loss
                    net_new.zero_grad()
                    losses.mean().backward()
                    with torch.no_grad():
                        gmodel_base = gmodel._module if hasattr(gmodel, '_module') else gmodel
                        gmodel_params = dict(gmodel_base.named_parameters())
                        for name, param in params_to_update:
                            if param.grad is None:
                                continue
                            gmodel_params[name].data.add_(-args.meta_lr * param.grad)
                    base_model.load_state_dict(gmodel_base.state_dict())
                    ##################################
                    del net_new, X_out_query, out
    
                if np.random.rand() < 0.005:
                    print('loss: {:.4f}'.format(loss_all.item()))
    
    
                acc_train = (torch.argmax(out_all, -1) == y_total).float().mean().item()
    
                del X_out_all,  out_all
                return acc_train
    
            else:
                use_logistic=True
    
                if use_logistic:
                    with torch.no_grad():
                        X_out_all, x_all, out_all = gmodel(torch.cat([X_total_sup, X_total_query], 0))
                        X_out_sup = X_out_all[:N * K]
                        X_out_query = X_out_all[N * K:]

                    support_features = torch.nan_to_num(l2_normalize(X_out_sup), nan=0.0, posinf=0.0, neginf=0.0)
                    query_features = torch.nan_to_num(l2_normalize(X_out_query), nan=0.0, posinf=0.0, neginf=0.0)

                    clf = LogisticRegression(support_features.size(1), N).to(support_features.device)
                    clf.fit(support_features, support_labels, max_iter=1000)

                    out = clf.predict_proba(query_features)

                    acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
                    max_value, index = torch.max(out, -1)

                    if test_only:
                        return acc_train, max_value, index
                    else:
                        return acc_train
    
                else:
    
                    acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
                    with torch.no_grad():
                        max_value, index=torch.max(out,-1)
    
    
    
                    del net_new, X_out_sup, X_out_query, out,net_para, param_require_grad, grad, X_total_query, X_total_sup
                    if test_only:
                        return acc_train, max_value, index
                    else:
                        return acc_train
        
        if not test_only:
            best_acc = 0
            accs_train = []
            for epoch in range(args.num_train_tasks):
                accs_train.append(train_epoch(epoch))
                if np.random.rand() < 0.05:
                    logger.info('Meta-train_Accuracy: {:.4f}'.format(np.mean(accs_train)))
                    print('Meta-train_Accuracy: {:.4f}'.format(np.mean(accs_train)))

            accs = []
            for epoch_test in range(args.num_test_tasks):
                accs.append(train_epoch(epoch_test, mode='test'))
            result = np.mean(accs)
        else:
            accs = []
            max_values = []
            indices = []
            accs_train = []

            #########################################
            #train before test
            #for epoch in range(args.num_train_tasks//5):
            #    accs_train.append(train_epoch(epoch))
            #########################################

            for epoch_test in range(args.num_test_tasks * args.num_true_test_ratio):
                acc, max_value, index = train_epoch(epoch_test, mode='test')
                accs.append(acc)
                max_values.append(max_value)
                indices.append(index)
                del acc, max_value, index
            result = (np.mean(accs), torch.cat(max_values, 0), torch.cat(indices, 0))

        if args.dp_mode == 'local' and args.grad_norms_ma:
            args.dp_clip = float(np.percentile(list(args.grad_norms_ma.values()), 90))
        if np.random.rand() < 0.3:
            print('Meta-test_Accuracy: {:.4f}'.format(np.mean(accs)))
        #logger.info("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))


    finally:
        if args.dp_mode == 'local':
            if hasattr(privacy_engine, 'detach'):
                gmodel, dp_optimizer, _ = privacy_engine.detach()
                privacy_engine = None
                gmodel.train()
            else:
                dp_optimizer = orig_optimizer
                gmodel = base_model
                gmodel.train()
            gmodel = remove_dp_hooks(gmodel)
        base_model.train()
    return result, epsilon
def local_train_net_few_shot(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, device='cpu', test_only=False, test_only_k=0):
    avg_acc = 0.0
    acc_list = []
    max_value_all_clients = []
    indices_all_clients = []
    epsilon = None
    deltas = {}

    for net_id, net in nets.items():
        print(net_id)
        nets[net_id] = net

        dataidxs = net_dataidx_map[net_id]

        n_epoch = args.epochs

        X_train_client = X_train[dataidxs]
        y_train_client = y_train[dataidxs]

        if test_only is False:
            prev_params = copy.deepcopy(net.state_dict())
            net.train()
            result, epsilon = train_net_few_shot_new(
                net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client, y_train_client, X_test, y_test,
                device=device, test_only=False
            )
            testacc = result
            if args.dp_mode == 'server':
                new_params = net.state_dict()
                deltas[net_id] = {k: new_params[k] - prev_params[k] for k in new_params}
        else:
            net.train()
            result, _ = train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client, y_train_client, X_test, y_test,
                                        device=device, test_only=True, test_only_k=test_only_k)
            testacc, max_values, indices = result
            max_value_all_clients.append(max_values)
            indices_all_clients.append(indices)

            acc_list.append(testacc)

            logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
            print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

            max_value_all_clients = torch.stack(max_value_all_clients, 0)
            indices_all_clients = torch.stack(indices_all_clients, 0)
            return acc_list, max_value_all_clients, indices_all_clients, epsilon

        avg_acc += testacc
        acc_list.append(testacc)

    logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
    print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

    if test_only:
        max_value_all_clients = torch.stack(max_value_all_clients, 0)
        indices_all_clients = torch.stack(indices_all_clients, 0)
        return acc_list, max_value_all_clients, indices_all_clients, epsilon

    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info('avg test acc %f' % avg_acc)
        logger.info('std acc %f' % np.std(acc_list))

    if args.dp_mode == 'server':
        return deltas, epsilon
    return nets, epsilon


def aggregate_deltas(global_w, deltas, args):
    """Aggregate client deltas with clipping and noise."""
    clipped = []
    for delta in deltas.values():
        flat = torch.cat([v.view(-1) for v in delta.values()])
        norm = torch.norm(flat)
        scale = min(1.0, args.dp_clip / (norm + 1e-12))
        clipped.append({k: v * scale for k, v in delta.items()})
    for key in global_w:
        if 'transform_layer' in key:
            continue
        stacked = torch.stack([d[key] for d in clipped])
        avg_update = stacked.mean(dim=0)
        noise = torch.randn_like(avg_update) * args.dp_noise * args.dp_clip
        global_w[key] += avg_update + noise


if __name__ == '__main__':
    args = get_args()
    print(args)

    # Debug
    import torch, time

    print(torch.cuda.is_available())  # True
    print(torch.cuda.current_device())  # 0
    print(torch.cuda.get_device_name(0))  # GeForce RTX 3070
    start = time.time()
    dummy = torch.rand(4096, 4096, device='cuda') @ torch.rand(4096, 4096, device='cuda')
    torch.cuda.synchronize()
    print("GEMM time:", time.time() - start, "s")  # should be < 0.2 s

    if torch.cuda.is_available():
        print(f"Running on GPU {torch.cuda.current_device()}:",
              torch.cuda.get_device_name(0))
    # Debug End

    if args.dataset=='FC100':
        fine_split_train_map={class_:i for i,class_ in enumerate(fine_split['train'])}
    elif args.dataset=='20newsgroup':
        fine_split_train_map={class_:i for i,class_ in enumerate([1, 5, 10, 11, 13, 14, 16, 18])}
    elif args.dataset=='fewrel':
        fine_split_train_map = {class_: i for i, class_ in enumerate([0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                                 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                 39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                                 59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                                 76, 77, 78])}
    elif args.dataset=='huffpost':
        fine_split_train_map = {class_: i for i, class_ in enumerate(list(range(20)))}
    
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    test_task_sample_seed=1
    np.random.seed(test_task_sample_seed)
    test_classes=[]
    test_index=[]
    for i in range(args.num_test_tasks):
        test_classes.append(np.random.choice(fine_split['test'], args.N, replace=False).tolist())
        test_index.append(np.random.rand(args.N, args.K+args.Q))



    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    if args.dataset=='20newsgroup':
        seed=13
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    print(X_train.shape)
    print(X_test.shape)
    N=args.N
    K=args.K
    Q=args.Q

    # Privacy accounting is handled by Opacus' PrivacyEngine.

    support_labels=torch.zeros(N*K,dtype=torch.long)
    for i in range(N):
        support_labels[i * K:(i + 1) * K] = i
    query_labels=torch.zeros(N*Q,dtype=torch.long)
    for i in range(N):
        query_labels[i * Q:(i + 1) * Q] = i
    if args.device!='cpu':
        support_labels=support_labels.cuda()
        query_labels=query_labels.cuda()
    
    
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))


    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device='gpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device='gpu')
    global_model = global_models[0]
    n_comm_rounds = args.comm_round
    if args.load_model_file and args.alg != 'plot_visual':
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0
    if args.alg == 'fedavg':
        use_minus=False
        best_acc=0
        best_acc_5=0
        best_confident_acc=0

        dp_steps = 0
        for round in range(n_comm_rounds):
            #logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}
            participating_ids = list(nets_this_round.keys())

            total_data_points = sum(len(net_dataidx_map[r]) for r in participating_ids)

            for client_id in participating_ids:
                net = nets_this_round[client_id]
                if use_minus:
                    net_para = net.state_dict()
                    for key in net_para:
                        net_para[key] = (global_w[key] * total_data_points - net_para[key] * len(net_dataidx_map[client_id])) / (total_data_points + 1e-9 - len(net_dataidx_map[client_id]))
                    net.load_state_dict(net_para)
                else:
                    net_para = net.state_dict()
                    for key in net_para:
                        if key != 'few_classify.weight' and key != 'few_classify.bias' and 'transformer' not in key and 'transform_layer' not in key:
                            net_para[key] = global_w[key]
                    net.load_state_dict(net_para)

            for k in [1,5]:
                global_acc, max_value_all_clients, indices_all_clients, _ = local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device, test_only=True, test_only_k=k)
                global_acc = max(global_acc)
                if k==1:
                    if global_acc > best_acc:
                        best_acc = global_acc
                    print('>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc))
                    logger.info(
                        '>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc))
                elif k==5:
                    if global_acc > best_acc_5:
                        best_acc_5 = global_acc
                    print('>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc_5))
                    logger.info(
                        '>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc_5))

            if args.dp_mode == 'server':
                deltas, _ = local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)
            else:
                local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)

            if args.dp_mode == 'local':
                dp_steps += args.num_train_tasks * len(participating_ids)
            elif args.dp_mode == 'server':
                dp_steps += 1
            if args.dp_mode != 'off':
                epsilon = dp_utils.compute_epsilon(
                    dp_steps,
                    args.dp_noise,
                    args.dp_delta,
                    accountant=args.dp_accountant,
                    sampling_rate=len(participating_ids) / args.n_parties,
                )

            if args.dp_mode == 'server':
                aggregate_deltas(global_w, deltas, args)
            else:
                total_data_points = sum(len(net_dataidx_map[r]) for r in participating_ids)
                fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in participating_ids]

                for net_id, client_id in enumerate(participating_ids):
                    net = nets_this_round[client_id]
                    net_para = net.state_dict()
                    if net_id == 0:
                        for key in net_para:
                            if 'transform_layer' in key:
                                continue
                            global_w[key] = net_para[key] * fed_avg_freqs[net_id]
                    else:
                        for key in net_para:
                            if 'transform_layer' in key:
                                continue
                            global_w[key] += net_para[key] * fed_avg_freqs[net_id]

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)


            #global_model.cuda()

            print('>> Current Round: {}'.format(round))
            logger.info('>> Current Round: {}'.format(round))
            if args.dp_mode != 'off' and args.print_eps:
                print('Current epsilon {:.4f}, delta {:.1e}'.format(epsilon, args.dp_delta))
                logger.info('Current epsilon {:.4f}, delta {:.1e}'.format(epsilon, args.dp_delta))

            mkdirs(args.modeldir+'fedavg/')

            if global_acc > best_acc:
                torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')
