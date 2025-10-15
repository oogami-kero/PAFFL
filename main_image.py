import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import logging
import os
import copy
import datetime
import random
import time
import math
from collections import OrderedDict
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from PIL import Image

from model import *
from utils import *
from dp_utils import RDPAccountant
import warnings

warnings.filterwarnings('ignore')


_HEAD_PARAM_PREFIXES = ('l1', 'l2', 'all_classify')
_IMAGE_DATASETS = {'FC100', 'miniImageNet'}


def _get_head_param_names(model):
    head_names = []
    for name, _ in model.named_parameters():
        if name.startswith(_HEAD_PARAM_PREFIXES):
            head_names.append(name)
    return head_names


def _parse_attack_rounds(rounds_arg, total_rounds):
    if rounds_arg is None:
        return {total_rounds - 1}
    rounds_arg = rounds_arg.strip()
    if rounds_arg.lower() == 'all':
        return set(range(total_rounds))
    result = set()
    for part in rounds_arg.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return {r for r in result if 0 <= r < total_rounds}


def _serialize_args(args):
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            serialized[key] = value
        elif isinstance(value, (list, tuple)):
            serialized[key] = list(value)
        else:
            serialized[key] = str(value)
    return serialized


def _apply_test_transform_image(dataset, array):
    if dataset == 'FC100':
        transform = transform_test(normalize_fc100)
    else:
        transform = transform_test(normalize_mini)
    return transform(array)


def _normalize_class_counts(counts):
    normalized = {}
    for client, cls_counts in counts.items():
        normalized[str(client)] = {int(cls): int(cnt) for cls, cnt in cls_counts.items()}
    return normalized


def _prepare_attack_probes_image(X_train, y_train, X_test, y_test, train_indices, test_indices, dataset, device):
    def _build_inputs(source_x, indices):
        tensors = []
        for idx in indices:
            tensors.append(_apply_test_transform_image(dataset, source_x[idx]))
        if tensors:
            stacked = torch.stack(tensors, 0).to(device)
        else:
            stacked = None
        return stacked

    probe_test_inputs = _build_inputs(X_test, test_indices)
    probe_train_inputs = _build_inputs(X_train, train_indices)
    probe_test_labels = torch.tensor(y_test[test_indices], device=device, dtype=torch.long) if len(test_indices) else None
    probe_train_labels = torch.tensor(y_train[train_indices], device=device, dtype=torch.long) if len(train_indices) else None
    return probe_train_inputs, probe_train_labels, probe_test_inputs, probe_test_labels


def _init_attack_context_image(args, device, global_model, X_train, y_train, X_test, y_test, client_class_counts):
    if not getattr(args, 'attack_dump', 0):
        return None

    attack_dir_root = Path(args.attack_dir)
    name_component = args.log_file_name if args.log_file_name else datetime.datetime.now().strftime('attack_%Y%m%d_%H%M%S')
    attack_dir = attack_dir_root / name_component
    attack_dir.mkdir(parents=True, exist_ok=True)

    attack_rounds = _parse_attack_rounds(args.attack_dump_rounds, args.comm_round)
    head_param_names = _get_head_param_names(global_model)
    # Build a mapping from raw class ids to head row indices when needed
    class_to_head_index = None
    if args.dataset == 'FC100':
        try:
            class_order = fine_split['train']
            class_to_head_index = {int(cls): int(i) for i, cls in enumerate(class_order)}
        except Exception:
            class_to_head_index = None
    elif args.dataset == 'miniImageNet':
        class_to_head_index = {int(i): int(i) for i in range(64)}
    rng = np.random.default_rng(args.init_seed)

    probe_size = max(0, min(args.attack_probe_size, len(y_test)))
    test_indices = rng.choice(len(y_test), size=probe_size, replace=False) if probe_size > 0 else np.array([], dtype=int)
    train_indices = rng.choice(len(y_train), size=probe_size, replace=False) if probe_size > 0 else np.array([], dtype=int)

    probe_train_inputs, probe_train_labels, probe_test_inputs, probe_test_labels = _prepare_attack_probes_image(
        X_train, y_train, X_test, y_test, train_indices, test_indices, args.dataset, device)

    norm_counts = _normalize_class_counts(client_class_counts)

    metadata = {
        'dataset': args.dataset,
        'use_transform_layer': bool(getattr(args, 'use_transform_layer', 0)),
        'attack_rounds': sorted(attack_rounds),
        'head_param_names': head_param_names,
        'probe_train_indices': train_indices.tolist(),
        'probe_test_indices': test_indices.tolist(),
        'client_class_counts': norm_counts,
        'class_to_head_index': class_to_head_index,
        'args_summary': _serialize_args(args),
    }
    (attack_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    return {
        'dir': attack_dir,
        'rounds': attack_rounds,
        'head_param_names': head_param_names,
        'dump_clients': bool(args.attack_dump_clients),
        'probe_train_inputs': probe_train_inputs,
        'probe_train_labels': probe_train_labels,
        'probe_test_inputs': probe_test_inputs,
        'probe_test_labels': probe_test_labels,
        'probe_train_indices': train_indices.tolist(),
        'probe_test_indices': test_indices.tolist(),
        'device': device,
    }


def _dump_attack_artifacts_image(ctx, round_idx, global_model, global_state, updated_state, nets_this_round):
    attack_dir = ctx['dir']
    suffix = f'round_{round_idx:04d}'
    head_names = ctx['head_param_names']

    head_state = {name: updated_state[name].detach().cpu().clone() for name in head_names}
    head_update = {name: (updated_state[name] - global_state[name]).detach().cpu().clone() for name in head_names}
    torch.save(head_state, attack_dir / f'{suffix}_head_state.pt')
    torch.save(head_update, attack_dir / f'{suffix}_head_update.pt')

    if ctx['dump_clients']:
        client_updates = {}
        for client_id, net in nets_this_round.items():
            net_state = net.state_dict()
            client_updates[str(client_id)] = {name: (net_state[name].detach().cpu().clone() - global_state[name].detach().cpu().clone()) for name in head_names}
        torch.save(client_updates, attack_dir / f'{suffix}_client_updates.pt')

    global_model.eval()
    try:
        if ctx['probe_test_inputs'] is not None:
            with torch.no_grad():
                _, _, logits_test = global_model(ctx['probe_test_inputs'], all_classify=True)
            torch.save({'indices': ctx['probe_test_indices'],
                        'logits': logits_test.cpu(),
                        'labels': ctx['probe_test_labels'].cpu()},
                       attack_dir / f'{suffix}_probe_test_logits.pt')
        if ctx['probe_train_inputs'] is not None:
            with torch.no_grad():
                _, _, logits_train = global_model(ctx['probe_train_inputs'], all_classify=True)
            torch.save({'indices': ctx['probe_train_indices'],
                        'logits': logits_train.cpu(),
                        'labels': ctx['probe_train_labels'].cpu()},
                       attack_dir / f'{suffix}_probe_train_logits.pt')
    finally:
        global_model.train()
fine_id_coarse_id = {0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3, 10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11, 20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15, 30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5, 40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10, 50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17, 60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19, 70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13, 80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19, 90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13}

coarse_id_fine_id = {0: [4, 30, 55, 72, 95], 1: [1, 32, 67, 73, 91], 2: [54, 62, 70, 82, 92], 3: [9, 10, 16, 28, 61], 4: [0, 51, 53, 57, 83], 5: [22, 39, 40, 86, 87], 6: [5, 20, 25, 84, 94], 7: [6, 7, 14, 18, 24], 8: [3, 42, 43, 88, 97], 9: [12, 17, 37, 68, 76], 10: [23, 33, 49, 60, 71], 11: [15, 19, 21, 31, 38], 12: [34, 63, 64, 66, 75], 13: [26, 45, 77, 79, 99], 14: [2, 11, 35, 46, 98], 15: [27, 29, 44, 78, 93], 16: [36, 50, 65, 74, 80], 17: [47, 52, 56, 59, 96], 18: [8, 13, 48, 58, 90], 19: [41, 69, 81, 85, 89]}

coarse_split={'train': [1,2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19], 'valid': [8, 11, 13, 16], 'test': [0, 7, 12, 14]}

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


def _get_image_transform(dataset, train=False):
    if dataset == 'FC100':
        if train:
            # A light augmentation option is enabled by --augment_normal_train
            aug = int(globals().get('args', type('obj', (), {})).augment_normal_train) if 'args' in globals() else 0
            if aug:
                return transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_fc100
                ])
            else:
                return transform_test(normalize_fc100)
        return transform_test(normalize_fc100)
    else:
        if train:
            return transform_test(normalize_mini)
        return transform_test(normalize_mini)


def _evaluate_global_normal(model, dataset, X_test, y_test, device, fallback_train=None):
    model.eval()
    tfm = _get_image_transform(dataset, train=False)
    correct = 0
    total = 0
    batch = []
    labels = []
    # Map raw labels to train-class indices (FC100)
    label_map = None
    try:
        label_map = globals().get('fine_split_train_map', None)
        if label_map is None and dataset == 'FC100':
            label_map = {int(c): int(i) for i, c in enumerate(fine_split['train'])}
    except Exception:
        label_map = None
    with torch.no_grad():
        for i in range(len(y_test)):
            raw_y = int(y_test[i])
            if label_map is not None and raw_y not in label_map:
                continue
            mapped_y = label_map[raw_y] if label_map is not None else raw_y
            batch.append(tfm(X_test[i]))
            labels.append(mapped_y)
            if len(batch) >= 64 or i == len(y_test) - 1:
                xb = torch.stack(batch, 0).to(device)
                yb = torch.tensor(labels, dtype=torch.long, device=device)
                _, _, out_all = model(xb, all_classify=True)
                pred = out_all.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0)
                batch.clear(); labels.clear()
    if total > 0:
        return correct / max(1, total)
    # Fallback: evaluate on provided train subset if no test samples match label map
    if fallback_train is not None:
        Xtr, ytr = fallback_train
        correct = 0; total = 0
        batch=[]; labels=[]
        for i in range(min(len(ytr), 2000)):
            raw_y = int(ytr[i])
            mapped_y = label_map[raw_y] if label_map is not None else raw_y
            batch.append(tfm(Xtr[i])); labels.append(mapped_y)
            if len(batch) >= 64 or i == len(ytr) - 1:
                xb = torch.stack(batch,0).to(device)
                yb = torch.tensor(labels, dtype=torch.long, device=device)
                _, _, out_all = model(xb, all_classify=True)
                pred = out_all.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += yb.size(0); batch.clear(); labels.clear()
        return correct / max(1, total)
    return 0.0


def local_train_net_normal(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, device='cpu'):
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        tfm = _get_image_transform(args.dataset, train=True)
        net.train()
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        else:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01, momentum=0.9, weight_decay=args.reg)
        loss_ce = nn.CrossEntropyLoss()
        # One light local epoch for speed
        batch = []
        labels = []
        step = 0
        # Map labels to contiguous train-class ids when needed
        try:
            label_map = globals().get('fine_split_train_map', None)
            if label_map is None and args.dataset == 'FC100':
                label_map = {int(c): int(i) for i, c in enumerate(fine_split['train'])}
        except Exception:
            label_map = None

        for i in range(len(dataidxs)):
            idx = dataidxs[i]
            batch.append(tfm(X_train[idx]))
            raw_y = int(y_train[idx])
            mapped_y = label_map[raw_y] if label_map is not None else raw_y
            labels.append(mapped_y)
            if len(batch) >= 64 or i == len(dataidxs) - 1:
                xb = torch.stack(batch, 0).to(device)
                yb = torch.tensor(labels, dtype=torch.long, device=device)
                optimizer.zero_grad()
                _, _, out_all = net(xb, all_classify=True)
                loss = loss_ce(out_all, yb)
                loss.backward()
                optimizer.step()
                batch.clear(); labels.clear()
                step += 1
                if step >= int(getattr(args, 'normal_local_steps', 10)):
                    break
    # Return simple global accuracy for logging if desired
    return _evaluate_global_normal(list(nets.values())[0], args.dataset, X_test, y_test, device)


def InforNCE_Loss(anchor, sample, tau, all_negative=False, temperature_matrix=None):
    def _similarity(h1: torch.Tensor, h2: torch.Tensor):
        h1 = F.normalize(h1)
        h2 = F.normalize(h2)
        return h1 @ h2.t()

    assert anchor.shape[0] == sample.shape[0]

    pos_mask = torch.eye(anchor.shape[0], dtype=torch.float, device=anchor.device)
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
    parser.add_argument('--contras_w', type=float, default=0.02, help='Weight for InfoNCE contrastive loss (per query), set 0 to disable')
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--dp_enable', type=int, default=0, help='Enable central DP on aggregated parameters (0/1)')
    parser.add_argument('--dp_clip_norm', type=float, default=1.0, help='Clip norm for per-client updates when DP is enabled')
    parser.add_argument('--dp_noise_multiplier', type=float, default=1.0, help='Gaussian noise multiplier (sigma) for DP')
    parser.add_argument('--dp_seed', type=int, default=0, help='Seed for DP noise (set negative to use global RNG state)')
    parser.add_argument('--dp_target', type=str, default='full', choices=['full', 'head'], help='Scope of parameters receiving DP noise')
    parser.add_argument('--dp_clip_mode', type=str, default='global', choices=['global', 'tensor', 'group'], help='Clipping mode for DP updates')
    parser.add_argument('--dp_delta', type=float, default=-1.0, help='Override DP delta; negative uses default N^-1.1')
    parser.add_argument('--dp_head_scope', type=str, default='full', choices=['full', 'logits'], help='When target=head, DP on full head or logits only')
    parser.add_argument('--head_train_scope', type=str, default='all', choices=['all', 'logits'], help='Train all head layers or logits only')
    parser.add_argument('--head_grad_clip', type=float, default=1.0, help='Max-norm gradient clip for head params (<=0 disables)')
    parser.add_argument('--dp_clip_norm_head', type=float, default=None, help='Override clip norm for head group (group mode)')
    parser.add_argument('--dp_clip_norm_feat', type=float, default=None, help='Override clip norm for features group (group mode)')
    parser.add_argument('--dp_adaptive', type=int, default=1, help='Enable adaptive per-tensor/group clipping via EMA of quantiles (0/1)')
    parser.add_argument('--dp_quantile', type=float, default=0.8, help='Quantile of per-client norms to target for adaptive clipping')
    parser.add_argument('--dp_ema_beta', type=float, default=0.2, help='EMA factor for adaptive clip norms')
    parser.add_argument('--train_expand_N', type=int, default=4, help='Multiply N in train episodes (e.g., FC100 default 4)')
    parser.add_argument('--freeze_backbone_after', type=int, default=-1, help='Round index after which backbone features stop training (-1 disables)')
    parser.add_argument('--use_transform_layer', type=int, default=0, help='Enable client-side transform layer before shared head (0/1)')
    parser.add_argument('--attack_dump', type=int, default=0, help='Enable attack artifact dumping (0/1)')
    parser.add_argument('--attack_dump_rounds', type=str, default=None, help='Comma-separated rounds or "all" to dump attack artifacts')
    parser.add_argument('--attack_dump_clients', type=int, default=0, help='Dump per-client head updates for attacks (0/1)')
    parser.add_argument('--attack_probe_size', type=int, default=64, help='Number of train/test samples to log as attack probes')
    parser.add_argument('--attack_dir', type=str, default='./attack_dumps', help='Directory to store attack artifacts')
    parser.add_argument('--fewshot_train_mode', type=str, default='meta', choices=['meta','normal','none'], help='Training mode used between few-shot evaluations: meta (original), normal (standard local training), or none')
    parser.add_argument('--normal_local_steps', type=int, default=10, help='Max optimizer steps per client per round in normal local training')
    parser.add_argument('--augment_normal_train', type=int, default=0, help='Use simple data augmentation in normal local training (0/1)')
    args = parser.parse_args()
    return args


def init_nets(net_configs, n_parties, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100' or args.dataset=='FC100' :
        total_classes=60 # FC100 uses 60 train classes; CIFAR100 handled below for normal mode
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
            ebd=WORDEBD(args.finetune_ebd)
        for net_i in range(n_parties):
            if args.dataset=='FC100' or args.dataset=='miniImageNet':
                net = ModelFed_Adp(args.model, args.out_dim, n_classes, total_classes, net_configs, args)
            else:
                net = LSTMAtt(WORDEBD(args.finetune_ebd), args.out_dim, n_classes, total_classes,args)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    elif args.mode!='few-shot' and args.method=='new':
        # Normal mode: initialize same model; training/eval will use all_classify head
        for net_i in range(n_parties):
            if args.dataset=='FC100' or args.dataset=='miniImageNet':
                net = ModelFed_Adp(args.model, args.out_dim, n_classes if 'n_classes' in locals() else 5, total_classes, net_configs, args)
            else:
                # For datasets like cifar10/cifar100/tinyimagenet, ensure total_classes is defined above
                net = ModelFed_Adp(args.model, args.out_dim, n_classes if 'n_classes' in locals() else 5, total_classes if 'total_classes' in locals() else (10 if args.dataset=='cifar10' else (100 if args.dataset=='cifar100' else 200)), net_configs, args)
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


def get_dp_parameter_names(model, target='full'):
    excluded_substrings = ('few_classify', 'transformer', 'transform_layer', 'bn', 'running_mean', 'running_var', 'num_batches_tracked')
    head_includes = ('l1', 'l2', 'all_classify')
    names = []
    # Only tensors returned here participate in DP aggregation; everything else remains
    # unchanged on the server for DP-enabled rounds.
    for name, _ in model.named_parameters():
        if target == 'head':
            if not any(name.startswith(prefix) or prefix in name for prefix in head_includes):
                continue
        else:
            # Exclude BN params and special layers (align with PrivateFL/FedBN behavior)
            if any(ex in name for ex in excluded_substrings):
                continue
        names.append(name)
    return names


def train_net_few_shot_new(net_id, net, n_epoch, lr, args_optimizer, args, X_train_client,y_train_client, X_test, y_test,
                                        device='cpu', test_only=False, test_only_k=0):
    #net = nn.DataParallel(net)
    #net=nn.parallel.DistributedDataParallel(net)
    #net.cuda()

    #logger.info('Training network %s' % str(net_id))
    #logger.info('n_training: %d' % X_train_client.shape[0])
    #logger.info('n_test: %d' % X_test.shape[0])
    
    # Build optimizer param list according to head training scope
    if getattr(args, 'head_train_scope', 'all') == 'logits':
        param_list = [p for n, p in net.named_parameters() if 'all_classify' in n and p.requires_grad]
    else:
        param_list = list(filter(lambda p: p.requires_grad, net.parameters()))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(param_list, lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(param_list, lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(param_list, lr=0.05, momentum=0.9, weight_decay=args.reg)
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()

    def train_epoch(epoch, mode='train'):

        if mode == 'train':

            if args.dataset=='fewrel' :
                N=args.N*3
                K=2
                Q=2
            elif args.dataset=='huffpost':
                N = args.N
                K = 5#args.K
                Q = args.Q
            elif args.dataset=='FC100':
                N=args.N*max(1, int(getattr(args, 'train_expand_N', 4)))
                K=2
                Q=2
            elif args.dataset=='miniImageNet':
                N=args.N*max(1, int(getattr(args, 'train_expand_N', 4)))
                K=2
                Q=2
            else:
                N = args.N
                K = 5#args.K
                Q = args.Q
            net.train()
            optimizer.zero_grad()
            if args.dataset == 'FC100':
                #X_transform = transform_train(normalize=normalize_fc100, crop_size=32, padding=4)
                X_transform=    transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize_fc100
                ])
            else:
                #X_transform = transform_train(normalize=normalize_mini, crop_size=84)
                X_transform=    transforms.Compose([
                    lambda x: Image.fromarray(x),
                                #transforms.ToPILImage(),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    normalize_mini
                ])

        else:
            N=args.N
            K=args.K
            Q=args.Q
            #N=args.N*2
            net.eval()
            if args.dataset == 'FC100':
                X_transform = transform_test(normalize=normalize_fc100)
            else:
                X_transform = transform_test(normalize=normalize_mini)

        if test_only==True:
            K=test_only_k

        support_labels = torch.zeros(N * K, dtype=torch.long)
        for i in range(N):
            support_labels[i * K:(i + 1) * K] = i
        query_labels = torch.zeros(N * Q, dtype=torch.long)
        for i in range(N):
            query_labels[i * Q:(i + 1) * Q] = i
        if args.device != 'cpu':
            support_labels = support_labels.to(device)
            query_labels = query_labels.to(device)

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




                y_total = torch.cat([torch.cat(y_sup, 0), torch.cat(y_query, 0)], 0).long().to(device)
        #y_total=torch.tensor(np.concatenate([np.concatenate(y_sup, 0),np.concatenate(y_query, 0)],0)).cuda()
        
        X_total_sup=np.concatenate(X_total_sup, 0)
        X_total_query=np.concatenate(X_total_query,0)


        if args.dataset=='FC100' or args.dataset=='miniImageNet':
            X_total_transformed_sup=[]
            X_total_transformed_query=[]
            for i in range(X_total_sup.shape[0]):
                X_total_transformed_sup.append(X_transform(X_total_sup[i]))
            X_total_sup=torch.stack(X_total_transformed_sup,0).to(device)

            for i in range(X_total_query.shape[0]):
                X_total_transformed_query.append(X_transform(X_total_query[i]))
            X_total_query=torch.stack(X_total_transformed_query,0).to(device)
        else:
            X_total_sup=torch.tensor(X_total_sup).to(device)
            X_total_query=torch.tensor(X_total_query).to(device)




        #net.load_state_dict(net_para_ori)
        #_,_,out_all=net_new(torch.cat([X_total_sup, X_total_query],0), all_classify=True)

            #print(out[:3])
        if mode == 'train':
            loss_all=0
            # all_classify update
            X_out_all, x_all, out_all = net(torch.cat([X_total_sup, X_total_query], 0), all_classify=True)
            out_sup=X_out_all[:N*K].reshape([N,K,-1]).transpose(0,1)
            out_query=X_out_all[N*K:].reshape([N,Q,-1]).transpose(0,1)



            # _, _, out_all = net(X_total_sup, all_classify=True)



            if args.fine_tune_steps>0:
                net_new = copy.deepcopy(net)

                for j in range(args.fine_tune_steps):
                    X_out_sup, X_transformer_out_sup, out = net_new(X_total_sup)
                    loss = loss_ce(out, support_labels)
                    #loss+=loss_ce(out, out_sup_on_N_class)
                    #loss+=loss_mse(out_sup_on_N_class.softmax(-1),out.softmax(-1))

                    net_para = net_new.state_dict()
                    param_require_grad = {}
                    for key, param in net_new.named_parameters():
                        if key == 'few_classify.weight' or key == 'few_classify.bias':
                            # if key !='all_classify.weight' and key !='all_classify.bias':
                            if param.requires_grad:
                                param_require_grad[key] = param
                    grad = torch.autograd.grad(loss, param_require_grad.values(), allow_unused=True)
                    for key, grad_ in zip(param_require_grad.keys(), grad):
                        if grad_ == None: continue
                        net_para[key] = net_para[key] - args.fine_tune_lr * grad_
                    # net_para = list(
                    #                map(lambda p: p[1] - fine_tune_lr * p[0], zip(grad, net_para)))
                    # net_para={key:value for key, value in zip(net.state_dict().keys(),net.state_dict().values())}
                    net_new.load_state_dict(net_para)

                X_out_query, _, out = net_new(X_total_query)
                X_out_sup, X_transformer_out_sup, _ = net_new(X_total_sup)

                X_transformer_out_sup = X_transformer_out_sup.reshape([N, K, -1]).transpose(0, 1)
                #############################
                # Q=K here update for all-model
                cw = float(getattr(args, 'contras_w', 0.02))
                if cw > 0.0:
                    for j in range(Q):
                        contras_loss, similarity = InforNCE_Loss(
                            X_transformer_out_sup[j], out_sup[(j + 1) % Q], tau=0.5)
                        loss_all += contras_loss / max(1, Q) * cw
                loss_all += loss_ce(out_all, y_total)
                loss_all.backward()
                # Clip gradients on head params to stabilize and reduce update norm
                if getattr(args, 'head_grad_clip', 1.0) and args.head_grad_clip > 0:
                    if getattr(args, 'head_train_scope', 'all') == 'logits':
                        clip_params = [p for n, p in net.named_parameters() if 'all_classify' in n and p.grad is not None]
                    else:
                        clip_params = [p for p in net.parameters() if p.grad is not None]
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=args.head_grad_clip)
                optimizer.step()
                ############################

                X_out_all, x_all, out_all = net(torch.cat([X_total_sup, X_total_query], 0), all_classify=True)
                ###################################
                # few_classify update
                net_para_ori=net.state_dict()

                param_require_grad={}
                for key, param in net_new.named_parameters():
                    if key=='few_classify.weight' or key=='few_classify.bias' or 'transformer' in key:
                    #if key != 'module.all_classify.weight' and key != 'module.all_classify.bias':
                        param_require_grad[key]=param

                #meta-update few-classifier on query
                loss = loss_ce(out, query_labels)
                # Compare query logits to the subset distribution over the N classes.
                # The previous cross-entropy with a multi-target tensor was invalid and caused
                # runtime errors under certain shapes; we omit this auxiliary term to match
                # earlier stable runs.
                out_sup_on_N_class = out_all[N * K:, transformed_class_list]
                out_sup_on_N_class = out_sup_on_N_class / (out_sup_on_N_class.sum(-1, keepdim=True) + 1e-12)
                # Auxiliary alignment loss disabled (kept for reference):
                # loss += loss_mse(out.softmax(-1), _expand_to_full_classes(out_sup_on_N_class)) * 0.1
                grad = torch.autograd.grad(loss, param_require_grad.values())
                for key, grad_ in zip(param_require_grad.keys(), grad):
                    net_para_ori[key]=net_para_ori[key]-args.meta_lr*grad_
                net.load_state_dict(net_para_ori)
                ##################################
                del net_new,X_out_query, out

            if np.random.rand() < 0.005:
                print('loss: {:.4f}'.format(loss_all.item()))


            acc_train = (torch.argmax(out_all, -1) == y_total).float().mean().item()

            del X_out_all,  out_all
            return acc_train

        else:
            use_logistic=True

            if use_logistic:
                with torch.no_grad():
                    X_out_all, x_all, out_all = net(torch.cat([X_total_sup, X_total_query], 0))
                    X_out_sup=X_out_all[:N*K]
                    X_out_query=X_out_all[N*K:]

                    support_features = l2_normalize(X_out_sup.detach().cpu()).numpy()
                    query_features = l2_normalize(X_out_query.detach().cpu()).numpy()

                    clf = LogisticRegression(penalty='l2',
                                             random_state=0,
                                             C=1.0,
                                             solver='lbfgs',
                                             max_iter=1000,
                                             multi_class='multinomial')
                    clf.fit(support_features, support_labels.detach().cpu().numpy())

                    query_ys_pred = clf.predict(query_features)

                    out=torch.tensor(clf.predict_proba(query_features)).to(device)

                    acc_train = (torch.argmax(out, -1) == query_labels).float().mean().item()
                    max_value, index=torch.max(out,-1)

                    #del net_new, X_out_sup, X_out_query, out, param_require_grad, grad
                    if test_only:
                        return acc_train, max_value, index
                    else:
                        return acc_train

                #return metrics.accuracy_score(query_labels.detach().cpu().numpy(), query_ys_pred)

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
        accs_train=[]
        for epoch in range(args.num_train_tasks):
            accs_train.append(train_epoch(epoch))
            if np.random.rand() < 0.05:
                logger.info("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train)))
                print("Meta-train_Accuracy: {:.4f}".format(np.mean(accs_train)))


        accs=[]
        for epoch_test in range(args.num_test_tasks):
            accs.append(train_epoch(epoch_test, mode='test'))
    else:
        accs=[]
        max_values=[]
        indices=[]
        accs_train=[]

        #########################################
        #train before test
        #for epoch in range(args.num_train_tasks//5):
        #    accs_train.append(train_epoch(epoch))
        #########################################

        for epoch_test in range(args.num_test_tasks*args.num_true_test_ratio):
            acc, max_value, index=train_epoch(epoch_test, mode='test')
            accs.append(acc)
            max_values.append(max_value)
            indices.append(index)
            del acc, max_value, index

        return np.mean(accs), torch.cat(max_values,0), torch.cat(indices,0)

    if np.random.rand()<0.3:
        print("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))
    #logger.info("Meta-test_Accuracy: {:.4f}".format(np.mean(accs)))

    return  np.mean(accs)


def local_train_net_few_shot(nets, args, net_dataidx_map, X_train, y_train, X_test, y_test, device="cpu", test_only=False, test_only_k=0):
    avg_acc = 0.0
    acc_list = []
    max_value_all_clients=[]
    indices_all_clients=[]

    for net_id, net in nets.items():
        print(net_id)

        #net.cuda()

        dataidxs = net_dataidx_map[net_id]

        #logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        
    
        n_epoch = args.epochs
        
        #_,_, train_ds, test_ds = get_dataloader(args.dataset, args.datadir, args.batch_size, len(dataidxs), dataidxs)
        
        #X_train_client=train_ds.data
        #y_train_client=train_ds.target
        
        X_train_client=X_train[dataidxs]
        y_train_client=y_train[dataidxs]
        
        #X_test=test_ds.data
        #y_test=test_ds.target


        if test_only==False:
            testacc = train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,y_train_client,X_test, y_test,
                                        device=device, test_only=False)
        else:
            #np.random.seed(1)
            testacc, max_values, indices=train_net_few_shot_new(net_id, net, n_epoch, args.lr, args.optimizer, args, X_train_client,y_train_client,X_test, y_test,
                                        device=device, test_only=True, test_only_k=test_only_k)
            max_value_all_clients.append(max_values)
            indices_all_clients.append(indices)
            #np.random.seed(int(time.time()))

            acc_list.append(testacc)

            logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
            print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

            max_value_all_clients = torch.stack(max_value_all_clients, 0)
            indices_all_clients = torch.stack(indices_all_clients, 0)
            return acc_list, max_value_all_clients, indices_all_clients

        #logger.info("net {} final test acc {:.4f}" .format(net_id, testacc))

        avg_acc += testacc
        acc_list.append(testacc)



        #net.cpu()

    logger.info(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))
    print(' | '.join(['{:.4f}'.format(acc) for acc in acc_list]))

    if test_only:
        max_value_all_clients=torch.stack(max_value_all_clients,0)
        indices_all_clients=torch.stack(indices_all_clients,0)
        return acc_list, max_value_all_clients, indices_all_clients

    avg_acc /= args.n_parties
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
        logger.info("std acc %f" % np.std(acc_list))

    return nets


if __name__ == '__main__':
    args = get_args()
    print(args)
    
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

    support_labels=torch.zeros(N*K,dtype=torch.long)
    for i in range(N):
        support_labels[i * K:(i + 1) * K] = i
    query_labels=torch.zeros(N*Q,dtype=torch.long)
    for i in range(N):
        query_labels[i * Q:(i + 1) * Q] = i
    if args.device!='cpu':
        support_labels=support_labels.to(device)
        query_labels=query_labels.to(device)
    
    
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
    # Honor CPU selection; otherwise use GPU path
    _dev_flag = 'cpu' if str(args.device).lower().startswith('cpu') else 'gpu'
    nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.n_parties, args, device=_dev_flag)

    global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 1, args, device=_dev_flag)
    global_model = global_models[0]
    attack_ctx = _init_attack_context_image(args, device, global_model, X_train, y_train, X_test, y_test, traindata_cls_counts)
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

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_parties)])
        N = max(1, args.n_parties)
        dp_delta = args.dp_delta if getattr(args, 'dp_delta', -1.0) and args.dp_delta > 0 else N ** (-1.1)
        dp_orders = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 32.0, 64.0]

        if args.dp_enable:
            dp_param_names = get_dp_parameter_names(global_model, target=args.dp_target)
            # Restrict head DP scope to logits only when requested
            if args.dp_target == 'head' and getattr(args, 'dp_head_scope', 'full') == 'logits':
                dp_param_names = [n for n in dp_param_names if n.startswith('all_classify') or 'all_classify' in n]
            accountant = RDPAccountant(dp_orders)
            logger.info(
                f"DP enabled on {len(dp_param_names)} parameter tensors (target={args.dp_target}, "
                f"clip_norm={args.dp_clip_norm:.6f}, noise_multiplier={args.dp_noise_multiplier:.6f}, delta={dp_delta:.2e})")
            dp_seed_base = args.dp_seed if args.dp_seed >= 0 else None
        else:
            dp_param_names = []
            accountant = None
            dp_seed_base = None

        backbone_frozen = False
        # Adaptive DP clip state per tensor/group
        dp_clip_state = {}
        if args.dp_enable:
            for k in dp_param_names:
                dp_clip_state[k] = {
                    'C': float(args.dp_clip_norm),
                }
            # Group-wise adaptive states
            dp_clip_state['__HEAD__'] = {'C': float(args.dp_clip_norm)}
            dp_clip_state['__FEAT__'] = {'C': max(float(args.dp_clip_norm) / 5.0, 0.5)}

        for round in range(n_comm_rounds):
            #logger.info("in comm round:" + str(round))
            party_list_this_round = party_list_rounds[round]

            global_w = global_model.state_dict()
            if args.server_momentum:
                old_w = copy.deepcopy(global_model.state_dict())

            nets_this_round = {k: nets[k] for k in party_list_this_round}

            if (not backbone_frozen) and args.freeze_backbone_after >= 0 and round >= args.freeze_backbone_after:
                logger.info('Freezing backbone parameters from round {} onwards'.format(round))
                backbone_frozen = True
                for net in nets.values():
                    for name, param in net.named_parameters():
                        if name.startswith('features') and param.requires_grad:
                            param.requires_grad = False
                for name, param in global_model.named_parameters():
                    if name.startswith('features') and param.requires_grad:
                        param.requires_grad = False

            for net_id, net in nets_this_round.items():
                if use_minus:
                    net_para = net.state_dict()
                    for key in net_para:
                        net_para[key]=(global_w[key]*total_data_points-net_para[key]*len(net_dataidx_map[net_id]))/(total_data_points+1e-9-len(net_dataidx_map[net_id]))
                    net.load_state_dict(net_para)
                else:
                    # Broadcast excluding BN (parameters and buffers), few-shot head, and transformer layers
                    net_para = net.state_dict()
                    for key in net_para:
                        if (
                            key!='few_classify.weight' and key!='few_classify.bias' and
                            'transformer' not in key and 'transform_layer' not in key and
                            'bn' not in key and 'running_mean' not in key and 'running_var' not in key and 'num_batches_tracked' not in key
                        ):
                            net_para[key]=global_w[key]
                    net.load_state_dict(net_para)

            if args.mode == 'few-shot':
                for k in [1,5]:
                    global_acc, max_value_all_clients, indices_all_clients=local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device, test_only=True, test_only_k=k)
                    global_acc = max(global_acc)
                    if k==1:
                        if global_acc > best_acc:
                            best_acc = global_acc
                        print('>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc))
                        logger.info('>> Global 1 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc))
                    elif k==5:
                        if global_acc > best_acc_5:
                            best_acc_5 = global_acc
                        print('>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f}'.format(global_acc, best_acc_5))
                        logger.info('>> Global 5 Model Test accuracy: {:.4f} Best Acc: {:.4f} '.format(global_acc, best_acc_5))
            else:
                acc_normal = _evaluate_global_normal(global_model, args.dataset, X_test, y_test, device, fallback_train=(X_train, y_train))
                if acc_normal > best_acc:
                    best_acc = acc_normal
                print('>> Global Model Test accuracy (normal): {:.4f} Best Acc: {:.4f}'.format(acc_normal, best_acc))
                logger.info('>> Global Model Test accuracy (normal): {:.4f} Best Acc: {:.4f}'.format(acc_normal, best_acc))


            if args.mode == 'few-shot':
                # Select few-shot training behavior between rounds
                if getattr(args, 'fewshot_train_mode', 'meta') == 'meta':
                    local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)
                elif args.fewshot_train_mode == 'normal':
                    # Train using standard local supervised step (no meta), while evaluating few-shot with LR
                    local_train_net_normal(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)
                elif args.fewshot_train_mode == 'none':
                    # No training / pure evaluation
                    pass
                else:
                    local_train_net_few_shot(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)
            else:
                local_train_net_normal(nets_this_round, args, net_dataidx_map, X_train, y_train, X_test, y_test, device=device)

            global_state = copy.deepcopy(global_model.state_dict())
            global_w = copy.deepcopy(global_state)

            m = len(nets_this_round)
            if m == 0:
                logger.warning('No clients selected in round %d; skipping aggregation', round)
                continue

            equal_weight = 1.0 / m

            if not args.dp_enable or not dp_param_names:
                first_client = True
                for net in nets_this_round.values():
                    net_state = net.state_dict()
                    if first_client:
                        for key in net_state:
                            global_w[key] = net_state[key] * equal_weight
                        first_client = False
                    else:
                        for key in net_state:
                            global_w[key] += net_state[key] * equal_weight
                dp_noise_std = 0.0
                epsilon_round = float('inf')
                epsilon_total = float('inf')
                alpha_round = None
                alpha_total = None
                snr = float('inf')
                clip_rate = 0.0
                q = min(1.0, m / N)
            else:
                clip_norm = args.dp_clip_norm
                sigma = args.dp_noise_multiplier
                dp_updates = OrderedDict((key, torch.zeros_like(global_state[key])) for key in dp_param_names)
                clip_events = 0

                clip_mode = getattr(args, 'dp_clip_mode', 'global')
                # Group mapping for group mode
                def is_head(name: str) -> bool:
                    return name.startswith('l1') or name.startswith('l2') or name.startswith('all_classify')

                # Initialize group Cs (may be adapted below)
                C_head = args.dp_clip_norm_head if args.dp_clip_norm_head is not None else dp_clip_state.get('__HEAD__',{}).get('C', clip_norm)
                C_feat = args.dp_clip_norm_feat if args.dp_clip_norm_feat is not None else dp_clip_state.get('__FEAT__',{}).get('C', max(clip_norm/5.0, 0.5))

                # For logging: a representative noise_std value
                if clip_mode == 'global' or clip_mode == 'tensor':
                    dp_noise_std = sigma * clip_norm / m if m > 0 else 0.0
                else:  # group
                    # Log the head group's effective noise std as representative
                    dp_noise_std = sigma * C_head / m if m > 0 else 0.0

                # Collect per-client per-tensor norms for adaptive clipping
                adaptive = getattr(args, 'dp_adaptive', 1) == 1
                per_key_norms = {k: [] for k in dp_param_names}

                for client_id, net in nets_this_round.items():
                    net_state = net.state_dict()
                    client_update = {key: net_state[key] - global_state[key] for key in dp_param_names}

                    if clip_mode == 'global':
                        flat = torch.cat([t.reshape(-1) for t in client_update.values()]) if client_update else torch.zeros(1)
                        norm = float(torch.norm(flat, p=2).item())
                        if adaptive:
                            # Use current clip_norm; collect per-key norms for later
                            for k, t in client_update.items():
                                per_key_norms[k].append(float(torch.norm(t.reshape(-1), p=2).item()))
                        coef = 1.0 if norm == 0.0 else min(1.0, clip_norm / (norm + 1e-12))
                        if coef < 0.999999:
                            clip_events += 1
                        for k, t in client_update.items():
                            dp_updates[k] += t * coef * equal_weight

                    elif clip_mode == 'tensor':
                        clipped_any = False
                        for k, t in client_update.items():
                            nrm = float(torch.norm(t.reshape(-1), p=2).item())
                            if adaptive:
                                per_key_norms[k].append(nrm)
                            # Use per-tensor C if adaptive; else global clip_norm
                            Ck = dp_clip_state.get(k, {}).get('C', clip_norm) if adaptive else clip_norm
                            coef = 1.0 if nrm == 0.0 else min(1.0, Ck / (nrm + 1e-12))
                            if coef < 0.999999:
                                clipped_any = True
                            dp_updates[k] += t * coef * equal_weight
                        if clipped_any:
                            clip_events += 1

                    elif clip_mode == 'group':
                        # Compute per-group coef and apply to tensors by group
                        # Head group
                        head_vec = torch.cat([client_update[k].reshape(-1) for k in client_update if is_head(k)]) if any(is_head(k) for k in client_update) else None
                        feat_vec = torch.cat([client_update[k].reshape(-1) for k in client_update if not is_head(k)]) if any((not is_head(k)) for k in client_update) else None
                        head_coef = 1.0
                        feat_coef = 1.0
                        clipped_any = False
                        if head_vec is not None:
                            hn = float(torch.norm(head_vec, p=2).item())
                            head_C = C_head
                            if adaptive:
                                for k in client_update:
                                    if is_head(k):
                                        per_key_norms[k].append(float(torch.norm(client_update[k].reshape(-1), p=2).item()))
                                head_C = dp_clip_state.get('__HEAD__',{}).get('C', C_head)
                            head_coef = 1.0 if hn == 0.0 else min(1.0, head_C / (hn + 1e-12))
                            clipped_any = clipped_any or (head_coef < 0.999999)
                        if feat_vec is not None:
                            fn = float(torch.norm(feat_vec, p=2).item())
                            feat_C = C_feat
                            if adaptive:
                                for k in client_update:
                                    if not is_head(k):
                                        per_key_norms[k].append(float(torch.norm(client_update[k].reshape(-1), p=2).item()))
                                feat_C = dp_clip_state.get('__FEAT__',{}).get('C', C_feat)
                            feat_coef = 1.0 if fn == 0.0 else min(1.0, feat_C / (fn + 1e-12))
                            clipped_any = clipped_any or (feat_coef < 0.999999)
                        if clipped_any:
                            clip_events += 1
                        for k, t in client_update.items():
                            coef = head_coef if is_head(k) else feat_coef
                            dp_updates[k] += t * coef * equal_weight

                # Compute SNR using concatenated dp_updates
                if dp_param_names:
                    clipped_mean_vec = torch.cat([dp_updates[k].reshape(-1) for k in dp_param_names])
                    clipped_mean_norm = float(torch.norm(clipped_mean_vec, p=2).item())
                else:
                    clipped_mean_norm = 0.0

                q = min(1.0, m / N)
                noise_norm_sq = 0.0
                # Add noise per key depending on mode
                for idx, key in enumerate(dp_param_names):
                    if clip_mode == 'group':
                        Ck = C_head if is_head(key) else C_feat
                    else:
                        Ck = dp_clip_state.get(key, {}).get('C', clip_norm) if getattr(args, 'dp_adaptive', 1) == 1 and clip_mode == 'tensor' else clip_norm
                    dp_noise_std_k = sigma * Ck / m if m > 0 else 0.0
                    if dp_noise_std_k > 0.0:
                        if dp_seed_base is not None:
                            device_key = global_state[key].device
                            local_generator = torch.Generator(device=device_key)
                            local_generator.manual_seed(dp_seed_base + round * 9973 + idx)
                            noise = torch.randn(dp_updates[key].shape, device=device_key, dtype=dp_updates[key].dtype, generator=local_generator) * dp_noise_std_k
                        else:
                            noise = torch.randn_like(dp_updates[key]) * dp_noise_std_k
                    else:
                        noise = torch.zeros_like(dp_updates[key])
                    noise_norm_sq += float((noise.view(-1) ** 2).sum().item())
                    global_w[key] = global_state[key] + dp_updates[key] + noise

                noise_norm = math.sqrt(noise_norm_sq)
                snr = float('inf') if noise_norm == 0.0 else clipped_mean_norm / (noise_norm + 1e-12)
                clip_rate = clip_events / m if m > 0 else 0.0

                # Adaptive update of per-tensor clip norms for next round
                if adaptive:
                    q = min(1.0, m / N)
                    quant = float(getattr(args, 'dp_quantile', 0.8))
                    beta = float(getattr(args, 'dp_ema_beta', 0.2))
                    for k in dp_param_names:
                        norms = per_key_norms.get(k, [])
                        if norms:
                            sorted_norms = sorted(norms)
                            # Use floor-based index selection to avoid shadowed built-in round()
                            idxq = int(max(0, min(len(sorted_norms)-1, int((len(sorted_norms)-1) * quant))))
                            q_val = sorted_norms[idxq]
                            prev = dp_clip_state.get(k, {}).get('C', clip_norm)
                            newC = (1.0 - beta) * prev + beta * q_val
                            dp_clip_state[k]['C'] = float(max(1e-8, newC))
                    # Also adapt group Cs for group mode by aggregating per-key norms
                    if clip_mode == 'group':
                        head_vals=[]; feat_vals=[]
                        for k, norms in per_key_norms.items():
                            if is_head(k): head_vals += norms
                            else: feat_vals += norms
                        if head_vals:
                            sh=sorted(head_vals); ih=int((len(sh)-1)*quant); ch=(1.0-beta)*dp_clip_state['__HEAD__']['C'] + beta*sh[ih]
                            dp_clip_state['__HEAD__']['C']=float(max(1e-8, ch))
                        if feat_vals:
                            sf=sorted(feat_vals); ife=int((len(sf)-1)*quant); cf=(1.0-beta)*dp_clip_state['__FEAT__']['C'] + beta*sf[ife]
                            dp_clip_state['__FEAT__']['C']=float(max(1e-8, cf))

                epsilon_round = float('inf')
                epsilon_total = float('inf')
                alpha_round = None
                alpha_total = None
                if accountant is not None and sigma > 0.0:
                    rdp_step = accountant.step(q, sigma)
                    epsilon_round, alpha_round = accountant.epsilon(dp_delta, rdp_step)
                    epsilon_total, alpha_total = accountant.epsilon(dp_delta)

            if args.server_momentum:
                delta_w = copy.deepcopy(global_w)
                for key in delta_w:
                    delta_w[key] = old_w[key] - global_w[key]
                    moment_v[key] = args.server_momentum * moment_v[key] + (1-args.server_momentum) * delta_w[key]
                    global_w[key] = old_w[key] - moment_v[key]

            global_model.load_state_dict(global_w)
            if attack_ctx and round in attack_ctx['rounds']:
                _dump_attack_artifacts_image(attack_ctx, round, global_model, global_state, global_w, nets_this_round)


            #global_model.cuda()

            print('>> Current Round: {}'.format(round))
            logger.info('>> Current Round: {}'.format(round))

            mkdirs(args.modeldir+'fedavg/')

            # Save best based on last computed metric (few-shot or normal)
            # In normal mode, best_acc was updated using acc_normal above
            if best_acc > 0:
                torch.save(global_model.state_dict(), args.modeldir+'fedavg/'+'globalmodel'+args.log_file_name+'.pth')
                torch.save(nets[0].state_dict(), args.modeldir+'fedavg/'+'localmodel0'+args.log_file_name+'.pth')

            if args.dp_enable and dp_param_names:
                round_idx = round + 1
                if math.isinf(epsilon_round):
                    epsilon_round_str = 'inf'
                elif alpha_round is None:
                    epsilon_round_str = f'{epsilon_round:.4f}'
                else:
                    epsilon_round_str = f'{epsilon_round:.4f} (alpha {alpha_round})'

                if math.isinf(epsilon_total):
                    epsilon_total_str = 'inf'
                elif alpha_total is None:
                    epsilon_total_str = f'{epsilon_total:.4f}'
                else:
                    epsilon_total_str = f'{epsilon_total:.4f} (alpha {alpha_total})'

                dp_msg = (
                    f"DP round {round_idx} | target={args.dp_target} | m={m} | N={N} | q={q:.6f} | "
                    f"clip_norm={args.dp_clip_norm:.6f} | sigma={args.dp_noise_multiplier:.6f} | "
                    f"noise_std={dp_noise_std:.6f} | epsilon_round={epsilon_round_str} | "
                    f"epsilon_total={epsilon_total_str} | delta={dp_delta:.2e} | snr={snr:.6f} | "
                    f"clip_rate={clip_rate:.6f}")
                print(dp_msg)
                logger.info(dp_msg)
    else:
        # Normal mode: set total_classes for standard datasets
        if args.dataset == 'cifar10':
            total_classes = 10
        elif args.dataset == 'cifar100':
            total_classes = 100
        elif args.dataset == 'tinyimagenet':
            total_classes = 200
