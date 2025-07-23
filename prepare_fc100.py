# scripts/make_fc100_cache.py
import os, pickle, torchvision, numpy as np
from torchvision.datasets import CIFAR100
root = "./data"
split_files = {
    'train': 'fc100-cache-train.pkl',
    'val'  : 'fc100-cache-val.pkl',
    'test' : 'fc100-cache-test.pkl'
}
cifar = CIFAR100(root, train=True, download=True)
images = cifar.data        # (50000, 32, 32, 3)
labels = np.array(cifar.targets)
# FC100 uses 60 base, 20 val, 20 novel classes (as in the F2L paper)
base_cls  = list(range(0, 60))
val_cls   = list(range(60, 80))
test_cls  = list(range(80, 100))
for name, cls in zip(['train','val','test'], [base_cls, val_cls, test_cls]):
    idx = np.isin(labels, cls)
    X, y = images[idx], labels[idx]
    with open(os.path.join(root, split_files[name]), 'wb') as f:
        pickle.dump((X, y), f)
    print(f"wrote {split_files[name]}  shape={X.shape}")
