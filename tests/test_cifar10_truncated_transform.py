import os
import sys
import numpy as np
from torchvision import transforms

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import CIFAR10_truncated


def test_cifar10_truncated_pil_transform():
    """CIFAR10_truncated should handle PIL-based transforms."""
    dataset = CIFAR10_truncated.__new__(CIFAR10_truncated)
    dataset.data = np.random.randint(0, 255, (1, 32, 32, 3), dtype=np.uint8)
    dataset.target = np.array([0])
    dataset.transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset.target_transform = None
    img, target = dataset[0]
    assert img.shape[0] == 3
    assert target == 0
