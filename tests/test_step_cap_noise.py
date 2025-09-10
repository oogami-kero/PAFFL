import argparse
import os
import sys
import types
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
opacus_stub = types.SimpleNamespace(PrivacyEngine=None, grad_sample=types.SimpleNamespace(GradSampleModule=None))
sys.modules.setdefault('opacus', opacus_stub)
sys.modules.setdefault('opacus.grad_sample', opacus_stub.grad_sample)
model_stub = types.SimpleNamespace(WordEmbed=object)
sys.modules.setdefault('model', model_stub)
utils_stub = types.SimpleNamespace()
sys.modules.setdefault('utils', utils_stub)
torchvision_transforms_stub = types.SimpleNamespace(
    Normalize=lambda *a, **k: None,
    Compose=lambda *a, **k: None,
    RandomCrop=lambda *a, **k: None,
    RandomHorizontalFlip=lambda *a, **k: None,
    RandomRotation=lambda *a, **k: None,
    ToTensor=lambda *a, **k: None,
    ToPILImage=lambda *a, **k: None,
)
torchvision_stub = types.SimpleNamespace(transforms=torchvision_transforms_stub)
sys.modules.setdefault('torchvision', torchvision_stub)
sys.modules.setdefault('torchvision.transforms', torchvision_transforms_stub)
import main_image


def test_r_k_uses_step_norm():
    torch.manual_seed(0)
    global_w = {'w': torch.zeros(1, 1)}
    deltas = {
        0: {'w': torch.tensor([[1.0]])},
        1: {'w': torch.tensor([[1.0]])},
    }
    client_norms = {}
    client_scales = {0: 1.0, 1: 1.0}
    args = argparse.Namespace(
        dp_clip=1.0,
        dp_noise=0.0,
        dp_constant_noise=False,
        server_momentum=0.0,
        server_lr=1.0,
        target_step=1.0,
    )
    layer_clips = {'w': 1.0}
    noise_multipliers = {'w': 1e6}

    num_clients = len(deltas)
    avg_update = torch.stack([d['w'] for d in deltas.values()]).mean(0)
    noise_std = noise_multipliers['w'] * layer_clips['w'] / num_clients
    noise = torch.randn_like(avg_update) * noise_std
    u = avg_update + noise
    step = args.server_lr * u
    step_norm = torch.norm(step).item()
    assert step_norm > 1e3
    expected_r_k = min(1.0, args.target_step / max(step_norm, 1e-12))

    torch.manual_seed(0)
    _, _, _, _, _, _, r_k, _ = main_image.aggregate_deltas(
        global_w,
        deltas,
        client_norms,
        client_scales,
        args,
        layer_clips,
        noise_multipliers,
    )
    assert abs(r_k - expected_r_k) < 1e-6

