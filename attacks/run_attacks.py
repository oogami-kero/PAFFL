import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def _load_metadata(dump_dir: Path):
    metadata_path = dump_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {dump_dir}")
    return json.loads(metadata_path.read_text())


def _load_round_tensor(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def label_inference_attack(head_update, topk=5):
    if head_update is None or "all_classify.weight" not in head_update:
        return {}
    weight = head_update["all_classify.weight"]
    matrix = weight.reshape(weight.shape[0], -1)
    norms = torch.norm(matrix, dim=1).cpu().numpy()
    top_indices = norms.argsort()[::-1][:topk]
    return {
        "top_classes": top_indices.tolist(),
        "top_norms": norms[top_indices].tolist(),
    }


def membership_inference_attack(train_logits_dict, test_logits_dict):
    if train_logits_dict is None or test_logits_dict is None:
        return {}

    train_logits = torch.tensor(train_logits_dict["logits"], dtype=torch.float32)
    train_labels = torch.tensor(train_logits_dict["labels"], dtype=torch.long)
    test_logits = torch.tensor(test_logits_dict["logits"], dtype=torch.float32)
    test_labels = torch.tensor(test_logits_dict["labels"], dtype=torch.long)

    if train_logits.size(0) == 0 or test_logits.size(0) == 0:
        return {}

    train_loss = F.cross_entropy(train_logits, train_labels, reduction="none").cpu().numpy()
    test_loss = F.cross_entropy(test_logits, test_labels, reduction="none").cpu().numpy()

    scores = np.concatenate([-train_loss, -test_loss])
    labels = np.concatenate([np.ones_like(train_loss), np.zeros_like(test_loss)])
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    return {
        "roc_auc": float(auc),
        "train_loss_mean": float(train_loss.mean()),
        "test_loss_mean": float(test_loss.mean()),
        "train_loss_std": float(train_loss.std()),
        "test_loss_std": float(test_loss.std()),
    }


def property_inference_attack(client_updates, client_class_counts):
    if not client_updates or not client_class_counts:
        return {}

    results = {}
    class_counts = {str(k): {int(cls): int(cnt) for cls, cnt in v.items()} for k, v in client_class_counts.items()}
    sample_client = next(iter(class_counts))
    class_ids = sorted(class_counts[sample_client].keys())

    for class_id in class_ids:
        norms_pos = []
        norms_neg = []
        for client_id, updates in client_updates.items():
            client_key = str(client_id)
            counts = class_counts.get(client_key, {})
            if "all_classify.weight" not in updates:
                continue
            weight = updates["all_classify.weight"]
            matrix = weight.reshape(weight.shape[0], -1)
            row_norm = torch.norm(matrix[class_id]).item()
            if counts.get(class_id, 0) > 0:
                norms_pos.append(row_norm)
            else:
                norms_neg.append(row_norm)
        if norms_pos and norms_neg:
            results[str(class_id)] = {
                "mean_norm_pos": float(np.mean(norms_pos)),
                "mean_norm_neg": float(np.mean(norms_neg)),
                "num_pos": len(norms_pos),
                "num_neg": len(norms_neg),
            }
    return results


def inversion_svd_attack(head_update, output_path: Path, topk=5):
    if head_update is None or "all_classify.weight" not in head_update:
        return {}
    weight = head_update["all_classify.weight"]
    matrix = weight.reshape(weight.shape[0], -1)
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        return {"error": "SVD failed"}
    torch.save({
        "singular_values": s.cpu(),
        "left_vectors": u.cpu(),
        "right_vectors": vh.cpu(),
    }, output_path)
    top_values = s[:topk].cpu().numpy()
    return {
        "top_singular_values": top_values.tolist(),
        "svd_dump": str(output_path.name),
    }


def process_round(dump_dir: Path, round_idx: int, metadata, attacks, topk):
    suffix = f"round_{round_idx:04d}"
    head_state = _load_round_tensor(dump_dir / f"{suffix}_head_state.pt")
    head_update = _load_round_tensor(dump_dir / f"{suffix}_head_update.pt")
    train_logits = _load_round_tensor(dump_dir / f"{suffix}_probe_train_logits.pt")
    test_logits = _load_round_tensor(dump_dir / f"{suffix}_probe_test_logits.pt")
    client_updates = _load_round_tensor(dump_dir / f"{suffix}_client_updates.pt")

    round_report = {}

    if 'label' in attacks:
        round_report['label_inference'] = label_inference_attack(head_update, topk=topk)

    if 'membership' in attacks:
        round_report['membership_inference'] = membership_inference_attack(train_logits, test_logits)

    if 'property' in attacks:
        round_report['property_inference'] = property_inference_attack(
            client_updates or {}, metadata.get('client_class_counts', {}))

    if 'inversion' in attacks:
        inversion_path = dump_dir / f"{suffix}_svd.pt"
        round_report['inversion_svd'] = inversion_svd_attack(head_update, inversion_path, topk=topk)

    return round_report


def main():
    parser = argparse.ArgumentParser(description="Run reproducible privacy attacks on F2L artifacts")
    parser.add_argument('--dump_dir', type=str, required=True, help='Path to attack dump directory')
    parser.add_argument('--attacks', type=str, default='label,membership,property,inversion',
                        help='Comma-separated list of attacks to run')
    parser.add_argument('--topk', type=int, default=5, help='Top-k classes/singular values to report')
    parser.add_argument('--output', type=str, default='attack_report.json', help='Output JSON report name')
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    if not dump_dir.exists():
        raise FileNotFoundError(f"Dump directory {dump_dir} does not exist")

    metadata = _load_metadata(dump_dir)
    attack_list = [name.strip() for name in args.attacks.split(',') if name.strip()]

    rounds = metadata.get('attack_rounds', [])
    report = {
        'dump_dir': str(dump_dir),
        'dataset': metadata.get('dataset'),
        'use_transform_layer': metadata.get('use_transform_layer'),
        'attacks': attack_list,
        'rounds': {},
    }

    for round_idx in rounds:
        round_report = process_round(dump_dir, round_idx, metadata, attack_list, args.topk)
        report['rounds'][str(round_idx)] = round_report

    output_path = dump_dir / args.output
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Attack report written to {output_path}")


if __name__ == '__main__':
    main()
