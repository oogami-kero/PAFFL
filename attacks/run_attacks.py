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
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
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


def _to_tensor(x, dtype=None):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=dtype) if dtype is not None else x.clone().detach()
    return torch.tensor(x, dtype=dtype) if dtype is not None else torch.tensor(x)


def _map_labels(labels: torch.Tensor, mapping: dict, num_classes: int):
    if mapping is None:
        # If no mapping, ensure labels are within range
        mask = (labels >= 0) & (labels < num_classes)
        return labels[mask], mask
    mapped = []
    mask_list = []
    for v in labels.tolist():
        mv = mapping.get(int(v)) if mapping is not None else int(v)
        if mv is not None and 0 <= int(mv) < num_classes:
            mapped.append(int(mv))
            mask_list.append(True)
        else:
            mask_list.append(False)
    if not mapped:
        return torch.empty(0, dtype=torch.long), torch.zeros_like(labels, dtype=torch.bool)
    mapped_t = torch.tensor(mapped, dtype=torch.long)
    mask = torch.tensor(mask_list, dtype=torch.bool)
    return mapped_t, mask


def membership_inference_attack(train_logits_dict, test_logits_dict, class_to_head_index=None):
    if train_logits_dict is None or test_logits_dict is None:
        return {}

    train_logits = _to_tensor(train_logits_dict.get("logits"), dtype=torch.float32)
    train_labels_raw = _to_tensor(train_logits_dict.get("labels"), dtype=torch.long)
    test_logits = _to_tensor(test_logits_dict.get("logits"), dtype=torch.float32)
    test_labels_raw = _to_tensor(test_logits_dict.get("labels"), dtype=torch.long)

    if train_logits is None or test_logits is None:
        return {}

    result = {}
    # Compute CE-based AUC if labels are available and mappable
    if train_labels_raw is not None and test_labels_raw is not None and train_logits.ndim == 2 and test_logits.ndim == 2:
        num_classes = train_logits.shape[1]
        train_labels, mask_train = _map_labels(train_labels_raw, class_to_head_index, num_classes)
        test_labels, mask_test = _map_labels(test_labels_raw, class_to_head_index, num_classes)
        lg_train_ce = train_logits[mask_train]
        lg_test_ce = test_logits[mask_test]
        if lg_train_ce.numel() > 0 and lg_test_ce.numel() > 0:
            train_loss = F.cross_entropy(lg_train_ce, train_labels, reduction="none").cpu().numpy()
            test_loss = F.cross_entropy(lg_test_ce, test_labels, reduction="none").cpu().numpy()
            scores_ce = np.concatenate([-train_loss, -test_loss])
            labels_ce = np.concatenate([np.ones_like(train_loss), np.zeros_like(test_loss)])
            try:
                auc_ce = roc_auc_score(labels_ce, scores_ce)
            except ValueError:
                auc_ce = float("nan")
            result.update({
                "roc_auc_ce": float(auc_ce),
                "train_loss_mean": float(train_loss.mean()),
                "test_loss_mean": float(test_loss.mean()),
                "train_loss_std": float(train_loss.std()),
                "test_loss_std": float(test_loss.std()),
            })

    # Always compute confidence-based AUC (label-free fallback)
    try:
        prob_train = torch.softmax(train_logits, dim=1).max(dim=1).values.cpu().numpy()
        prob_test = torch.softmax(test_logits, dim=1).max(dim=1).values.cpu().numpy()
        if prob_train.size > 0 and prob_test.size > 0:
            scores_conf = np.concatenate([prob_train, prob_test])
            labels_conf = np.concatenate([np.ones_like(prob_train), np.zeros_like(prob_test)])
            auc_conf = roc_auc_score(labels_conf, scores_conf)
            result.update({
                "roc_auc_conf": float(auc_conf),
                "train_conf_mean": float(prob_train.mean()),
                "test_conf_mean": float(prob_test.mean()),
            })
    except Exception:
        pass

    if not result:
        result = {"note": "no valid samples after label mapping"}
    return result


def property_inference_attack(client_updates, client_class_counts, class_to_head_index=None):
    if not client_updates or not client_class_counts:
        return {}

    results = {}
    class_counts = {str(k): {int(cls): int(cnt) for cls, cnt in v.items()} for k, v in client_class_counts.items()}
    sample_client = next(iter(class_counts), None)
    if sample_client is None:
        return {}
    class_ids = sorted(class_counts[str(sample_client)].keys())

    for class_id in class_ids:
        norms_pos = []
        norms_neg = []
        for client_id, updates in client_updates.items():
            client_key = str(client_id)
            counts = class_counts.get(client_key, {})
            if "all_classify.weight" not in updates:
                continue
            weight = updates.get("all_classify.weight")
            if weight is None:
                continue
            matrix = weight.reshape(weight.shape[0], -1)
            # Map raw class id to head row index if mapping provided
            if class_to_head_index is not None:
                head_idx = class_to_head_index.get(int(class_id))
                if head_idx is None:
                    continue
            else:
                head_idx = int(class_id)
            if not (0 <= head_idx < matrix.shape[0]):
                continue
            row_norm = torch.norm(matrix[head_idx]).item()
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
        round_report['membership_inference'] = membership_inference_attack(
            train_logits, test_logits, metadata.get('class_to_head_index'))

    if 'property' in attacks:
        round_report['property_inference'] = property_inference_attack(
            client_updates or {}, metadata.get('client_class_counts', {}), metadata.get('class_to_head_index'))

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
