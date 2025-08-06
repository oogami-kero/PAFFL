import torch
import torch.nn as nn
import torch.nn.functional as F

# Patch for torch versions missing RMSNorm required by Opacus
if not hasattr(nn, "RMSNorm"):
    class RMSNorm(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
    nn.RMSNorm = RMSNorm

from opacus.grad_sample import register_grad_sampler


@register_grad_sampler(nn.MultiheadAttention)
def multiheadattention_grad_sampler(module, activations, backprops):
    """Per-sample gradients for ``nn.MultiheadAttention``.

    Computes gradients for ``in_proj_weight`` & ``in_proj_bias`` as well as
    ``out_proj.weight`` and ``out_proj.bias``.
    """
    query, key, value = activations[:3]
    grad_output = backprops

    # Move batch dimension to front for simpler indexing
    if not module.batch_first:
        query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        grad_output = grad_output.transpose(0, 1)

    batch_size, seq_len, embed_dim = query.shape
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    with torch.no_grad():
        w_q, w_k, w_v = module.in_proj_weight.chunk(3, dim=0)
        if module.in_proj_bias is not None:
            b_q, b_k, b_v = module.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None

        q = F.linear(query, w_q, b_q)
        k = F.linear(key, w_k, b_k)
        v = F.linear(value, w_v, b_v)

        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn, v)
        context_reshape = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Gradients for out projection parameters
        gs_out_w = torch.einsum("ble,blf->bef", grad_output, context_reshape)
        gs_out_b = grad_output.sum(dim=1)

        grad_context = torch.einsum("ble,ef->blf", grad_output, module.out_proj.weight)
        grad_context = grad_context.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        grad_attn = torch.matmul(grad_context, v.transpose(-2, -1))
        grad_v = torch.matmul(attn.transpose(-2, -1), grad_context)

        tmp = grad_attn * attn
        grad_scores = tmp - attn * tmp.sum(dim=-1, keepdim=True)

        grad_q = torch.matmul(grad_scores, k) * scale
        grad_k = torch.matmul(grad_scores.transpose(-2, -1), q) * scale

        grad_q = grad_q.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        grad_k = grad_k.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        grad_v = grad_v.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        gs_w_q = torch.einsum("ble,blf->bef", grad_q, query)
        gs_w_k = torch.einsum("ble,blf->bef", grad_k, key)
        gs_w_v = torch.einsum("ble,blf->bef", grad_v, value)
        gs_in_w = torch.cat([gs_w_q, gs_w_k, gs_w_v], dim=1)

        if module.in_proj_bias is not None:
            gs_b_q = grad_q.sum(dim=1)
            gs_b_k = grad_k.sum(dim=1)
            gs_b_v = grad_v.sum(dim=1)
            gs_in_b = torch.cat([gs_b_q, gs_b_k, gs_b_v], dim=1)
        else:
            gs_in_b = None

    ret = {
        module.in_proj_weight: gs_in_w,
        module.out_proj.weight: gs_out_w,
        module.out_proj.bias: gs_out_b,
    }
    if gs_in_b is not None:
        ret[module.in_proj_bias] = gs_in_b
    return ret
