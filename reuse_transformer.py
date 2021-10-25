import torch
import torchsummary
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import torch.nn.functional as F


class ExactAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., reuse_ratio=0.5):
        super(ExactAttentionLayer, self).__init__()
        self.reuse_multihead_attn = nn.MultiheadAttention(embed_dim, int(num_heads * reuse_ratio), dropout,
                                                          batch_first=True)

    def forward(self, q, k, v, attn_mask=None):
        outputs, reuse_attn = self.reuse_multihead_attn(q, k, v, attn_mask)
        # outputs [bs, tgt_len, dims]
        # reuse_attn [bs, tgt_len, tgt_len]
        return outputs, reuse_attn


class ReuseAttenLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, reuse_ratio=0.5, dropout=0.):
        super(ReuseAttenLayer, self).__init__()
        self.exact_layer = ExactAttentionLayer(embed_dim, num_heads, dropout, 1 - reuse_ratio)
        self.v_proj = nn.Parameter(torch.randn([embed_dim, embed_dim]))
        self.proj = nn.Linear(2 * embed_dim, embed_dim)
        self.register_buffer('reuse_atten', None)

    def forward(self, q, k, v, atten_mask=None):
        outputs, reuse_attn = self.exact_layer(q, k, v, atten_mask)
        self.reuse_atten = reuse_attn.detach()
        reuse_outputs = torch.bmm(self.reuse_atten, v)  # [bs, tgt_len, dims]
        out = torch.einsum('b l d, d d -> b l d', reuse_outputs, self.v_proj)
        out = torch.cat([outputs, out], dim=2)  # [bs, tgt_len, 2*dims]
        out = self.proj(out)
        return out


reuse_attn = ReuseAttenLayer(128, 8, reuse_ratio=0.5).cuda()
a = torch.randn([2, 256, 128]).cuda()
m = reuse_attn(a, a, a)
l = m.sum()
l.backward()

for name, layer in reuse_attn.named_parameters():
    print('{}, {}'.format(name, layer.grad))
