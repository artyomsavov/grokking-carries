import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from grokking_carries.config import ModelConfig
from grokking_carries.types import TokenIndices, HiddenStates
from grokking_carries.interpretability.hooks import HookPoint

class TokenEmbedder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.hook_embed = HookPoint()

    def forward(self, tokens: TokenIndices) -> HiddenStates:
        x = self.embed(tokens)
        return self.hook_embed(x)

class PositionalEmbedder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.hook_pos_embed = HookPoint()

    def forward(self, tokens: TokenIndices) -> HiddenStates:
        batch_size, seq_len = tokens.shape
        pos = torch.arange(seq_len, dtype=torch.long, device=self.pos_embed.weight.device)
        x = self.pos_embed(pos)
        x = x.expand(batch_size, -1, -1)
        return self.hook_pos_embed(x)

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_K = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_V = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.W_O = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.hook_q = HookPoint()
        self.hook_k = HookPoint()
        self.hook_v = HookPoint()
        self.hook_pattern = HookPoint()
        self.hook_z = HookPoint()

    def forward(self, x: HiddenStates) -> HiddenStates:
        batch, seq_len, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_head

        q = self.W_Q(x).view(batch, seq_len, n_heads, d_head)
        q = self.hook_q(q)
        k = self.W_K(x).view(batch, seq_len, n_heads, d_head)
        k = self.hook_k(k)
        v = self.W_V(x).view(batch, seq_len, n_heads, d_head)
        v = self.hook_v(v)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)

        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        mask = torch.tril(mask).view(1, 1, seq_len, seq_len)
        scores = scores.masked_fill(~mask, float('-inf'))

        pattern = F.softmax(scores, dim=-1)
        pattern = self.hook_pattern(pattern)

        z = torch.matmul(pattern, v)
        z = z.transpose(1, 2)
        z = self.hook_z(z)

        z_flat = z.reshape(batch, seq_len, self.cfg.d_model)
        out = self.W_O(z_flat)
        return out

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.W_in = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.W_out = nn.Linear(cfg.d_mlp, cfg.d_model)
        self.act = nn.GELU()
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x: HiddenStates) -> HiddenStates:
        x = self.W_in(x)
        x = self.hook_pre(x)
        x = self.act(x)
        x = self.hook_post(x)
        return self.W_out(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: HiddenStates) -> HiddenStates:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
