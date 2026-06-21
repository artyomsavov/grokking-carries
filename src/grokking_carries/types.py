from typing import Callable, Literal
from torch import Tensor
from jaxtyping import Float, Int

# Размеры осей (документационные алиасы)
type Batch = int
type SeqLen = int
type DModel = int
type DHead = int
type NumHeads = int
type Vocab = int
type DMlp = int

# === Семантические тензоры ===
type TokenIndices = Int[Tensor, 'batch seq_len']
type Logits = Float[Tensor, 'batch seq_len vocab_size']
type HiddenStates = Float[Tensor, 'batch seq_len d_model']

# === Тензоры внутренних механизмов ===
type QKVTensor = Float[Tensor, 'batch seq_len num_heads d_head']
type AttentionScores = Float[Tensor, 'batch num_heads seq_len seq_len']
type MLPHidden = Float[Tensor, 'batch seq_len d_mlp'] 

# === Типизация для системы хуков ===
# Строгий контракт для избежания опечаток при извлечении активаций
type HookName = Literal[
    "embed",
    "pos_embed",
    "blocks.attn.hook_q",
    "blocks.attn.hook_k",
    "blocks.attn.hook_v",
    "blocks.attn.hook_z",
    "blocks.attn.hook_pattern",
    "blocks.mlp.hook_pre",
    "blocks.mlp.hook_post",
    "unembed"
] | str

type HookFunction = Callable[[Tensor, HookName], Tensor | None]
type ActivationCache = dict[HookName, Tensor]

