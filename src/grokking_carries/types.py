from typing import Callable, Literal
from torch import Tensor
from jaxtyping import Float, Int

# Размерности осей
type Batch = int
type SeqLen = int
type DModel = int
type DHead = int
type NumHeads = int
type Vocab = int
type LayerIdx = int

# Семантические тензоры
type TokenIndices = Int[Tensor, 'batch seq_len']
type Logits = Float[Tensor, 'batch seq_len vocab']
type HiddenStates = Float[Tensor, 'batch seq_len d_model']

# Тензоры внутренних механизмов 
type QKVTensor = Float[Tensor, 'batch seq_len num_heads d_head']
type AttentionScores = Float[Tensor, 'batch num_heads seq_len seq_len']
type MLPHidden = Float[Tensor, 'batch seq_len 4*d_model']

# Типизация для системы хуков
type HookName = str
type HookFunction = Callable[[Tensor, HookName], Tensor | None]
type ActivationCache = dict[HookName, Tensor]

