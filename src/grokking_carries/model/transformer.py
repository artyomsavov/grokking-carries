import torch
import torch.nn as nn
from grokking_carries.config import ModelConfig
from grokking_carries.types import TokenIndices, Logits
from grokking_carries.interpretability.hooks import HookedRootModule, HookPoint
from grokking_carries.model.components import TokenEmbedder, PositionalEmbedder, TransformerBlock

class GrokkingCarriesTransformer(HookedRootModule):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = TokenEmbedder(cfg)
        self.pos_embed = PositionalEmbedder(cfg)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.hook_logits = HookPoint()

        self.setup_hooks()

    def forward(self, tokens: TokenIndices) -> Logits:
        x = self.embed(tokens) + self.pos_embed(tokens)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.unembed(x)

        return self.hook_logits(logits)
