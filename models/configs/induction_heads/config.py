from dataclasses import dataclass, field

@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 2
    d_intermediate: int = 0
    vocab_size: int = 7
    attn_layer_idx: list = None
    attn_cfg: dict = None
    ssm_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 1
    tie_embeddings: bool = False