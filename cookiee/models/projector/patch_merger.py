import torch
from torch import nn
from .mlp import MLPProjector

from . import PROJECTOR

# copy from Qwen2VL
class PatchMerger(nn.Module):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class PatchMergeProjector(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.patch_merger = PatchMerger(
                                dim=config.text_config.hidden_size, 
                                context_dim=config.vision_config.embed_dim, 
                                spatial_merge_size=config.vision_config.spatial_merge_size
                            )
        if config.enable_mlp:
            self.mlp = MLPProjector(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_merger(x)
        if hasattr(self, "mlp"):
            x = self.mlp(x)
        return x
    

PROJECTOR._do_register("patch_merger", PatchMergeProjector)