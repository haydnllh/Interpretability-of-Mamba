import torch.nn as nn 
from mamba_ssm.modules.mamba import Mamba
from einops.layers.torch import Rearrange

class MambaImageClassifier(nn.Module):
    def __init__(self, mamba_config, img_channels, patch_size, num_classes):
        super().__init__()
        
        d_model = mamba_config.d_model
        
        self.rearrange = Rearrange(
            'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
            p1=patch_size, p2=patch_size
        ) 
        patch_dim = img_channels * patch_size * patch_size
        
        self.embed = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.mamba = Mamba(mamba_config)
        
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.rearrange(x)
        x = self.embed(x)
        
        x = self.mamba(x)
            
        x = self.norm(x.mean(dim=1))
        logits = self.classifier(x)
        return logits
    
    def get_params(self, x, requires_grad=False):
        x = self.rearrange(x)
        x = self.embed(x)
        
        return self.mamba.get_params(x, requires_grad=requires_grad)