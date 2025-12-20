import torch
import torch.nn as nn 
from mamba_ssm.modules.mamba import Mamba


class MambaAudioClassifier(nn.Module):
    def __init__(self, mamba_config, num_classes, n_input, pooling_factor, n_layers):
        super().__init__()
        
        d_model = mamba_config.d_model
        
        self.embed = nn.Sequential(
            nn.Linear(n_input, d_model),
            nn.LayerNorm(d_model)
        )
        
        """ self.embed = nn.Sequential(
            nn.LayerNorm(n_input),
            nn.Linear(n_input, d_model),
            nn.LayerNorm(d_model)
        ) """
        
        self.layers = nn.ModuleList([MambaBlock(mamba_config, pooling_factor) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        x = x[:,:,0].unsqueeze(2)
        x = self.embed(x)
        
        for layer in self.layers:
            x = layer(x)
            
        logits = self.classifier(x.mean(1))
        return logits
    
    def get_parameters(self, x):
        x = self.embed(x)
        
        for block in self.layers:
            mamba = block[0]
            yield mamba.get_params(x)
            
class MambaBlock(nn.Module):
    def __init__(self, mamba_config, pooling_factor):
        super().__init__()
        self.mamba = Mamba(mamba_config)
        self.pool = nn.AvgPool1d(pooling_factor)
        
    def forward(self, x):
        x = self.mamba(x)
        x = x.transpose(1, 2)
        x = self.pool(x)
        x = x.transpose(1, 2)
        return x
