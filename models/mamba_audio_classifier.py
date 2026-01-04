import torch
import torch.nn as nn 
from mamba_ssm.modules.mamba import Mamba


class MambaAudioClassifier(nn.Module):
    def __init__(self, mamba_config, num_classes, n_input):
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
        
        self.mamba = Mamba(mamba_config)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        x = self.embed(x)
        
        x = self.mamba(x)
            
        logits = self.classifier(x.mean(1))
        return logits
    
    def get_params(self, x):
        x = self.embed(x)
        
        return self.mamba.get_params(x)
