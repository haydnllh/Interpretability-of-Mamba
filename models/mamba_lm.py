from mamba_ssm.modules.mamba import Mamba
import torch.nn as nn
import torch

class MambaLM(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.d_model)
        #self.pos_embedding = nn.Embedding(2048, config.d_model)
        self.dropout = nn.Dropout(0.2)
        self.mamba = Mamba(config)
        self.lm_head = nn.Linear(config.d_model, vocab_size)
        #self.lm_head.weight = self.embedding.weight  

    def forward(self, x):
        B, T = x.shape
        #pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        #x = self.embedding(x) + self.pos_embedding(pos)
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.mamba(x) 
        x = self.dropout(x)
        logits = self.lm_head(x)
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=200, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device
        cfg = self.mamba.config

        B = prompt_ids.shape[0]
        dtype = torch.cfloat if "Complex" in cfg.ssm_type else torch.float
        caches = [
            (
                torch.zeros(B, cfg.d_inner, cfg.d_state, device=device, dtype=dtype),
                torch.zeros(B, cfg.d_inner, cfg.d_conv - 1, device=device),
            )
            for _ in self.mamba.layers
        ]

        L = prompt_ids.shape[1]
        for t in range(L):
            tok = prompt_ids[:, t]
            emb = self.embedding(tok)
            with torch.no_grad():
                _, caches = self.mamba.step(emb, caches)

        generated = prompt_ids
        for i in range(max_new_tokens):
            emb = self.embedding(tok)

            with torch.no_grad():
                out, caches = self.mamba.step(emb, caches)

            logits = self.lm_head(out) / temperature
            probs  = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tok = next_token.squeeze(1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated
    
    def get_params(self, x, requires_grad=False):
        x = self.embedding(x)
        
        return self.mamba.get_params(x, requires_grad=requires_grad)