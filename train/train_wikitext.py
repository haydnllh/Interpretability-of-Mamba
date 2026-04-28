import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import os
from mamba_ssm.modules.mamba import MambaConfig
from models.mamba_lm import MambaLM
from models.dataloaders.dataloaders import get_wikitext
from models.selective_copying_mamba import train
from models.train_induction_head import train_induction_head

class MambaWikitextTrainer(L.LightningModule):
    def __init__(self, mamba_config, meta, lr=2e-3, weight_decay=0.1):
        super().__init__()
        self.meta = meta
        self.weight_decay = weight_decay
        
        mamba_cfg = MambaConfig(**mamba_config)
        self.model = MambaLM(mamba_cfg, meta['vocab_size'])
        
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        logits = self.model(x)
        return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch 

        logits = self(x) 

        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        self.log("val_loss", loss, on_step=True, prog_bar=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )
        
        self.log("test_loss", loss, prog_bar=True)
        return loss

        
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [scheduler]
    
    def generate(self, prompt, max_new_tokens=200, temperature=1.0):
        self.eval()
        device = next(self.parameters()).device

        if self.meta["tokenizer"] == "char":
            stoi = self.meta["stoi"]
            itos = self.meta["itos"]
            input_ids = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)

        elif self.meta["tokenizer"] == "gpt2":
            tok = self.meta["hf_tokenizer"]
            input_ids = torch.tensor(tok.encode(prompt), dtype=torch.long).unsqueeze(0)

        input_ids = input_ids.to(device)

        output_ids = self.model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature)

        if self.meta["tokenizer"] == "char":
            itos = self.meta["itos"]
            text = "".join([itos[int(i)] for i in output_ids[0]])

        elif self.meta["tokenizer"] == "gpt2":
            tok = self.meta["hf_tokenizer"]
            text = tok.decode(output_ids[0].tolist())

        return text
    

class TestEveryNSteps(Callback):
    def __init__(self, test_loader, every_n_steps=10000):
        self.test_loader = test_loader
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step % self.every_n_steps == 0 and 
                trainer.global_step > 0):
            pl_module.eval()
            losses = []
            with torch.no_grad():
                for batch in self.test_loader:
                    x, y = batch
                    x = x.to(pl_module.device)
                    y = y.to(pl_module.device)
                    logits = pl_module(x)
                    loss = pl_module.loss_fn(
                        logits.view(-1, logits.size(-1)), y.view(-1)
                    )
                    losses.append(loss.item())
            avg_loss = sum(losses) / len(losses)
            pl_module.log("test_loss_step", avg_loss, prog_bar=True)
            pl_module.train()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
    else:
        cfg = OmegaConf.load("models/configs/wikitext/wikitext.yaml")
        
    train_loader, val_loader, meta = get_wikitext(batch_size=cfg.train.batch_size, tokenizer=cfg.train.tokenizer, block_size=cfg.train.block_size, dataset_name="wikitext-2-raw-v1")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/scratch/lhl1g23/mamba/checkpoints/{timestamp if args.checkpoint_path is None else args.checkpoint_path}",
        filename="best",
        save_top_k=-1,
        save_last=True,
        monitor="val_loss"
    )
    
    version = "version_0"
    logger_tb = TensorBoardLogger(
        save_dir="/scratch/lhl1g23/mamba/lightning_logs/",
        name=args.checkpoint_path,
        version=version
    )
    
    if args.from_checkpoint is not None:
        model = MambaWikitextTrainer.load_from_checkpoint(
            args.from_checkpoint,
            mamba_config=cfg.mamba,
            meta=meta
        ).to('cuda')
        model.train()
    else:
        model = MambaWikitextTrainer(cfg.mamba, 
                                    meta,
                                    lr=cfg.train.lr,
                                    weight_decay=cfg.train.weight_decay)
    
    test_callback = TestEveryNSteps(val_loader, every_n_steps=10000)
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs, 
        accelerator="auto", 
        gradient_clip_val=1.0,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, RichProgressBar(), test_callback],
        enable_progress_bar=True,
        logger=[logger_tb],
        deterministic=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(f"/scratch/lhl1g23/mamba/checkpoints/{args.checkpoint_path}/final.ckpt")
    
    trainer.test(model, dataloaders=val_loader)

    
if __name__ == "__main__":
    print("Starting training...")
    main()