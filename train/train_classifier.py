import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
import argparse
from datetime import datetime
from omegaconf import OmegaConf
import os
from mamba_ssm.modules.mamba import MambaConfig
from models.mamba_image_classifier import MambaImageClassifier
from models.mamba_audio_classifier import MambaAudioClassifier
from models.dataloaders.dataloaders import get_mnist, get_cifar, get_sc
from models.selective_copying_mamba import train
from models.train_induction_head import train_induction_head

class MambaClassifierTrainer(L.LightningModule):
    def __init__(self, mamba_config, model_config, data, lr=2e-3, weight_decay=0.1):
        super().__init__()
        self.data = data
        self.weight_decay = weight_decay
        
        mamba_cfg = MambaConfig(**mamba_config)
        
        if data == "cifar" or data == "mnist":
            self.classifier = MambaImageClassifier(mamba_cfg, **model_config)
            
        if data == "sc":
            self.classifier = MambaAudioClassifier(mamba_cfg, **model_config)
        
        self.data = data
        self.lr = lr
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, x):
        logits = self.classifier(x)
        return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
                
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return [opt], [scheduler]
    
def main():
    parser = argparse.ArgumentParser()
    datasets = ["mnist", "sc", "cifar", "selective_copying"]
    parser.add_argument("--dataset", type=str, default="mnist", choices=datasets)
    parser.add_argument("--slurm", type=bool, default=False)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    args = parser.parse_args()
    
    if args.dataset == "mnist":
        if args.config is not None:
            cfg = OmegaConf.load(args.config)
        else:
            cfg = OmegaConf.load("models/configs/mnist/mnist.yaml")
        train_loader, val_loader = get_mnist(batch_size=cfg.train.batch_size)
    elif args.dataset == "cifar":
        if args.config is not None:
            cfg = OmegaConf.load(args.config)
        else:
            cfg = OmegaConf.load("models/configs/cifar/cifar_4_patchsize.yaml")
        train_loader, val_loader = get_cifar(batch_size=cfg.train.batch_size)
    elif args.dataset == "sc":
        if args.config is not None:
            cfg = OmegaConf.load(args.config)
        else:
            cfg = OmegaConf.load("models/configs/sc/sc.yaml")
        train_loader, val_loader, test_loader = get_sc(batch_size=cfg.train.batch_size)
        
    elif args.dataset == "selective_copying":
        train(f'{args.checkpoint_path}')
        return

    else:
        train_induction_head(out_file=args.checkpoint_path)
        return
        
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/scratch/lhl1g23/mamba/checkpoints/{timestamp if args.checkpoint_path is None else args.checkpoint_path}",
        filename="best",
        save_top_k=-1,
        save_last=True,
        monitor="val_acc"
    )
    
    version = "version_0"
    logger_tb = TensorBoardLogger(
        save_dir="/scratch/lhl1g23/mamba/lightning_logs/",
        name=args.checkpoint_path,
        version = version
    )
    
    model = MambaClassifierTrainer(cfg.mamba, 
                                   cfg.model,
                                   data=args.dataset,
                                   lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs, 
        accelerator="auto", 
        gradient_clip_val=1.0,
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback, RichProgressBar()],
        enable_progress_bar=True,
        logger=[logger_tb],
        deterministic=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader)

    
if __name__ == "__main__":
    print("Starting training...")
    main()