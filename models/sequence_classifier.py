import argparse
import datetime
import os
import shutil

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from timm.loss import LabelSmoothingCrossEntropy
from timm.models.layers import DropPath, trunc_normal_
from torchmetrics.classification import MulticlassF1Score

# from dataloader_UCR import get_datasets
from dataloader_UEA import get_datasets
from utils import get_clf_report, save_copy_of_files, str2bool, random_masking_3D

from mamba_simple import Mamba
import torch.nn.functional as F

class PatchEmbed(L.LightningModule):
    def __init__(self, seq_len, patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        num_patches = int((seq_len - patch_size) / stride + 1)
        self.num_patches = num_patches
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

# --- AFB ----

def complex_relu(x):
    real = F.relu(x.real)
    imag = F.relu(x.imag)
    return torch.complex(real, imag)

class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_1 = nn.Parameter(torch.randn(dim, 2) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(dim, 2) * 0.02)
        trunc_normal_(self.complex_weight_1, std=.02)
        trunc_normal_(self.complex_weight_2, std=.02)

    def forward(self, x_fft):
        weight_1 = torch.view_as_complex(self.complex_weight_1)
        weight_2 = torch.view_as_complex(self.complex_weight_2)
        x_weighted = x_fft * weight_1
        x_weighted = complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted

class Adaptive_Fourier_Filter_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)
        self.threshold_param = nn.Parameter(torch.rand(1)*0.5)
        self.low_pass_cut_freq_param = nn.Parameter(torch.tensor(float(dim // 2)) - torch.rand(1)*0.5)
        self.high_pass_cut_freq_param = nn.Parameter(torch.tensor(float(dim // 4)) - torch.rand(1)*0.5)

    def adaptive_freq_pass(self, x_fft, flag="high"):
        # x_fft: [B, N_freq, D]
        B, N_freq, D = x_fft.shape
        # N_freq mask
        W = (N_freq - 1) * 2     # orginal sequence length, N_freq = W//2+1
        freqs = torch.fft.rfftfreq(W, d=1.0 / W).to(x_fft.device)    # [N_freq]

        # mask shape [1, N_freq, 1] & x_fft [B, N_freq, D] 
        if flag == "high":
            mask = (torch.abs(freqs) >= self.high_pass_cut_freq_param.to(x_fft.device)).float()
        else:
            mask = (torch.abs(freqs) <= self.low_pass_cut_freq_param.to(x_fft.device)).float()
        mask = mask.view(1, N_freq, 1)
        return x_fft * mask

    def forward(self, x_in):
        B, N, D = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, N_freq, D]
        if args.adaptive_filter:
            x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")  # [B, N_freq, D]
            x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")
        else:
            x_low_pass = 0
            x_high_pass = 0
        x_weighted = (
            self.learnable_filter_layer_1(x_fft) +
            self.learnable_filter_layer_2(x_low_pass) +
            self.learnable_filter_layer_3(x_high_pass)
        )
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, D)
        return x


class IDMamba_Block(L.LightningModule):
    def __init__(self, configs, dim, drop=0.):
        super().__init__()
        self.configs = configs
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.conv = nn.Conv1d(dim, dim, 1)
        self.mamba = Mamba(dim, d_state=self.configs.d_state, d_conv_1=self.configs.d_conv_1, d_conv_2=self.configs.d_conv_2, expand=self.configs.e_fact)
        
    def forward(self, x):
        x_act = self.act(x)
        x1, x2 = self.mamba(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2 * x_act
        out2 = x2 * x1_2 * x_act
        x = out1 + out2
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

class FAIM_layer(L.LightningModule):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, configs=None):
        super().__init__()
        self.configs = configs
        self.AFB = Adaptive_Fourier_Filter_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.idmamba = IDMamba_Block(configs, dim, drop=drop)
    def forward(self, x_in):
        # AFB+Mamba
        x = self.norm1(x_in)
        x = self.AFB(x)
        x = self.norm2(x)
        x = self.idmamba(x)
        x = x_in + self.drop_path(x)
        return x

# --- FAIM ---
class FAIM(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed(
            seq_len=args.seq_len, patch_size=args.patch_size,
            in_chans=args.num_channels, embed_dim=args.emb_dim
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.emb_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=args.dropout_rate)
        dpr = [x.item() for x in torch.linspace(0, args.dropout_rate, args.depth)]

        self.FAIM_blocks = nn.ModuleList([
            FAIM_layer(dim=args.emb_dim, drop=args.dropout_rate, drop_path=dpr[i], configs=args)
            for i in range(args.depth)
        ])
        self.head = nn.Linear(args.emb_dim, args.num_classes)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def pretrain(self, x_in):
        x = self.patch_embed(x_in)
        x = x + self.pos_embed
        x_patched = self.pos_drop(x)
        x_masked, _, self.mask, _ = random_masking_3D(x, mask_ratio=args.masking_ratio)
        self.mask = self.mask.bool()  # mask: [bs x num_patch x n_vars]
        for block in self.FAIM_blocks:
            x_masked = block(x_masked)
        return x_masked, x_patched

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.FAIM_blocks:
            x = block(x)
        x = x.mean(1)   # Pooling
        x = self.head(x)
        return x

# --- PyTorch Lightning ---
class model_pretraining(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = FAIM()
    def forward(self, x): return self.model(x)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.pretrain_lr, weight_decay=1e-4)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        preds, target = self.model.pretrain(data)
        loss = (preds - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * self.model.mask).sum() / self.model.mask.sum()
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def training_step(self, batch, batch_idx): return self._calculate_loss(batch, mode="train")
    def test_step(self, batch, batch_idx): self._calculate_loss(batch, mode="test")

class model_training(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = FAIM()
        self.f1 = MulticlassF1Score(num_classes=args.num_classes)
        self.criterion = LabelSmoothingCrossEntropy()
    def forward(self, x):
        return self.model(x)
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=args.train_lr, weight_decay=1e-4)
        return optimizer
    def _calculate_loss(self, batch, mode="train"):
        data = batch[0]
        labels = batch[1].to(torch.int64)
        preds = self.model(data)
        loss = self.criterion(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        f1 = self.f1(preds, labels)
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"{mode}_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def training_step(self, batch, batch_idx): return self._calculate_loss(batch, mode="train")
    def test_step(self, batch, batch_idx): self._calculate_loss(batch, mode="test")

# --- Train ---
def pretrain_model():
    PRETRAIN_MAX_EPOCHS = args.pretrain_epochs
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        max_epochs=PRETRAIN_MAX_EPOCHS,
        callbacks=[
            pretrain_checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(42)
    model = model_pretraining()
    trainer.fit(model, train_loader)
    return pretrain_checkpoint_callback.best_model_path

def train_model(pretrained_model_path):
    trainer = L.Trainer(
        default_root_dir=CHECKPOINT_PATH,
        accelerator="auto",
        devices=1,
        max_epochs=args.num_epochs,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor("epoch"),
            TQDMProgressBar(refresh_rate=500)
        ],
    )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    L.seed_everything(42)
    if args.load_from_pretrained:
        model = model_training.load_from_checkpoint(pretrained_model_path)
    else:
        model = model_training()
    trainer.fit(model, train_loader)
    model = model_training.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    acc_result = {"test": test_result[0]["test_acc"]}
    f1_result = {"test": test_result[0]["test_f1"]}
    get_clf_report(model, test_loader, CHECKPOINT_PATH, args.class_names)
    return model, acc_result, f1_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='....')
    parser.add_argument('--dataset_root', type=str, default='...', help='UCR/UEA root')
    parser.add_argument('--dataset_name', type=str, default='ArticularyWordRecognition', help='dataset')
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--load_from_pretrained', type=str2bool, default=True, help='False: without pretraining')
    parser.add_argument('--AFB', type=str2bool, default=True)
    parser.add_argument('--Mamba', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_conv_1', type=int, default=2)
    parser.add_argument('--d_conv_2', type=int, default=4)
    parser.add_argument('--e_fact', type=int, default=1)

    args = parser.parse_args()

    DATASET_PATH = os.path.join(args.dataset_root, args.dataset_name)
    print(DATASET_PATH)
    run_description = f"{os.path.basename(args.dataset_name)}_dim{args.emb_dim}_depth{args.depth}__" \
                      f"AFB_{args.AFB}__Mamba_{args.Mamba}_preTr_{args.load_from_pretrained}_" 
    
    print(f"========== {run_description} ===========")

    CHECKPOINT_PATH = f"lightning_logs/{run_description}"
    pretrain_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        filename='pretrain-{epoch}',
        monitor='train_loss',
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH,
        save_top_k=1,
        monitor='train_loss',
        mode='min'
    )
    save_copy_of_files(pretrain_checkpoint_callback)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loaders = get_datasets(DATASET_PATH, args)
    if len(loaders) == 3:
        train_loader, _, test_loader = loaders  
    elif len(loaders) == 2:
        train_loader, test_loader = loaders
    else:
        raise ValueError("DataLoader is not correct!!!")
    print("Dataset loaded ...")
    args.num_classes = len(np.unique(train_loader.dataset.y_data))
    args.class_names = [str(i) for i in range(args.num_classes)]
    args.seq_len = train_loader.dataset.x_data.shape[-1]
    args.num_channels = train_loader.dataset.x_data.shape[1]

    if args.load_from_pretrained:
        best_model_path = pretrain_model()
    else:
        best_model_path = ''
    model, acc_results, f1_results = train_model(best_model_path)
    print("ACC results", acc_results)
    print("F1  results", f1_results)

    text_save_dir = ".."
    os.makedirs(text_save_dir, exist_ok=True)
    f = open(f"{text_save_dir}/{args.model_id}.txt", 'a')
    f.write(run_description + "  \n")
    f.write(f"FAIM_{os.path.basename(args.dataset_name)}_l_{args.depth}" + "  \n")
    f.write('acc:{}, mf1:{}'.format(acc_results, f1_results))
    f.write('\n\n')
    f.close()