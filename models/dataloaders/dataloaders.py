import torch
from torchvision import datasets, transforms
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import soundfile as sf
import os
from models.dataloaders.induction_heads import InductionData
from models.dataloaders.s4_sc import _SpeechCommands
from models.dataloaders.wikitext import get_dataloaders
from einops import rearrange

def get_mnist(batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1).unsqueeze(-1))
    ])
    
    train_data = datasets.MNIST("/scratch/lhl1g23/mamba/data/mnist/train", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("/scratch/lhl1g23/mamba/data/mnist/test", train=False, download=True, transform=transform)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=64), \
        DataLoader(test_data, batch_size=batch_size, num_workers=64)

def get_cifar(batch_size=32):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    train_data = datasets.CIFAR10("/scratch/lhl1g23/mamba/data/cifar/train", train=True, download=True, transform=train_transform)
    test_data = datasets.CIFAR10("/scratch/lhl1g23/mamba/data/cifar/test", train=False, download=True, transform=test_transform)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=64), \
        DataLoader(test_data, batch_size=batch_size, num_workers=64)
    
def get_sc(batch_size=32):
    train_data = _SpeechCommands(
        partition='train',
        length=16000,
        mfcc=False,
        sr=1,
        dropped_rate=0,
        path='/scratch/lhl1g23/mamba/data/sc/train',
        all_classes=False,
        gen=False,
        discrete_input=False,
    )
    
    val_data = _SpeechCommands(
        partition='val',
        length=16000,
        mfcc=False,
        sr=1,
        dropped_rate=0,
        path='/scratch/lhl1g23/mamba/data/sc/train',
        all_classes=False,
        gen=False,
        discrete_input=False
    ) 

    
    test_data = _SpeechCommands(
        partition='test',
        length=16000,
        mfcc=False,
        sr=1,
        dropped_rate=0,
        path='/scratch/lhl1g23/mamba/data/sc/train',
        all_classes=False,
        gen=False,
        discrete_input=False
    ) 

    return DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4), \
        DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4), \
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
def get_wikitext(
    dataset_name="wikitext-103-raw-v1",
    tokenizer="gpt2", 
    batch_size=32, 
    block_size=256
):
    return get_dataloaders(dataset_name=dataset_name, tokenizer=tokenizer, batch_size=batch_size, block_size=block_size)