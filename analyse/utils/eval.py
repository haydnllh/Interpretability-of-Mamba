import torch
from tqdm import tqdm

def accuracy(loader, model, device='cuda'):
    model.eval()
    
    n_samples, n_correct = 0, 0
    
    with torch.no_grad():
        for X, y in tqdm(loader):
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            
            n_correct += (torch.argmax(pred, dim=1) == y).sum()
            n_samples += X.shape[0]
            
    acc = n_correct / n_samples
    return acc