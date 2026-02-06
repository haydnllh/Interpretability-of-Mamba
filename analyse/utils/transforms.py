import torch
import numpy as np

def frequency_response_one_step(dA, dB, C, t):
    frequencies = torch.arange(360) * (2 * torch.pi / 360)
    z = torch.exp(1j * frequencies).unsqueeze(0)

    dA = dA.unsqueeze(-1)
    dB = dB.unsqueeze(-1)
    C  = C.unsqueeze(2).unsqueeze(-1)
    
    H = (C[:,t] / (z - dA[:,t]) * dB[:,t]).sum(dim=2)
    return H

def z_transform_one_step(dA, dB, C, t):
    nx, ny = 120, 120
    real = np.linspace(-1.18, 1.2, nx)
    imag = np.linspace(1.18, -1.2, ny)
    xv, xy = np.meshgrid(real, imag)
    z = torch.tensor(xv + 1j * xy)
    dA_t = dA.unsqueeze(-1).unsqueeze(-1)
    dB_t = dB.unsqueeze(-1).unsqueeze(-1)
    C_t  = C.unsqueeze(2).unsqueeze(-1).unsqueeze(-1)
    t = 0

    H = (C_t[:,t] / (z - dA_t[:,t]) * dB_t[:,t]).sum(dim=2)
    return H