import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analyse.utils.transforms import frequency_response_one_step, z_transform_one_step

def plot_eigen(M, ax, title="Complex plane", s=1, circle=True):
    if circle:
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle=':')
    ax.scatter(M.real, M.imag, s=s)
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(title)
    ax.set_aspect('equal')

def plot_frequency_repsponse_one_step(dA, dB, C, b, d, t, ax):
    H = torch.abs(frequency_response_one_step(dA, dB, C, t))
    frequencies = torch.linspace(0, 2*torch.pi, 360) / (2 * torch.pi)* 16000

    ax.plot(frequencies, H[b,d])
    ax.set_title(f"Frequency response at D = {d}, t = {t}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_box_aspect(1)
    
def plot_z_transform_one_step(dA, dB, C, b, d, t, ax, circle=True, log=True):
    if circle:
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle=':')
    
    H = torch.abs(z_transform_one_step(dA, dB, C, t))
    if log: 
        H = torch.log(H)
        ax.set_title(f"Logarithmic Z-transform at D = {d}, t = {t}")
    else:
        ax.set_title(f"Z-transform at D = {d}, t = {t}")
        
    ax.set_ylabel("Imag")
    ax.set_xlabel("Real")
    im = ax.imshow(H[b,d], extent=[-1.2, 1.2, -1.2, 1.2])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Value")