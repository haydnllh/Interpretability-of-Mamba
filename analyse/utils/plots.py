import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from analyse.utils.transforms import frequency_response_one_step, z_transform_one_step
from analyse.utils.jacobian import jacobian_dy_dt

def plot_eigen(
    M,
    ax,
    title="Complex Plane",
    s=20,
    circle=True,
    cmap="viridis",
    alpha=0.85,
    edgecolor="black",
    linewidth=0.3,
    grid=True
):
    real = M.real.detach().cpu().numpy() if hasattr(M.real, "detach") else np.array(M.real)
    imag = M.imag.detach().cpu().numpy() if hasattr(M.imag, "detach") else np.array(M.imag)

    if circle:
        theta = np.linspace(0, 2 * np.pi, 500)
        ax.plot(
            np.cos(theta),
            np.sin(theta),
            linestyle="--",
            linewidth=1.2,
            color="gray",
            alpha=0.7,
            label="Unit Circle"
        )

    colors = np.sqrt(real**2 + imag**2)
    scatter = ax.scatter(
        real,
        imag,
        c=colors,
        cmap=cmap,
        s=s,
        alpha=alpha,
        edgecolors=edgecolor,
        linewidths=linewidth
    )

    ax.set_xlabel("Real", fontsize=11)
    ax.set_ylabel("Imaginary", fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold")

    ax.set_aspect("equal", adjustable="box")

    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    for spine in ax.spines.values():
        spine.set_alpha(0.6)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Magnitude", fontsize=10)

    if circle:
        ax.legend(frameon=False, fontsize=9)

    ax.figure.tight_layout()
    
def plot_real_eigen(x, title="Real eigenvalues", s=1, alpha=0.9, grid=True):
    t = np.arange(len(x))

    fig, ax = plt.subplots()

    ax.scatter(t, x, marker='o', linewidth=1.5,
            alpha=alpha, label="values", s=s)

    ax.axhline(y=1, linestyle='--', linewidth=1.2,
               color="gray", alpha=0.8, label="y = 1")

    ax.set_xlabel("Index", fontsize=11)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold")

    ax.set_ylim(0, max(1.1, np.max(x) + 0.1))

    if grid:
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_alpha(0.6)

    ax.legend(frameon=False, fontsize=9, loc="lower left")

    plt.tight_layout()
    plt.show()

def plot_frequency_repsponse_one_step(dA, dB, C, b, d, t, ax):
    H = torch.abs(frequency_response_one_step(dA, dB, C, t)).to('cpu')
    frequencies = torch.linspace(0, 2*torch.pi, 360) / (2 * torch.pi)* 16000

    ax.plot(frequencies, H[b,d])
    ax.set_title(f"Frequency response at D = {d}, t = {t}", fontsize=13, weight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    
def plot_z_transform_one_step(dA, dB, C, b, d, t, ax, circle=True, log=True):
    if circle:
        theta = np.linspace(0, 2*np.pi, 400)
        ax.plot(np.cos(theta), np.sin(theta), linestyle=':')
    
    H = torch.abs(z_transform_one_step(dA, dB, C, t)).to('cpu')
    if log: 
        H = torch.log(H)
        ax.set_title(f"Logarithmic Z-transform at D = {d}, t = {t}", fontsize=13, weight="bold")
    else:
        ax.set_title(f"Z-transform at D = {d}, t = {t}", fontsize=13, weight="bold")
        
    ax.set_ylabel("Imag")
    ax.set_xlabel("Real")
    im = ax.imshow(H[b,d], extent=[-1.2, 1.2, -1.2, 1.2])
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label="Value")
    
def plot_jacobian_dy_dt(params, X_mamba, b_dim, d_dim, ax, log=False, cbar=True):
    J = jacobian_dy_dt(params, X_mamba, b_dim, d_dim)
    J = torch.sign(J) * torch.log(abs(J) + 1) if log else J
    
    sns.heatmap(
        J.detach().cpu().numpy(),
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={'label': 'Jacobian value'},
        ax=ax,
        xticklabels=1600,
        yticklabels=1600,
        cbar=cbar
    )
    
    if log:
        ax.set_title(fr"Logarithmic Jacobian Heatmap (b={b_dim}, d={d_dim}). "
            r"A point is $\log|\partial y_t / \partial \Delta_{t'}|$")
    else:
        ax.set_title(fr"Jacobian Heatmap (b={b_dim}, d={d_dim}). "
            r"A point is $\partial y_t / \partial \Delta_{t'}$")
    ax.set_xlabel(r"$\partial \Delta_{t'}$ (Right t' = 0)")
    ax.set_ylabel(r"$\partial y_t$ (Top t = 0)")

def plot_variance(M, ax, b_dim, window=100):
    M = M[b_dim]

    with torch.no_grad():
        L = M.shape[0]

        M = M.reshape(L, -1)

        cumsum = torch.cat([
            torch.zeros(1, M.shape[1], device=M.device, dtype=M.dtype),
            torch.cumsum(M, dim=0)
        ], dim=0)

        cumsum_sq = torch.cat([
            torch.zeros(1, M.shape[1], device=M.device, dtype=M.dtype),
            torch.cumsum(M ** 2, dim=0)
        ], dim=0)

        sum_w = cumsum[window:] - cumsum[:-window]
        sum_sq_w = cumsum_sq[window:] - cumsum_sq[:-window]

        mean_w = sum_w / window
        var_w = (sum_sq_w / window) - mean_w ** 2

        variances = var_w.mean(dim=1)

        pad = torch.full(
            (window - 1,),
            float("nan"),
            device=variances.device,
            dtype=variances.dtype
        )

        variances = torch.cat([variances, pad]).cpu().numpy()

    x = np.arange(len(variances))

    kernel = np.ones(5) / 5
    smooth = np.convolve(np.nan_to_num(variances), kernel, mode="same")

    ax.plot(x, variances, alpha=0.3, linewidth=1)
    ax.plot(x, smooth, linewidth=2)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Variance")
    ax.set_title("Eigenvalue Variance Over Time")
    ax.grid(alpha=0.2)