import torch
from torchsummary import summary
from mamba_ssm import Mamba2

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=4,    # Block expansion factor
).to("cuda")
y = model(x)
assert y.shape == x.shape
