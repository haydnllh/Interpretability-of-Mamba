import torch

def jacobian_dy_dt(params, X_mamba, b_dim, d_dim, conj_sym=True):
    A = params['A']
    B = params['B']
    C = params['C']
    dA = params['dA']
    states = params['states']
    
    dA_bd = dA[b_dim, :, d_dim, :]
    A_d = A[d_dim]
    B_b = B[b_dim]
    C_b = C[b_dim]
    h_d  = states[b_dim, :, d_dim, :]
    X_bd = X_mamba[b_dim, :, d_dim]

    h_prev_all = torch.cat([torch.zeros_like(h_d[:1]), h_d[:-1]], dim=0)

    g = A_d * dA_bd * h_prev_all + B_b * X_bd.unsqueeze(-1)
    
    lambda_cp = torch.cumprod(dA_bd, dim=0, dtype=torch.float64)
    sign = torch.sign(lambda_cp)
    sign[sign == 0] = 1
    lambda_cp = sign * lambda_cp.abs().clamp(min=1e-60)
            
    u = C_b * lambda_cp
    v = g / lambda_cp

    jacobian = u @ v.mT
    jacobian = torch.tril(jacobian)
    
    if conj_sym:
        jacobian *= 2
    return jacobian.real