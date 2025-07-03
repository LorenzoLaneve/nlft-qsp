# Convolution optimization algorithm for polynomial completion (taken from github.com/Danimhn/GQSP-Code)

import torch
from torchaudio.transforms import FFTConvolve

from poly import Polynomial

import numerics as bd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONV_OPT_MAX_ATTEMPTS = 3

def complex_convolve(x):
    real_part = x.real
    imag_part = x.imag

    real_flip = torch.flip(real_part, dims=[0])
    imag_flip = torch.flip(-1*imag_part, dims=[0])

    conv_real_part = FFTConvolve("full").forward(real_part, real_flip)
    conv_imag_part = FFTConvolve("full").forward(imag_part, imag_flip)

    conv_real_imag = FFTConvolve("full").forward(real_part, imag_flip)
    conv_imag_real = FFTConvolve("full").forward(imag_part, real_flip)

    # Compute real and imaginary part of the convolution
    real_conv = conv_real_part - conv_imag_part
    imag_conv = conv_real_imag + conv_imag_real

    # Combine to form the complex result
    return torch.complex(real_conv, imag_conv)

def objective_torch(x, Pm1):
    x.requires_grad = True

    real_part = x[:len(x) // 2]
    imag_part = x[len(x) // 2:]

    Q = complex_convolve(torch.complex(real_part, imag_part))

    # Compute loss using squared distance function
    loss = torch.norm(Pm1 + Q)**2
    return loss

def complete(b: Polynomial) -> Polynomial:
    """Uses the convolution optimization algorithm (arXiv:2308.01501) to find a complementary polynomial to the given one.

    Args:
        b (Polynomial): The polynomial to complete.

    Note:
        Numerical stability is not guaranteed.

    Returns:
        Polynomial: A polynomial :math:`a(z)` satisfying :math:`|a|^2 + |b|^2 = 1` on the unit circle.
        In particular (a, b) will be in the image of the NLFT.
    """
    poly = torch.tensor(b.coeffs, dtype=torch.cdouble)

    conv_p_negative = complex_convolve(poly)
    conv_p_negative[poly.shape[0] - 1] -= 1

    # Initializing Q randomly to start with
    initial = torch.randn(poly.shape[0]*2, device=device, requires_grad=True)
    initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([initial], max_iter=1000)

    def closure():
        optimizer.zero_grad()
        loss = objective_torch(initial, conv_p_negative)
        loss.backward()
        return loss

    optimizer.step(closure)

    threshold = closure().item()
    attempts = 0
    while closure().item() > bd.machine_threshold():
        optimizer.step(closure)

        new_thr = closure().item()
        if threshold <= new_thr:
            attempts += 1
            if attempts >= CONV_OPT_MAX_ATTEMPTS:
                break
        else:
            threshold = new_thr
            attempts = 0

    opt_initial = initial.detach()
    real = opt_initial[:len(opt_initial) // 2]
    imag = opt_initial[len(opt_initial) // 2:]

    a = Polynomial(torch.complex(real, imag).tolist())
    a = a.shift(-b.effective_degree())
    a *= bd.exp(-1j*bd.arg(a[0])) # we want a[0] > 0
    return a