# Computing the phase factors for a given Python function

```python exec="on" session="plt"
import io, matplotlib.pyplot as plt

def _svg_show(*a, **k):
    buf = io.StringIO()
    plt.savefig(buf, format="svg", bbox_inches="tight")
    plt.close()
    print("<div style=\"text-align: center;\">")
    print(buf.getvalue())
    print("</div>")

plt.show = _svg_show
```

While it is possible to specify polynomials, it is also possible to directly approximate desired functions specified as sets of points or Python functions.


## Direct implementation of Python functions

It is possible to use [`ChebyshevQSPPhaseFactors.approximate`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.ChebyshevQSPPhaseFactors.approximate) and [`QSVTPhaseFactors.approximate`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.QSVTPhaseFactors.approximate) to this for Chebyshev QSP and QSVT respectively. They take the Python function as well as the degree `deg` of the approximating polynomial to compute.

```python exec="on" session="plt" html="true" source="block"
from nlft_qsp import *
import numpy as np

def f(x):
    return 0.7 * np.exp(-3 * np.cos(5*x) ** 2)

qsp = ChebyshevQSPPhaseFactors.approximate(f, deg=28)

P, Q = qsp.polynomials(mode='laurent')

plot_chebyshev({
    "Original": f,
    "QSP": ChebyshevTExpansion.from_laurent_polynomial(P)
})
```

!!! note

    For Chebyshev QSP and QSVT remember to take `deg` to be even or odd depending on the parity of the desired function.


### Computing Chebyshev and Fourier approximations

`nlft-qsp` also exposes some useful functions to directly approximate given Python functions with `Polynomial` or `ChebyshevTExpansion` objects so that they can be further manipulated before being fed into a QSP solver.

Here is how to compute the Chebyshev expansion of the given function $f(x) : [-1, 1] \rightarrow \mathbb{C}$ using [`chebyshev_approximate`](../reference/nlft_qsp/approximate.md#nlft_qsp.approximate.chebyshev_approximate)

```python exec="on" session="plt" html="true" source="block"
from nlft_qsp import *
import numpy as np

def f(x):
    return 0.8 * np.arctan(20 * x)

T = chebyshev_approximate(f, 15)

plot_chebyshev({
    "Original": f,
    "Approx.": T
})
```

And the following snippet shows how to compute the Fourier expansion (as a Laurent polynomial) of a function $f(z)$ where $z = e^{i\theta}$ using [`fourier_approximate`](../reference/nlft_qsp/approximate.md#nlft_qsp.approximate.fourier_approximate)

```python exec="on" session="plt" html="true" source="block"
from nlft_qsp import *
import numpy as np

def f(z):
    theta = np.angle(z)
    if 0 < theta < np.pi/2:
        return 1
    return 0.2

P = fourier_approximate(f, 14)

plot_fourier({
    "Original": f,
    "Approx.": P
})
```


## Using the Command Line Interface

`qspx` has an `approximate` subcommand which works similarly to `qspx solve`, except that a Python function and an approximation degree must be specified instead of a file containing a polynomial.

The following command approximates the function $f(x) = 0.7 \cos(10x)$ with a polynomial of degree 14 and computes a QSVT protocol.
```bash
qspx approx -tqsvt '0.7*np.cos(10*x)' -d 14
```

It is also possible to directly specify the desired polynomials. For example, the command below computes the analytic Generalized QSP protocol for the polynomial $f(z) = \frac{1}{4} + \frac{i}{4} z - \frac{1}{3} z^2$:
```bash
qspx approx -tag '1/4 + 1j/4 * z - 1/3 * z ** 2' -d 2
```

!!! note
    Python expressions can use `np` and `sp` to access numpy and scipy functions, respectively.


!!! note
    For QSVT (`-tqsvt`) and Chebyshev QSP (`-tcheb`), `x` will be the free variable $x \in [-1, 1]$ for the expression.
    
    For all the other variants, `z` is the free variable $|z| = 1$. It is possible to also use `t` for the phase $t \in [-\pi, \pi]$ such that $z = e^{it}$.


### Saving approximations

It is also possible to print or save the polynomial approximations without synthesizing a QSP protocol for it. Pass `--poly-only/-p` to approximate on the unit circle or `--cheb-only/-c` to approximate on $[-1, 1]$.

The command below approximates $f(x) = \mathrm{erf}(30 x)$ with a Chebyshev expansion of degree 15 and saves it to `apx.qspx`.
```bash
qspx approx -c 'sp.special.erf(30*x)' -d 15 -o apx.qspx
```