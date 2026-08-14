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