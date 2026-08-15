
The package provides two helper functions to plot polynomials and any callable object, either in $[-1, 1]$ or along the unit circle.

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


```python exec="on" session="plt" html="true" source="block"
from nlft_qsp import *

P = Polynomial([0.1, 0, -0.2, 0, 0.4]) # 0.1 - 0.2 x^2 + 0.4 x^4

def f(x):
    return 0.2 - 0.3 * x ** 2

plot_chebyshev({
    "Pretty polynomial": P,
    "Pretty function": f
})
```

!!! note
    `plot_chebyshev` implicitly takes the real part of the functions.


---


```python exec="on" session="plt" html="true" source="block"
from nlft_qsp import *
import numpy as np

P = Polynomial([0.7, 0.1, 0.5, 0.4])

def f(z):
    theta = np.angle(z)
    
    if 0 < theta and theta < np.pi/2:
        return 0.8

    if -np.pi < theta and theta < -np.pi/2:
        return 0.8

    return 0.1

plot_fourier({
    "Poly": P,
    "Square wave": f
})
```

!!! note
    Functions plotted by `plot_fourier` are meant to be of a complex variable $z = e^{i\theta}$. Use `np.angle` as above to work with the variable $\theta$.

!!! note
    `plot_fourier` implicitly takes the absolute value of the functions.




## Plotting using the Command Line Interface

The subcommand `qspx plot` uses the two above functions to plot any function specified as input, either as a file or as a Python expression. If `P` is saved in a file `P.qspx`, the first snippet above is equivalent to the following call:

```bash
qspx plot P.qspx -f '0.2 - 0.3 * x ** 2'
```

Note that the `-f` flag should be put in front of arguments that are intended to be Python expressions. This call above will use `plot_chebyshev` and plot the given functions in $[-1, 1]$. Use the `-u` flag to plot along the unit circle.

```bash
qspx plot -u P.qspx -f '1 + 0.5 * z ** 2'
```

!!! warning
    When plotting $[-1, 1]$, the Python expression are with respect to the variable `x`, whereas functions to be plotted on the unit circle will be expected to be written with respect to variable `z`.