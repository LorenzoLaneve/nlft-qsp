# Norms

Functional norms are useful to quantify metrics such as distance between polynomials.
The `Polynomial` class provides mainly two norms that can be used.

## $L^2$ norm { #l2-norm }

The $L^2$ [norm] of a function is defined as

$$ \lVert P \rVert_{L^2}^2 := \frac{1}{2\pi} \int_0^{2\pi} | P(e^{i\theta}) |^2 \ d\theta = \sum_k |p_k|^2 $$

The package easily computes the $L^2$ norm using the rightmost expression, exposed by the methods `l2_squared_norm()` and `l2_norm()`.

[norm]: https://en.wikipedia.org/wiki/Lp_space#Lp_spaces_and_Lebesgue_integrals

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([1, 2, 3])
print(P.l2_squared_norm())
print(P.l2_norm())
```


## Supremum norm { #sup-norm }

We define the supremum norm, also known as the $L^{\infty}$ [norm], as follows
$$ \lVert P \rVert_{\infty} := \sup_{|z| = 1} |P(z)| $$

Unlike the $L^2$ norm, we cannot compute the supremum norm exactly. The `sup_norm()` method takes an argument $N$
and computes the maximum absolute value over the polynomial computed in the $N$-th roots of unity $e^{2\pi i k/N}$.
We recommend $N$ to be at least twice the highest frequency appearing in the polynomial.

[norm]: https://en.wikipedia.org/wiki/Uniform_norm