# Computing complementary polynomials

An important step in the synthesis of a QSP protocol for a desired polynomial transformation $P(z)$ is the **completion problem**: compute a *complementary polynomial*, i.e., a polynomial $Q(z)$ such that $|P(z)|^2 + |Q(z)|^2 = 1$ whenever $|z| = 1$.


## Using Weiss' algorithm <small>recommended</small> { #weiss data-toc-label="Weiss' algorithm" }

`nlft-qsp` provides the [`weiss`](../reference/nlft_qsp/solvers/completion/weiss.md) module which takes care of this.


```python exec="on" source="block" result="text"
from nlft_qsp import *
from nlft_qsp.solvers import weiss

P = Polynomial([1/3, 1/5, 1/7])
Q = weiss.complete(P)

print("P(z) =", P)
print("Q(z) =", Q)

print("|P|^2 + |Q|^2 = ", P * P.conjugate() + Q * Q.conjugate())
```

!!! note

    The $Q$ computed above actually has non-positive frequencies. This because `weiss.complete()` actually returns a complementary polynomial following the convention for the nonlinear Fourier transform (see the definition [here](https://arxiv.org/abs/2503.03026)).

    It is possible to obtain a normal polynomial with non-negative frequencies by using [`Polynomial.shift()`](../reference/nlft_qsp/poly.md#nlft_qsp.poly.Polynomial.shift).


### Computing the ratio $P/Q$ { #weiss data-toc-label="Computing the ratio" }

First approaches based on nonlinear Fourier analysis (namely [`riemann_hilbert`](../reference/nlft_qsp/solvers/riemann_hilbert.md) and [`half_cholesky`](../reference/nlft_qsp/solvers/half_cholesky.md)) do not need the complementary polynomial $Q$, but the Laurent expansion of the ratio $P/Q$. For this reason, the `weiss` module also conveniently provides a `ratio()` method for this. Note that it also returns $Q$ itself.


```python exec="on" source="block" result="text"
from nlft_qsp import *
from nlft_qsp.solvers import weiss

P = Polynomial([1/3, 1/5, 1/7])

Q, R = weiss.ratio(P)

print("P(1)/Q(1) =", P(1)/Q(1))
print("R(1) =", R(1))
```


## Using Prony's method { #weiss data-toc-label="Prony's method" }

The package also provides the [`prony`](../reference/nlft_qsp/solvers/completion/prony.md) module (taken and adapted from [this repo](https://github.com/quantum-programming/gqsp-angle-finding)), which completes polynomials based on [Prony's method](https://quantum-journal.org/papers/q-2022-10-20-842/).

```python exec="on" source="block" result="text"
from nlft_qsp import *
from nlft_qsp.solvers import prony

P = Polynomial([1/3, 1/5, 1/7])
Q = prony.complete(P)

print("P(z) =", P)
print("Q(z) =", Q)

print("|P|^2 + |Q|^2 = ", P * P.conjugate() + Q * Q.conjugate())
```

!!! warning

    This is provided only for comparison purposes and **should not be used**, as it does not guarantee numerical stability for large polynomial degrees.