# Chebyshev QSP and QSVT

## Chebyshev QSP

Chebyshev QSP is the [original QSP formulation](https://doi.org/10.1103/PRXQuantum.2.040203), defined as follows

$$ e^{i\phi_0 Z} \tilde{x} e^{i\phi_1 Z} \tilde{x} \cdots \tilde{x} e^{i\phi_n Z} = \begin{pmatrix} P(x) & iQ(x) \sqrt{1 - x^2} \\ iQ^*(x) \sqrt{1 - x^2} & P^*(x) \end{pmatrix} $$

where the signal operator $\tilde{x}$ is

$$ \tilde{x} = \begin{pmatrix} x & i\sqrt{1 - x^2} \\ i\sqrt{1 - x^2} & x \end{pmatrix} $$

and $P(x)$ is a polynomial in $[-1, 1]$ with degree $n$ and parity $(n \bmod 2)$. Here the part we can control is $P_{re} = \mathrm{Re}(P(x))$, while $P_{im} = \mathrm{Im}(P(x))$ and $Q$ are chosen in such a way that $|P(x)|^2 + (1 - x^2)|Q(x)|^2 = 1$ for any choice of $x \in [-1, 1]$.

We call this *Chebyshev QSP* because $P(x)$ is best expressed as an expansion in terms of the [Chebyshev polynomials of the first kind](https://en.wikipedia.org/wiki/Chebyshev_polynomials) $T_k(x)$.

$$ P(x) = \sum_{k = 0}^n c_k T_k(x) $$


### Computing the phase factors

By a change of variables $x = \frac{z + z^{-1}}{2}$, we can rewrite $P$ as a Laurent polynomial

$$ P(z) = \sum_{k = 0}^n c_k \frac{z^k + z^{-k}}{2} $$

and reduce Chebyshev QSP to [Laurent XQSP](gqsp.md#xqsp). This is exactly what the [`ChebyshevQSPPhaseFactors.solve()`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.ChebyshevQSPPhaseFactors.solve) function does: it takes the list $\{ c_k \}_k$, either as a Python list or as a `ChebyshevTExpansion` object (see [here](../polynomials/chebyshev-expansions.md) for more info).


```python exec="on" source="block" result="text"
from nlft_qsp import *
import numpy as np

T = ChebyshevTExpansion([0.2, 0, 0.4, 0, 0.1])

qsp = ChebyshevQSPPhaseFactors.solve(T)

P, Q = qsp.polynomials(mode='laurent')

T_qsp = ChebyshevTExpansion.from_laurent_polynomial(P)

print(T(0.6))
print(T_qsp(0.6))
```

!!! note

    `ChebyshevQSPPhaseFactors.polynomials()` still returns `Polynomial` objects. In particular the Laurent polynomial $P$ in the code is in the form written above, and `ChebyshevTExpansion.from_laurent_polynomial()` essentially performs the change of variable $x = \frac{z + z^{-1}}{2}$.

!!! warning

    Make sure that $T$ as in the code above has definite parity. The solver will raise a `ValueError` if this is not the case.

### Solving for the phase factors directly from the polynomial

Most of the time we do not really have the Chebyshev expansion for our desired transformation, but we have the polynomial $P(x)$ in its usual monomial expansion. We can simply convert the `Polynomial` into a `ChebyshevTExpansion` as follows

```python exec="on" source="block" result="text"
from nlft_qsp import *
import numpy as np

P = Polynomial([0.2, 0, 0.4, 0, 0.1]) # 0.2 + 0.4 x^2 + 0.1 x^4

qsp = ChebyshevQSPPhaseFactors.solve(ChebyshevTExpansion.from_polynomial(P))

P_qsp, Q = qsp.polynomials(mode='laurent')

T_qsp = ChebyshevTExpansion.from_laurent_polynomial(P_qsp)

print(P(0.6))
print(T_qsp(0.6))
```

!!! note

    Keep in mind that `from_polynomial()` and `from_laurent_polynomial()` do fundamentally different things: the former computes the polynomial $P(x)$ which satisfies $P(x) = T(x)$, whereas the latter builds $P(z)$ such that $P(z) = T(\frac{z + z^{-1}}{2})$, effectively performing the change of variables mentioned in the [previous subsection](#computing-the-phase-factors).



## QSVT

`nlft-qsp` also supports the computation of phase factors for the [quantum singular value transformation](
https://doi.org/10.1145/3313276.3316366). The *reflection QSP* ansatz used by QSVT is pretty similar to Chebyshev QSP.

$$ e^{i\phi_0 Z} \tilde{r} e^{i\phi_1 Z} \tilde{r} \cdots \tilde{r} e^{i\phi_n Z} = \begin{pmatrix} P(x) & \cdot \\ \cdot & \cdot \end{pmatrix} $$

where this time the signal operator $\tilde{r}$ is a reflection:

$$ \tilde{r} = \begin{pmatrix} x & \sqrt{1 - x^2} \\ \sqrt{1 - x^2} & -x \end{pmatrix} $$

In order to solve for QSVT phase factors we can simply use `QSVTPhaseFactors.solve()` in the same way you would use `ChebyshevQSPPhaseFactors.solve()` for Chebyshev QSP:

```python exec="on" source="block" result="text"
from nlft_qsp import *
import numpy as np

P = Polynomial([0.2, 0, 0.4, 0, 0.1]) # 0.2 + 0.4 x^2 + 0.1 x^4

qsvt = QSVTPhaseFactors.solve(ChebyshevTExpansion.from_polynomial(P))

P_qsvt, Q = qsvt.polynomials(mode='laurent')

T_qsvt = ChebyshevTExpansion.from_laurent_polynomial(P_qsvt)

print(P(0.6))
print(T_qsvt(0.6))
```


### Converting between QSVT and Chebyshev QSP

The relationship between reflection and Chebyshev QSP is quite straightforward (see [here](https://doi.org/10.1103/PRXQuantum.2.040203), (A5)). `nlft-qsp` provides methods to easily change between these two variants:

- To convert Chebyshev QSP to QSVT, use the method [`QSVTPhaseFactors.from_chebqsp()`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.QSVTPhaseFactors.from_chebqsp);

- To convert QSVT to Chebyshev QSP, use the method [`QSVTPhaseFactors.to_chebqsp()`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.QSVTPhaseFactors.to_chebqsp).


## Using the Command Line Interface

The subcommand `qspx solve` can be used to produce protocols for given polynomials or Chebyshev expansions:

```bash
qspx solve -tqsvt P.qspx -o protocol.qspx
```

As for GQSP, `--type/-t` option is used to specify whether to produce a Chebyshev QSP protocol (`-tcheb`) or a QSVT protocol (`-tqsvt`).

Given a protocol in a file `protocol.qspx`, it is possible to compute the implementing polynomial as a Chebyshev expansion.

```bash
qspx make protocol.qspx
```