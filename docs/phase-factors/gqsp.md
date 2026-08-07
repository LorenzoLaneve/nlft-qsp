# Generalized quantum signal processing

In generalized quantum signal processing (QSP) we have the following construction:

$$ e^{i\lambda Z} e^{i\phi_0 X} e^{i\theta_0 Z} \tilde{w} e^{i\phi_1 X} e^{i\theta_1 Z} \tilde{w} \cdots \tilde{w} e^{i\phi_n X} e^{i\theta_n Z} = \begin{pmatrix} P(z) & Q(z) \\ \cdot & \cdot \end{pmatrix} $$

where $\tilde{w} = \mathrm{diag}(z, 1)$ is the signal operator and $X, Y, Z$ are the [Pauli matrices](https://en.wikipedia.org/wiki/Pauli_matrices). Any pair $(P, Q)$ satisfying $|P(z)|^2 + |Q(z)|^2 = 1$ for $|z| = 1$ admits such a decomposition, and in particular any desired $P$ with $|P(z)| \le 1$ in this domain can be constructed.

!!! note

    This is slightly different from the original ansatz by [Motlagh, Wiebe](https://doi.org/10.1103/PRXQuantum.5.020368)! See [the subsection below](#mw-gqsp) to see how to convert phase factors to their convention.



## Computing the generated polynomials { #computing-generated-polynomials }

The `GQSPPhaseFactors` class provides a variety of methods to handle the GQSP ansatz. In particular, it is possible, given a set of phase factors $\lambda, \{ \phi_k \}, \{ \theta_k \}$ to generate the corresponding polynomials $P(z), Q(z)$.

```python exec="on" source="block" result="text"
from nlft_qsp import *


qsp = GQSPPhaseFactors(phi=[0.1, 0.7, 0.2], lbd=0.2, theta=[0.5, 0.1, 0.3])

P, Q = qsp.polynomials()

print(P)
print(Q)
```

Note that the degrees of $P, Q$ is 2 since $\vec{\phi}$ and $\vec{\theta}$ have length 3.


### Laurent and analytic quantum signal processing

It is possible to pass `mode='laurent'` to the `polynomials()` method to use the signal operator $\tilde{v} = \mathrm{diag}(z, z^{-1})$ instead of $\tilde{w} = \mathrm{diag}(z, 1)$ in the QSP construction. This has just the effect to distribute the coefficients of $P, Q$ in $\{ -n, -n+2, \ldots, n-2, n \}$ instead of $\{ 0, 1, \ldots, n \}$ (see the difference between analytic and Laurent QSP [here](https://arxiv.org/abs/2503.03026)).

```python exec="on" source="block" result="text"
from nlft_qsp import *

qsp = GQSPPhaseFactors(phi=[0.1, 0.7, 0.2], lbd=0.2, theta=[0.5, 0.1, 0.3])

P, Q = qsp.polynomials()

print(P)
print(Q)

P, Q = qsp.polynomials(mode='laurent')

print(P)
print(Q)
```


## Using the GQSP solver

Computing the phase factors for a desired `Polynomial` $P(z)$ can be done using the [`gqsp_solve()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.gqsp_solve) function. Note that $P(z)$ is assumed to be a polynomial with non-negative frequencies.

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([1/3, 1/7, 1/5])

qsp = gqsp_solve(P)

P2, _ = qsp.polynomials()

print(P)
print(P2)
```

!!! warning

    Make sure that $|P(z)| < 1 - \eta$ on the unit circle, for some sufficiently large gap $\eta > 0$. You can estimate this with the [`Polynomial.sup_norm()`](../../reference/nlft_qsp/poly/#nlft_qsp.poly.Polynomial.sup_norm) method. The complexity and numerical stability degrades as $\eta \rightarrow 0$, and the package might raise a `WeissConvergenceError` as it would not be able to find a complementary polynomial to $P$. This is an instrinsic instability of the problem, not a limitation of the package. It is possible in principle to partially overcome this problem by increasing the floating point precision, e.g., setting the underlying `complex_type` to `numpy.complex256`, but this is not supported by all architectures.



## XQSP and YQSP { #xqsp }

XQSP and YQSP are defined respectively in the following ways:

$$ e^{i\phi_0 X} \tilde{w} e^{i\phi_1 X} \tilde{w} \cdots \tilde{w} e^{i\phi_n X} = \begin{pmatrix} P(z) & Q(z) \\ \cdot & \cdot \end{pmatrix} $$

$$ e^{i\phi_0 Y} \tilde{w} e^{i\phi_1 Y} \tilde{w} \cdots \tilde{w} e^{i\phi_n Y} = \begin{pmatrix} P(z) & Q(z) \\ \cdot & \cdot \end{pmatrix} $$

Both are special cases of GQSP where we set $\lambda = \theta_k = 0$ for XQSP and $\lambda = -\frac{\pi}{4}, \theta_n = \frac{\pi}{4}$ and $\theta_0 = \cdots = \theta_{n-1} = 0$ for YQSP. As a consequence we obtain that $P(z)$ has real coefficients and $Q(z)$ has imaginary coefficients (for XQSP), or both have real coefficients (for YQSP).

`nlft-qsp` provides the `XQSPPhaseFactor` and `YQSPPhaseFactor` classes, as well as the solvers:

- [`xqsp_solve()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.xqsp_solve)
- [`yqsp_solve()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.yqsp_solve)
- [`xqsp_solve_laurent()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.xqsp_solve_laurent)
- [`yqsp_solve_laurent()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.yqsp_solve_laurent)

The first two functions work exactly like `gqsp_solve`, while `*qsp_solve_laurent` are the counterparts for Laurent QSP, expecting a definite-parity Laurent polynomial $P(z)$ with frequencies in $\{ -n, -n+2, \ldots, n-2, n \}$.


## Original GQSP ansatz { #mw-gqsp }

While the construction given above relates more naturally with the other QSP variants, the [original formulation](https://doi.org/10.1103/PRXQuantum.5.020368) was slightly different.

$$ R(\theta_0, \phi_0, \lambda) \tilde{w} R(\theta_1, \phi_1, 0) \tilde{w} \cdots \tilde{w} R(\phi_n, \theta_n, 0) = \begin{pmatrix} P(z) & Q(z) \\ \cdot & \cdot \end{pmatrix} $$

where the matrix $R$ is of the form

$$ R(\theta, \phi, \lambda) = \begin{pmatrix} e^{i(\lambda + \phi)} \cos \theta & e^{i\phi} \sin \theta \\ e^{i\lambda} \sin \theta & -\cos \theta \end{pmatrix} $$

The `GQSPPhaseFactors` class provides a [`to_mw_gqsp()`](../../reference/nlft_qsp/qsp/#nlft_qsp.qsp.GQSPPhaseFactors.to_mw_gqsp) method, which returns the tuple $(\vec{\theta}, \vec{\phi}, \lambda)$ in this order.

!!! note

    This conversion only preserves $P$. $Q$ will turn out to be equal up to a phase.