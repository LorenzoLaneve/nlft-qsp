# Computing phase factors

`nlft-qsp` provides various methods to synthesize QSP protocols in several variants.

It is possible to provide a `Polynomial` or `ChebyshevTExpansion` to directly implement,
or specify a Python function implementing the desired function $f(z) : \mathbb{T} \rightarrow \mathbb{C}$
or $f(x) : [-1, 1] \rightarrow \mathbb{C}$, depending on the variant.


- [Generalized quantum signal processing (GQSP)](gqsp.md)

- [XQSP and YQSP](gqsp.md#xqsp)

- [Chebyshev QSP](qsvt.md#chebyshev-qsp)

- [QSVT](qsvt.md#qsvt)

!!! note

    The naming conventions for the variants of QSP used by this package follow [this paper](https://arxiv.org/abs/2503.03026).