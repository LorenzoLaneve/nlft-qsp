
## Polynomials as Chebyshev expansions { #chebyshev-expansions }

In some QSP variants (including QSVT) generated polynomials are better expressed in terms of Chebyshev expansions
$$ P(x) = \sum_{k = 0}^n c_k T_k(x) $$
where $ T_k(\cos \theta) = \cos(k \theta) $ are the [Chebyshev polynomials of the first kind].

[Chebyshev polynomials of the first kind]: https://en.wikipedia.org/wiki/Chebyshev_polynomials

`nlft-qsp` provides a class `ChebyshevTExpansion`, as well as methods to convert Laurent polynomials into the corresponding Chebyshev expansions.

```py
P = ChebyshevTExpansion([2, 6, 3])
```
<div class="result" markdown>
$$
    P(x) = 2 + 6 T_1(x) + 3 T_2(x) = 2 + 6x + 3 (2x^2 - 1) = -1 + 6x + 6x^2
$$
</div>

---

## Converting to/from Laurent polynomials { #laurent-to-chebyshev }

By passing a Laurent polynomial $P(z)$ to a `ChebyshevTExpansion` you get $T(x)$ such that $T(\cos \theta) = P(e^{i\theta})$ for any $\theta \in [0, 2\pi]$.
```python exec="on" source="block" result="text"
from nlft_qsp import *
import numpy as np

P = Polynomial([2, 1, 5, 1, 2], support_start=-2)
print("P(z) =", P)

T = ChebyshevTExpansion.from_laurent_polynomial(P)

print(P(np.exp(1j*np.pi/3)), T(np.cos(np.pi/3)))
```

Vice versa, it is possible to convert a Chebyshev expansion into a Laurent polynomial:
```python exec="on" source="block" result="text"
from nlft_qsp import *
import numpy as np

T = ChebyshevTExpansion([1, 5, 7])

P = T.to_laurent()
print("P(z) =", P)

print(P(np.exp(1j*np.pi/4)), T(np.cos(np.pi/4)))
```

## Converting to/from polynomials { #poly-to-chebyshev }

The package also provides methods to convert a `Polynomial` $P(x)$ into a Chebyshev expansion $T(x)$ such that $P(x) = T(x)$ and vice versa, i.e., they perform a change of basis between the monomial basis $\{ x^k \}_k$ and the Chebyshev basis $\{ T_k(x) \}_k$.
```python exec="on" source="block" result="text"
from nlft_qsp import *

T = ChebyshevTExpansion([2, 6, 3])

P = T.to_polynomial()

print("P(z) =", P)
print(P(0.75), T(0.75))
```


```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([-1, 6, 6])

T = ChebyshevTExpansion.from_polynomial(P)

print("T(x) =", T)
print(P(0.75), T(0.75))
```