# Representing polynomials { #representing-polynomials }

The package uses the `Polynomial` class to represent and manipulate Laurent polynomials.

```py
P = Polynomial([0.4, 0.3, 0.5, 0.7])
```
<div class="result" markdown>
$$
    P(z) = 0.4 + 0.3z + 0.5z^2 + 0.7 z^3
$$
</div>

It is possible to shift the coefficients by specifying where the support should start with `support_start`

```py
P = Polynomial([0.4, 0.3, 0.5, 0.7], support_start=-2)
```
<div class="result" markdown>
$$
    P(z) = 0.4z^{-2} + 0.3z^{-1} + 0.5 + 0.7 z
$$
</div>

## Accessing coefficients { #accessing-coefficients }

After the polynomial is allocated, it is possible to get or set individual coefficients using the subscript operation:
```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([0.4, 0.3, 0.5, 0.7], support_start=-2)
print(P[-2])
P[4] = 5
print(P[3])
print(P[4])
```

The subscript does not work only with the originally allocated indices. If an element outside the original support is accessed, then zero is returned, while if such an element is set then the coefficient list and support start will be reallocated accordingly.

!!! note

    Setting the coefficients is the only way to modify a `Polynomial` object after it is created.
    All the other methods will always produce a new polynomial.


## Support of the polynomial { #support }

The `support()` method returns a Python `range` containing the indices of all the coefficients appearing in the polynomial:
```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([0.4, 0.3, 0.5, 0.7], support_start=-2)
print(0 in P.support())
print(2 in P.support())
```

!!! warning

    `support()` simply checks `support_start` and the length of the internal coefficient list, not the actual mathematical support of the polynomial.
    ```python exec="on" source="block" result="text"
    from nlft_qsp import *

    Q = Polynomial([1, 0, 0, 0])
    print(3 in Q.support())
    ```


## Evaluation { #evaluation }

A polynomial can be evaluated by simply calling it with an input complex number.

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([4j, 2])
print("P(z) =", P)
print("P(1 + 2j) =", P(1 + 2j))
```

Sometimes it might be useful to evaluate $P(e^{2\pi i k/N})$ for the $N$-th roots of unity $e^{2\pi i k/N}$.
While it is possible to evaluate each term separately in $\mathcal{O}(N^2)$ using the above code, it is
possible to do it in $\mathcal{O}(N \log N)$ using a [Fast Fourier transform]. This is done by the `eval_at_roots_of_unity()` method.

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([4j, 2])
print(P.eval_at_roots_of_unity(16))
```

!!! note

    Currently `eval_at_roots_of_unity(N)` method assumes $N$ to be a power of two. If a non-power of two is passed then the next power of two is assumed.


[Fast Fourier transform]: https://en.wikipedia.org/wiki/Fast_Fourier_transform