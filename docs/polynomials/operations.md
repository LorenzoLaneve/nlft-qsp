# Available operations between polynomials

## Arithmetic operations { #arithmetic data-toc-label="Arithmetic operations" }

All basic arithmetic operations between polynomials are supported and exposed by standard Python operations.

- Addition and subtraction between polynomials 
- Addition/subtraction of a scalar
- Unary negation operation
- Multiplication between polynomials
- Multiplication/division with a scalar

The support of the resulting polynomial, i.e. the length of its coefficient list, will be the minimal required to keep all the coefficients.

Adding/subtracting a scalar means adding/subtracting it to the constant term of the polynomial.

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([0.4, 0.3, 0.5, 0.7], support_start=-2)
print(P + 30)
```


## Truncating support { #truncate data-toc-label="Truncating support" }

It is possible to use `truncate()` easily take only parts of a polynomial, discarding frequencies we are not interested in, or
that are likely to contain negligible coefficients.

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([1, 2, 3, 4, 5, 6, 7], support_start=-2)
print(P)
print(P.truncate(1, 3))
```

Note that both endpoints are **included**.



## Conjugate polynomials { #conjugate data-toc-label="Conjugate polynomials" }

Given a Laurent polynomial $P(z) = \sum_k p_k z^k$, the conjugate polynomial is

$$ P^*(z) = \overline{P(1/\overline{z})} = \sum_k \overline{p_k} z^{-k} $$
Such polynomial can be easily obtain using the `conjugate()` method:

```python exec="on" source="block" result="text"
from nlft_qsp import *

P = Polynomial([7+3j, 4+2j, 5+1j])
print(P)
print(P.conjugate())
```