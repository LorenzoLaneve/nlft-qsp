
The package provides methods to load and save many classes to the disk. In particular, the following classes are serializable:

- [`Polynomial`](../reference/nlft_qsp/poly.md#nlft_qsp.poly.Polynomial)
- [`ChebyshevTExpansion`](../reference/nlft_qsp/poly.md#nlft_qsp.poly.ChebyshevTExpansion)
- [`XQSPPhaseFactors`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.XQSPPhaseFactors)
- [`YQSPPhaseFactors`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.YQSPPhaseFactors)
- [`GQSPPhaseFactors`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.GQSPPhaseFactors)
- [`ChebyshevQSPPhaseFactors`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.ChebyshevQSPPhaseFactors)
- [`QSVTPhaseFactors`](../reference/nlft_qsp/qsp.md#nlft_qsp.qsp.QSVTPhaseFactors)

Each of these classes provides two methods `dump_json()` and `load_json()`, both of which take a path to a file or a file stream.

```python
from nlft_qsp import *

P = Polynomial([1/7, 0, 1/5, 0, 1/6])

P.dump_json('P.qspx')
```

```python
from nlft_qsp import *

P = Polynomial.load_json('P.qspx')

pf = QSVTPhaseFactors.solve(P)

pf.dump_json('protocol.qspx')
```


### Command line interface

`qspx` takes files as arguments for its inputs and outputs. The following command computes the phase factors of an analytic generalized QSP protocol (`ag`) for the polynomial given in the file `P.qspx` and saves the computed protocol into the file `protocol.qspx`.

```bash
qspx solve -tag P.qspx -o protocol.qspx
```

Both inputs and outputs can be specified as `-`, which is the standard POSIX for standard input/standard output, respectively (for the output, this is the default). This is useful for usual piping in bash:

```bash
qspx make protocol.qspx | qspx plot - -f '0.5*np.cos(3*x)'
```

The above example takes the QSVT protocol saved in `protocol.qspx`, and plots the implemented polynomial in $[-1, 1]$ along with another function $f(x) = \frac{1}{2} \cos(3x)$ for comparison.