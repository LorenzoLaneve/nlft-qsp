## NLFT/QSP solver

[![tests](https://github.com/LorenzoLaneve/nlft-qsp/actions/workflows/tests.yml/badge.svg)](https://github.com/LorenzoLaneve/nlft-qsp/actions/workflows/tests.yml)
[![Last Release](https://img.shields.io/github/v/release/LorenzoLaneve/nlft-qsp?logo=github&label=latest)](https://github.com/LorenzoLaneve/nlft-qsp/releases)
[![Python Versions](https://img.shields.io/pypi/pyversions/nlft-qsp.svg)](https://pypi.org/project/nlft-qsp/)

This Python package provides tools for the computation of quantum signal processing phase factors for given desired transformations.

Read the [full documentation](https://lorenzolaneve.github.io/nlft-qsp/) for installation and usage.


## Examples

The `examples/` folder contains notebooks applying QSP/QSVT or QSP/QET-U to different settings.

- `grover.ipynb` computes the QSP protocol for Grover's search/fixed-point amplitude amplification.

- `matrix_inversion.ipynb` computes the QSP protocol approximating the eigenvalue transformation $f(x) = 1/x$.

- `eigenvalue_filtering.ipynb` shows how to locate an eigenphase on the unit circle.

Performance analysis of the QSP/NLFT solvers are available in the `benchmarks.ipynb` notebook.