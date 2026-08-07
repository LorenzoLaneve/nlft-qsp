# Installing the package

## Using pip <small>recommended</small> { #with-pip data-toc-label="Install with pip" }

The package is publicly available as a [Python package] and can be installed using `pip`.

=== "Latest"

    ``` sh
    pip install nlft-qsp
    ```

=== "1.0.x"

    ``` sh
    pip install nlft-qsp=="1.0.*"
    ```

    [Python package]: https://pypi.org/project/nlft-qsp/

!!! note

    Python 3.10+ is required, and Python 3.11+ is recommended.


## With git (editable mode) { #with-git data-toc-label="Install with git" }
It is also possible to clone the [GitHub repo] and then install the package in
editable mode. This is particularly useful if it is necessary to modify the
source code or to use features under development.

``` sh
cd /your/target/directory # (1)!

git clone https://github.com/LorenzoLaneve/nlft-qsp .
pip install -e .
```

1. This should be an empty directory which will hold the package, keep it in a safe place.

    [GitHub repo]: https://github.com/LorenzoLaneve/nlft-qsp


## After installation { #after-install data-toc-label="After installation" }

The package is available in Python by a standard import:
```py
from nlft_qsp import *
```

The package exposes the following components:

- Classes for representing polynomials and expansions relevant for QSP
- Classes to represent QSP protocols
- Functions for QSP/QSVT solvers in several variants
- Utility functions for plotting QSP functions 