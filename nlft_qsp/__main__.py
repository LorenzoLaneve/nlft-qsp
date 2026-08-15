import sys
import functools

import click
import cloup
from cloup.constraints import RequireAtLeast, mutually_exclusive

import numpy as np
import scipy as sp

from importlib.metadata import version as get_version, PackageNotFoundError

from typing import Callable, TextIO

import nlft_qsp.numerics as bd
from nlft_qsp.approximate import chebyshev_approximate, fourier_approximate
from nlft_qsp.file import load_any
from nlft_qsp.qsp import PhaseFactors, get_qsp_variant_class, is_definite_parity, laurent_to_analytic
from nlft_qsp.poly import Polynomial, ChebyshevTExpansion
from nlft_qsp.plot import plot_chebyshev, plot_fourier
from nlft_qsp.solvers import weiss

arg_input_file = cloup.argument('target', type=click.File('r'), required=True)
arg_input_code = cloup.argument('target', type=click.STRING, required=True)
arg_input_files = cloup.argument('targets', nargs=-1, type=click.File('r'), help="Files containing polynomials or Chebyshev expansions to be plotted.")
arg_input_codes = cloup.option('-f', '--func', 'code_inputs', multiple=True, help="Add this flag in front of an argument to interpret it as a Python function.")
arg_qsp_file = cloup.argument('phase_factors', type=click.File('r'), required=True)

opt_output_file = cloup.option('--output', '-o', type=click.File('w'), default='-', help='Output file path. default is - (stdout)')
opt_qsp_variant = cloup.option('--type', '-t', 'variant', type=click.Choice(['qsvt', 'cheb', 'ag', 'lg', 'ax', 'lx', 'ay', 'ly']), default='ag', help='QSP variant to use. a = analytic, l = laurent, g = Generalized QSP, x = XQSP, y = YQSP. default is ag (analytic generalized).')
opt_appr_degree = cloup.option('--degree', '-d', type=click.IntRange(min=0), required=True, help='Degree of the approximation')
opt_poly_only_f = cloup.option('--poly-only', '-p', 'poly_only', is_flag=True, help='Only compute polynomial approximation, skip QSP synthesis')
opt_cheb_only_f = cloup.option('--cheb-only', '-c', 'cheb_only', is_flag=True, help='Only compute Chebyshev approximation, skip QSP synthesis')
opt_plot_circle = cloup.option('--unit-circle', '-u', 'unit_circle', is_flag=True, help='Plot along the unit circle between [-pi, pi] instead.')
opt_qsp_mode = cloup.option('--mode', '-m', 'mode', type=click.Choice(['d', 'l', 'a', 'c']), default='d', help='Mode for the polynomials: [d]efault for the given QSP variant, [a]nalytic, [l]aurent, or [c]hebyshev.')
opt_qsp_both = cloup.option('--both', '-b', is_flag=True, help='Also make the complementary polynomial')

opt_nlft_conv = cloup.option('--nlft', 'nlft_conv', is_flag=True, help='If present, the complementary polynomial is chosen so that the pair of polynomials is in the image of the nonlinear Fourier transform, i.e., the support of the complementary polynomial will end at the zero frequency.')

def catch_errors(
    default_exit_code: int = 1,
    show_traceback: bool = False
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            
            except click.Abort:
                raise

            except Exception as e:
                click.secho("error: ", fg="red", bold=True, err=True, nl=False)
                click.secho(e, err=True)

                if show_traceback:
                    import traceback
                    click.secho("\nTraceback:", fg="yellow", err=True)
                    traceback.print_exc(file=sys.stderr)

                raise SystemExit(default_exit_code)

        return wrapper
    return decorator

def report_warning(msg: str):
    click.secho("warning: ", fg="yellow", bold=True, err=True, nl=False)
    click.secho(msg, err=True)


def solve_variant(P: Polynomial | ChebyshevTExpansion, variant: str):
    if len(variant) < 4:
        variant, mode = variant[1], variant[0]
    else: # cheb/qsvt
        mode = 'c'

    qsp_class = get_qsp_variant_class(variant)

    match mode:
        case 'l':
            return qsp_class.solve_laurent(P)
        case 'a':
            return qsp_class.solve(P)
        case 'c':
            return qsp_class.solve(P)

    raise ValueError(f'Unknown QSP mode {mode}.')

def get_variant_mode(variant: str) -> str:
    if len(variant) < 4:
        return variant[0]

    return 'c'

def get_variant_display_name(variant: str) -> str:
    mode = get_variant_mode(variant)

    mode_str = ''
    match mode:
        case 'l':
            mode_str = 'Laurent '
        case 'a':
            mode_str = 'Analytic '

    if len(variant) < 4:
        variant = variant[1]

    return mode_str + get_qsp_variant_class(variant)._get_variant_display_name

def evaluate_function(code_str: str, mode: str) -> Callable:
    if mode == 'c':
        bound_var = 'x'
    else:
        bound_var = 'z'

    return eval(
        f"lambda {bound_var}: {code_str}",
        {"__builtins__": {}, "np": np, "sp": sp}
    )

def normalize_polynomial(P: Polynomial | ChebyshevTExpansion, variant: str):
    mode = get_variant_mode(variant)

    if mode == 'c':
        if not isinstance(P, ChebyshevTExpansion):
            if (P - P.analytic_part()).l2_squared_norm() <= bd.machine_threshold():
                P = P.analytic_part()

            if P.support_start < 0: # Try to convert Laurent polynomial
                if not P.is_symmetric() or not P.is_real():
                    raise ValueError(f"only real and symmetric Laurent polynomials are compatible with variant '{variant}'.")

                P = ChebyshevTExpansion.from_laurent_polynomial(P)
            else: # normal polynomial, convert from monomial to Chebyshev basis
                if not P.is_real():
                    raise ValueError(f"only real polynomials are compatible with variant '{variant}'.")

                P = ChebyshevTExpansion.from_polynomial(P)

        if not is_definite_parity(P):
            raise ValueError("target is not of definite parity, which is required by Chebyshev QSP variants.")

        return P

    if isinstance(P, ChebyshevTExpansion):
        raise ValueError(f"Chebyshev expansions are compatible only with Chebyshev QSP and QSVT, not for {variant}.")

    if mode == 'a':
        if (P - P.analytic_part()).l2_squared_norm() > bd.machine_threshold():
            report_warning("analytic QSP was chosen but the given function has a Laurent "
                           "expansion with non-negative frequencies. These are discarded.")

        return P.analytic_part()

    if mode == 'l':
        if not is_definite_parity(P):
            raise ValueError("target polynomial is not of definite parity, "
                             f"which is required by variant '{variant}'.")

        return P

    raise ValueError(f"unknown QSP mode '{mode}'.")


@cloup.group(invoke_without_command=True, no_args_is_help=True)
@cloup.option('--version', '-v', is_flag=True, help='Print package version and exit')
@catch_errors()
def main(version: bool):
    """The Quantum Signal Processing eXtractor."""
    if version:
        try:
            ver = get_version('nlft-qsp')
        except PackageNotFoundError:
            ver = 'unknown'
        print('nlft-qsp', ver)
        raise SystemExit(0)


@main.command('solve')
@arg_input_file
@opt_output_file
@opt_qsp_variant
@catch_errors()
def solve(target: TextIO, variant: str, output: TextIO):
    """Synthesize a QSP protocol for a given polynomial or Chebyshev expansion."""
    P = load_any(target)

    if not isinstance(P, (Polynomial, ChebyshevTExpansion)):
        raise ValueError("Target must be either a polynomial or a Chebyshev expansion.")

    P = normalize_polynomial(P, variant)

    pf = solve_variant(P, variant)
    pf.dump_json(output)


@main.command('approximate', aliases=['approx'])
@arg_input_code
@opt_appr_degree
@opt_poly_only_f
@opt_cheb_only_f
@opt_qsp_variant
@opt_output_file
@cloup.constraint(mutually_exclusive, ['poly_only', 'cheb_only'])
@catch_errors()
def approximate(target: str, degree: int, poly_only: bool, cheb_only: bool, variant: str, output: TextIO):
    """Solve for a QSP protocol approximating the given function given as a Python expression."""
    if cheb_only:
        variant = 'cheb'
    f = evaluate_function(target, get_variant_mode(variant))

    if cheb_only:
        chebyshev_approximate(f, degree).dump_json(output)
        return

    if poly_only:
        fourier_approximate(f, degree).dump_json(output)
        return

    match get_variant_mode(variant):
        case 'a':
            P = fourier_approximate(f, degree)
        case 'l':
            P = fourier_approximate(f, degree)
        case 'c':
            P = chebyshev_approximate(f, degree)

    P = normalize_polynomial(P, variant)

    pf = solve_variant(P, variant)
    pf.dump_json(output)


@main.command('plot')
@arg_input_files
@arg_input_codes
@opt_plot_circle
@cloup.constraint(RequireAtLeast(1), ["targets", "code_inputs"])
@catch_errors()
def plot(targets: list[TextIO], code_inputs: list[str], unit_circle: bool):
    """Plot given polynomials and functions in the interval [-1, 1]."""
    funcs = {}

    if unit_circle:
        mode = 'l'
    else:
        mode = 'c'

    idx = 0
    for file in targets:
        P = load_any(file)

        if mode == 'l' and not isinstance(P, Polynomial):
            raise ValueError('Only polynomials can be plotted along the unit circle.')

        if mode == 'c' and not isinstance(P, (Polynomial, ChebyshevTExpansion)):
            raise ValueError('Only polynomials and Chebyshev expansions can be plotted in [-1, 1].')

        funcs[f"Poly {idx}"] = P
        idx += 1

    idx = 0
    for code in code_inputs:
        funcs[f"Function {idx}"] = evaluate_function(code, mode)

    if mode == 'l':
        plot_fourier(funcs)
    else:
        plot_chebyshev(funcs)


@main.command('make')
@arg_qsp_file
@opt_qsp_mode
@opt_output_file
@catch_errors()
def make(phase_factors: TextIO, output: TextIO, mode: str):
    """Make polynomials/Chebyshev expansions for given QSP phase factors."""
    pf = load_any(phase_factors)

    if not isinstance(pf, PhaseFactors):
        raise ValueError("input file does not contain a valid set of phase factors.")

    cls = pf.__class__
    if mode == 'd':
        mode = cls._variant_modes[0]

    if mode not in cls._variant_modes:
        raise ValueError(f"chosen mode '{mode}' is incompatible with variant '{cls._variant_tag}' "
                         f"found in file (compatible variants: '{"', '".join(cls._variant_modes)}').")
    
    P, _ = pf.polynomials(mode='laurent')

    match mode:
        case 'l':
            P.dump_json(output)
        case 'a':
            laurent_to_analytic(P).dump_json(output)
        case 'c':
            ChebyshevTExpansion.from_laurent_polynomial(P).dump_json(output)

    
@main.command('complete')
@arg_input_file
@opt_output_file
@opt_nlft_conv
@catch_errors()
def complete(target: TextIO, output: TextIO, nlft_conv: bool):
    """Compute a complementary polynomial to TARGET."""
    P = load_any(target)
    if not isinstance(P, Polynomial):
        raise ValueError("TARGET is not a valid polynomial.")

    Q = weiss.complete(P)
    if not nlft_conv:
        Q = Polynomial(Q.coeffs, P.support_start)

    Q.dump_json(output)