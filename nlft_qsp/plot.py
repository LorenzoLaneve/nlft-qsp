"""Helper functions to plot polynomials using matplotlib."""

from matplotlib import pyplot as plt

import numpy as np

def plot_chebyshev(funcs: dict, num_points: int=1000):
    r"""
    Plots the real part of each object in funcs over the interval $[-1, 1]$.
    These can be Python functions, `Polynomial` objects, `ChebyshevTExpansion` objects, or
    any callable object.

    Args:
        funcs (dict): a dictionary where each key is the name appearing in the legend of the corresponding function plot.
        num_points (int): number of sampling points.
    """
    plt.figure(figsize=(6, 3))
    
    for name, f in funcs.items():
        try:
            x_vals = [-1 + 2*k/num_points for k in range(num_points+1)]
            y_vals = [np.real(f(x)) for x in x_vals]
            plt.plot(x_vals, y_vals, label=name)
        except Exception as e:
            print(f"Error evaluating function {name}: {e}")
    
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_fourier(funcs: dict, num_points: int=1000):
    r"""
    Plots the absolute value of each object in funcs over the unit circle, i.e., plugging $z = e^{it}$ for $t \in [-\pi, \pi]$.
    These can be Python functions, `Polynomial` objects, or any callable object.

    Args:
        funcs (dict): a dictionary where each key is the name appearing in the legend of the corresponding function plot.
        num_points (int): number of sampling points.
    """
    plt.figure(figsize=(6, 3))
    
    for name, f in funcs.items():
        try:
            x_vals = [-np.pi + 2*np.pi*k/num_points for k in range(num_points+1)]
            y_vals = [np.abs(f(np.exp(1j * x))) for x in x_vals]
            plt.plot(x_vals, y_vals, label=name)
        except Exception as e:
            print(f"Error evaluating function {name}: {e}")
    
    plt.xlabel("x")
    plt.xlim(left=-np.pi, right=np.pi)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_support_2d(l: list, rng: tuple[range]):
    r"""Plots where the given objects have non-zero elements (up to machine threshold) in rng.
    
    Args:
        l (list): A list of objects that support subscript operation.
        rng (tuple[range]): A 2D tuple of ranges defining the rectangle in the $\Z^2$ grid to plot."""

    if len(rng) != 2:
        raise ValueError("rng must have dimension 2.")

    for k in range(len(l)):
        px, py = zip(*[
            (x+k*0.05, y+k*0.05) for x in rng[0] for y in rng[1] if np.abs(l[k][x, y]) > bd.machine_threshold()
        ])
        
        plt.scatter(px, py, marker='o', label=f"Support #{k+1}")

    # Formatting
    plt.xlabel("k")
    plt.ylabel("h")
    plt.title("Supports")
    plt.grid(False)
    plt.legend()
    plt.gca().set_aspect('equal')

    plt.show()