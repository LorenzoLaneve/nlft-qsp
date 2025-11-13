# Utility functions

from typing import Iterable

import csv
from collections import defaultdict


def next_power_of_two(n):
    """Returns the smallest power of two that is `>= n`."""
    return 1 << (n - 1).bit_length()

def sequence_shift(c, s):
    """Shifts the coefficients in the given list by s to the right,
    so that the returned vector `r` satisfies `r[k + s] = c[k]`."""
    s %= len(c)
    return c[-s:] + c[:-s]

def flatten(l):
    """Flattens the multi-dimensional list, as iterable."""
    for x in l:
        if isinstance(x, Iterable):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x

def shape(lst: list):
    """Returns the shape of the given multi-dimensional list, intended as a tuple
    containing the maximum length of the lists along each axis."""
    if not isinstance(lst, list):
        return ()
    
    N0 = len(lst)

    sub_shapes = [shape(sub) for sub in lst]
    if not sub_shapes:
        return (N0,)

    max_sub_shape = tuple(max(sizes) for sizes in zip(*[s + (0,)*(max(len(s) for s in sub_shapes) - len(s)) for s in sub_shapes]))
    return (N0,) + max_sub_shape

def plot_data(test_results: dict):
    degree_to_time = defaultdict(dict)
    degree_to_err = defaultdict(dict)
    all_degrees = set()
    all_keys = list(test_results.keys())

    for key, lst in test_results.items():
        for entry in lst:
            d = entry["degree"]
            t = entry["time"]/1000000000
            e = entry["inlft_err"]
            all_degrees.add(d)
            degree_to_time[d][key] = t
            degree_to_err[d][key] = e

        all_degrees = sorted(all_degrees)

    with open("benchmarks/benchmark_time.dat", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree"] + all_keys)

        for d in all_degrees:
            row = [d] + [degree_to_time[d].get(k, "") for k in all_keys]
            writer.writerow(row)

    with open("benchmarks/benchmark_err.dat", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree"] + all_keys)

        for d in all_degrees:
            row = [d] + [degree_to_err[d].get(k, "") for k in all_keys]
            writer.writerow(row)