
import numpy as np
import scipy as sp

from ...util import next_power_of_two

from ... import numerics as bd
from ...poly import Polynomial
from ...numerics import complex_type
from ...solvers.half_cholesky import half_cholesky_matrix_ldl, solve_ldl_system
from ...approximate import laurent_approximation

from .weiss import WEISS_MAX_ATTEMPTS, WeissConvergenceError


def scalar_spectral_phase(f_evals: list[complex]) -> list[complex]:
    """Evaluations of the phases that make f outer, given its evaluations."""
    N = len(f_evals)

    log_f = laurent_approximation(np.log(np.abs(f_evals)))

    return [np.exp(iHlog_fk) for iHlog_fk in log_f.hilbert_transform().eval_at_roots_of_unity(N)]

def pointwise_cholesky(P: Polynomial, N: int) -> Polynomial:
    """Returns a polynomial approximation to the function M with the following properties:
     (1) M @ M.H = P
     (2) M is lower triangular with positive-mean outer diagonal entries.
     
     Note:
      N is assumed to be a power of two."""
    N = next_power_of_two(N)
    
    if len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square and positive-definite on the unit circle.")
    
    d = P.shape[0]

    M_evals = [sp.linalg.cholesky(Pz, lower=True) for Pz in P.eval_at_roots_of_unity(N)]

    fplus_phase_evals = [scalar_spectral_phase([eval[k][k] for eval in M_evals]) for k in range(d)]

    Mp_evals = [M_evals[j] @ np.diag([fplus_phase_evals[k][j] for k in range(d)]) for j in range(N)] # TODO: multiply in place

    return laurent_approximation(Mp_evals)


def scalar_spectral_factor_from_evals(f_evals: list[complex]) -> Polynomial:
    """Returns a polynomial approximation to the outer function q(z) satisfying |q|^2 = f, given its vector of evaluations. Its degree will be len(f_evals)//2"""
    p = laurent_approximation([np.abs(f_eval) * eiHlog_fk for f_eval, eiHlog_fk in zip(f_evals, scalar_spectral_phase(f_evals))])
    return p.analytic_part()

def scalar_spectral_factor(f: Polynomial, N: int) -> Polynomial:
    """Returns a polynomial approximation to the outer function q(z) satisfying |q|^2 = f, with degree N//2."""
    return scalar_spectral_factor_from_evals(f.eval_at_roots_of_unity(N))

def inverse_polynomial(f: Polynomial, N: int) -> Polynomial:
    """Given f (assumed to be outer), returns the Taylor approximation for 1/f up to order N//2."""
    return laurent_approximation([1/fk for fk in f.eval_at_roots_of_unity(N)]).analytic_part()

def matrix_inverse_polynomial(F: Polynomial, N: int) -> Polynomial:
    """Given a matrix polynomial F (assumed to be outer), returns a Taylor approximation or F^{-1} up to order N//2."""
    return laurent_approximation([np.linalg.inv(Fk) for Fk in F.eval_at_roots_of_unity(N)]).analytic_part()


def block_hankel(Zeta: Polynomial) -> np.ndarray:
    """Returns the block Hankel tensor associated to Zeta.

    If Zeta.shape = (d1, d2), returns an array of shape (deg+1, deg+1, d1, d2),
    where deg = -Zeta.support_start and block (i, j) is Zeta[-(i+j)] when i+j <= deg,
    zero otherwise.
    """
    deg = -Zeta.support_start
    d1, d2 = Zeta.shape

    Gamma_blocks = np.zeros((deg + 1, deg + 1, d1, d2), dtype=complex_type)
    for i in range(deg + 1):
        for j in range(deg + 1):
            if i + j <= deg:
                Gamma_blocks[i, j] = np.array(Zeta[-(i + j)])
    return Gamma_blocks

def flatten_block_hankel(Gamma_blocks: np.ndarray) -> np.ndarray:
    """Flattens a block Hankel tensor (deg+1, deg+1, d1, d2) to 2D matrix form."""
    nb1, nb2, d1, d2 = Gamma_blocks.shape
    return np.transpose(Gamma_blocks, (0, 2, 1, 3)).reshape(nb1 * d1, nb2 * d2)

def id_block_tensor(d: int, N: int) -> np.ndarray:
    """Returns a tensor of N+1 d x d blocks, with first block I and others 0."""
    bten = np.zeros((N + 1, d, d), dtype=complex)
    bten[0] = np.eye(d)
    return bten

def id_block_vector(d: int, N: int):
    """Returns a column vector of N+1 d x d blocks, where all the blocks are zeros except the first one which is the identity."""
    return id_block_tensor(d, N).reshape(d * (N + 1), d)

def block_conjugate_transpose_blocks(bten: np.ndarray) -> np.ndarray:
    """Blockwise conjugate-transpose for tensors of shape (N+1, d1, d2)."""
    return np.conjugate(np.transpose(bten, (0, 2, 1)))

def block_conjugate_transpose(bvec, N: int):
    """Given a column vector (or tensor) of N+1 d1 x d2 blocks, returns blockwise conjugate-transpose."""
    if isinstance(bvec, np.ndarray) and bvec.ndim == 3:
        return block_conjugate_transpose_blocks(bvec)
    d1 = bvec.shape[0] // (N + 1)
    d2 = bvec.shape[1]
    blocks = bvec.reshape(N + 1, d1, d2)
    return block_conjugate_transpose_blocks(blocks).reshape((N + 1) * d2, d1)

def block_transpose(bvec, N: int):
    """Given a column vector of N+1 d1 x d2 blocks, returns a column vector of N+1 d2 x d1 blocks.

    Each block is transposed. The resulting vector has shape (d2*(N+1), d1).
    """
    d1 = bvec.shape[0] // (N + 1)
    d2 = bvec.shape[1]

    result = np.zeros((d2 * (N+1), d1), dtype=complex)
    for i in range(N + 1):
        block = bvec[i * d1:(i + 1) * d1, :d2]
        result[i * d2:(i + 1) * d2, :d1] = np.transpose(block)
    return result

def solve_jl_system(Zeta):
    """Returns the (approximate) unitary solution to the Janashia-Lagvilava system of equations associated to Zeta, i.e., the
    unitary matrix U such that [[I, 0], [Zeta, I]] @ U is outer.

    Note: if Zeta has shape (d1, d2) then U has shape (d1 + d2, d1 + d2).
    """
    d1, d2 = Zeta.shape
    N = -Zeta.support_start

    Gamma_blocks = block_hankel(Zeta)
    Gamma = flatten_block_hankel(Gamma_blocks)
    Gamma_H = Gamma.conj().T

    #IGG = np.eye(Gamma.shape[0], dtype=complex) + Gamma @ Gamma_H

    #E1 = id_block_vector(d1, N)
    E2 = id_block_vector(d2, N)

    # Zeta.coeffs is in reverse order, but the method below computes matrix inverse for a Toeplitz matrix, not Hankel.
    L, D = half_cholesky_matrix_ldl(np.array([Zeta[j] for j in range(Zeta.support_start, 1)], dtype=complex_type))

    #Db = sp.linalg.block_diag(*D)
    #IGG_bflipped = np.flip(IGG.reshape(N+1, d1, N+1, d1), axis=(0, 2)).reshape((N+1) * d1, (N+1) * d1)
    #print("err: ", np.linalg.norm(IGG_bflipped - L @ Db @ L.conj().T))

    V_22_H = solve_ldl_system(L, D, id_block_tensor(d1, N), mode='hankel') #V_22_H = GG_inv @ E1
    V_21 = Gamma_H @ V_22_H # TODO: Toeplitz structure can be exploited here

    V_12_H = solve_ldl_system(L, D, Gamma @ E2, mode='hankel') #V_12_H = GG_inv @ (Gamma @ E2)
    V_11 = Gamma_H @ V_12_H - E2

    V_11_blocks = V_11.reshape(N + 1, d2, d2)
    V_21_blocks = V_21.reshape(N + 1, d2, d1)
    V_12_blocks = block_conjugate_transpose_blocks(V_12_H.reshape(N + 1, d1, d2))
    V_22_blocks = block_conjugate_transpose_blocks(V_22_H.reshape(N + 1, d1, d1))

    v_11 = Polynomial(V_11_blocks)
    v_21 = Polynomial(V_21_blocks)
    v_12 = Polynomial(V_12_blocks)
    v_22 = Polynomial(V_22_blocks)

    V = Polynomial.block_matrix([
        [v_11,             v_21],
        [v_12.conjugate(), v_22.conjugate()]
    ])
    
    return V @ np.linalg.inv(V(1))


def janashia_lagvilava_unitary(M: Polynomial, N: int, d_min:int = 0, d_max:int = -1) -> Polynomial:
    """Returns an (approximate) unitary matrix function U such that Mx @ U is outer.
    Mx here is the submatrix M[d_min : d_max, d_min : d_max] (d_min included, d_max excluded).

    Note: it is assumed that M is lower triangular and already contains (approximate) outer functions on the diagonal.

    Both the outer spectral factor and the outerizing unitary are returned.
    """
    if d_max < 0:
        d_max = M.shape[0]

    if d_max - d_min <= 1:
        return Polynomial(np.array([[[1]]], dtype=complex_type))

    M_1 = M[:, d_min : d_max // 2, d_min : d_max // 2]
    M_2 = M[:, d_max // 2 : d_max, d_max // 2 : d_max]
    L   = M[:, d_max // 2 : d_max, d_min : d_max // 2]

    U_1 = janashia_lagvilava_unitary(M_1, N)
    U_2 = janashia_lagvilava_unitary(M_2, N)

    #S_1 = (M_1 * U_1).analytic_part()
    S_2 = (M_2 @ U_2).truncate(0, N-1)
    Zeta = (L @ U_1).truncate(-N+1, 0)

    Finv = matrix_inverse_polynomial(S_2, N)

    U = solve_jl_system((Finv @ Zeta).truncate(-N, 1))

    UD = Polynomial.diagonal_block_matrix([U_1, U_2])
    return UD @ U


def spectral_factor(P: Polynomial, eta: float, eps: float=1e-10, verbose: bool=False):
    # Pointwise Cholesky factorization of S

    if P.shape[0] != P.shape[1]:
        raise ValueError(f"Expected square matrix polynomial, got {P.shape}.")

    if eps < 0:
        eps = bd.machine_threshold() * 10e-3

    n = P.effective_degree() // 2 # P is the symmetric Laurent polynomial

    N = next_power_of_two(int(n/eta))//2

    threshold = 1
    attempts = 0
    while threshold > eps:
        N *= 2
        if verbose:
            print(f"N = {N}")

        print("Computing pointwise Cholesky... ")
        M = pointwise_cholesky(P, N)
        print(M.support())

        print("Computing Janashia-Lagvilava unitary...")
        U = janashia_lagvilava_unitary(M, N)

        S = (M @ U).truncate(0, n)

        new_thr = (S @ S.conjugate() - P).l2_norm()
        if verbose:
            print(f"N = {N:>7}, threshold = {new_thr}")

        if threshold <= new_thr:
            attempts += 1
            if attempts >= WEISS_MAX_ATTEMPTS:
                raise WeissConvergenceError()
        else:
            threshold = new_thr
            attempts = 0

    return S, U