

from .backend import NumericBackend
from .backend import generic_complex, generic_real

from ..util import next_power_of_two


class MPMathBackend(NumericBackend):
    """Numeric backend for the mpmath package. This exposes the mpmath implementation.
    
    Note:
        The mpmath package allows to compute with arbitrary precision, but its performance is limited.
    """

    def __init__(self, ctx):
        """Initializes an mpmath backend interface with the given mpmath context.
        
        Args:
            ctx (mpmath.MPContext): the context object exposed by mpmath. It can be `mpmath.mp`, `mpmath.fp`, or `mpmath.iv`.
        """
        self.ctx = ctx

    def __getattr__(self, item): # redirect any other function to the mpmath context
        return getattr(self.ctx, item)

    def pi(self):
        return self.ctx.pi

    def machine_eps(self):
        return 10 ** (-self.ctx.dps)
    
    def machine_threshold(self):
        return 10 ** (-self.ctx.dps//2)
    
    def workdps(self, x: int):
        return self.ctx.workdps(x)
    
    def workprec(self, x: int):
        return self.ctx.workprec(x)
    
    def extradps(self, x: int):
        return self.ctx.extradps(x)
    
    def extraprec(self, x: int):
        return self.ctx.extraprec(x)
    
    def chop(self, x: generic_complex):
        return self.ctx.chop(x)

    def make_complex(self, x: generic_complex):
        return self.ctx.mpc(x)
    
    def make_float(self, x: generic_real):
        return self.ctx.mpf(x)

    def abs(self, x: generic_complex):
        return self.ctx.fabs(x)
    
    def abs2(self, x: generic_complex):
        return self.ctx.re(x)**2 + self.ctx.im(x)**2
    
    def arcsin(self, x: generic_complex):
        return self.ctx.asin(x)
    
    def arccos(self, x: generic_complex):
        return self.ctx.acos(x)
    
    def arctan(self, x: generic_complex):
        return self.ctx.atan(x)
    
    def unitroots(self, N: int):
        return self.ctx.unitroots(N)

    def cooley_tukey_fft(self, x):
        N = len(x)
        if N == 1:
            return x
    
        with self.ctx.extraprec(2):
            # use two extra bits of precision to compute the even and odd FFTs
            # so that eps' = eps/4
            even = self.cooley_tukey_fft(x[0::2])
            odd = self.cooley_tukey_fft(x[1::2])
    
            W_N = self.unitroots(N)
    
            X = [self.make_complex(0)] * N
            for k in range(N // 2):
                X[k] = even[k] + W_N[-k] * odd[k]
                X[k + N // 2] = even[k] - W_N[-k] * odd[k]
    
        return X
    
    def fft(self, x: list[generic_complex], normalize=False):
        N = len(x)
        M = next_power_of_two(N)

        if M > N:
            x = x + [self.make_complex(0)] * (M - N)

        dft = self.cooley_tukey_fft(x)
        if normalize:
            N = len(dft)
            return [y / N for y in dft]
        else:
            return dft
    
    def ifft(self, x: list[generic_complex], normalize=True):
        return [self.conj(y) for y in self.fft([self.conj(xi) for xi in x], normalize)]
    
    def matrix(self, x: list):
        return self.ctx.matrix(x)
    
    def to_list(self, x):
        return x.tolist()
    
    def transpose(self, x):
        return x.transpose()
    
    def conj_transpose(self, x):
        return x.transpose_conj()
    
    def zeros(self, m: int, n: int):
        return self.ctx.zeros(m, n)
    
    def eye(self, n: int):
        return self.ctx.eye(n)
    
    def solve_system(self, A, b):
        return self.ctx.lu_solve(A, b)
    
    def qr_decomp(self, A):
        if A.rows >= A.cols:
            return self.ctx.qr(A)
        
        A1 = A[:A.rows, :A.rows]
        Q, R1 = self.ctx.qr(A1)
        R2 = Q.H * A[:A.rows, A.rows:]

        return Q, self.ctx.matrix([
            [R1[k, j] for j in range(R1.cols)] + [R2[k, j] for j in range(R2.cols)] for k in range(R1.rows)
        ])