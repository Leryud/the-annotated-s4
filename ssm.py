from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh, inv, matrix_power
from jax.scipy.signal import convolve
from layers import SSMLayer, SequenceBlock, Embedding, StackedModel

def random_SSM(rng, N: int):
    """Initialization of a random SSM with states A, B and C of size N

    Args:
        - rng: pseudo-random generator for random values initialization inside the matrices
        - N: the size of the matrices

    Returns:
        tuple: (A, B, C), the random matrices needed for a SSM.
    """
    a_r, b_r, c_r = jax.random.split(rng, N)
    A = jax.random.uniform(a_r, (N, N))
    B = jax.random.uniform(b_r, (N, 1))
    C = jax.random.uniform(c_r, (1, N))

    return A, B, C

def discretize(A, B, C, step):
    """Discretize SMM matrices by `step` size, allowing to process discrete input data

    Args:
        - A (np.array): matrix of size (N, N)
        - B np.array): matrix of size (N, N)
        - Cnp.array): matrix of size (N, N)
        - step (int): the step size

    Returns:
        tuple: (Ab, Bb, C), the coefficients of a sequence-to-sequence map
            between an function `u` at step k and its output `y`
    """
    I = np.eye(A.shape[0])  # noqa: E741
    BL = inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B

    return Ab, Bb, C

def scan_SSM(Ab, Bb, Cb, u, x0):
    """Step function, looking a lot like an RNN's own"""
    def step(x_k_1, x_k):
        x_k = Ab @ x_k_1 + Bb @ x_k
        y_k = Cb @ x_k

        return x_k, y_k

    return jax.lax.scan(step, x0, u)

def run_SSM(A, B, C, u):
    """
    Run the SSM with the input vector u.

    Parameters:
    - A: numpy.ndarray, the matrix A.
    - B: numpy.ndarray, the matrix B.
    - C: numpy.ndarray, the matrix C.
    - u: numpy.ndarray, the input vector u.

    Returns:
    - numpy.ndarray, the result of the function call `scan_SSM`.
    """
    L = u.shape[0]
    N = A.shape[0]

    Ab, Bb, Cb = discretize(A, B, C, step=1.0/L)

    return scan_SSM(Ab, Bb, Cb, u[:, np.newaxis], np.zeros((N,)))[1]

def K_conv(Ab, Bb, Cb, L):
    """
    Generates a K-convolution matrix using the given parameters.

    Args:
        Ab (np.array): The matrix Ab.
        Bb (np.array): The matrix Bb.
        Cb (np.array): The matrix Cb.
        L (int): The number of iterations.

    Returns:
        np.array: The K-convolution matrix.
    """
    return np.array(
        [(Cb @ matrix_power(Ab, exponent) @ Bb).reshape() for exponent in range(L)]
    )

def causal_convolution(u, K, no_fft=False):
    """
    Perform causal convolution between two arrays using either FFT or direct convolution.

    Args:
        u (ndarray): The input array.
        K (ndarray): The kernel array.
        no_fft (bool, optional): If True, use direct convolution. Defaults to False.

    Returns:
        ndarray: The result of the causal convolution.
    """
    if no_fft:
        return convolve(u, K, mode="full")[: u.shape[0]]

    else:
        assert K.shape[0] == u.shape[0]
        ud = np.fft.rfft(np.pad(u, (0, K.shape[0])))
        Kd = np.fft.rfft(np.pad(K, (0, u.shape[0])))
        out = ud * Kd

        return np.fft.irfft(out)[: u.shape[0]]

def test_cnn_is_rnn(N=3, L=16, step=1.0 / 16):
    """
    Test if the CNN is equivalent to the RNN under the SSM model.

    The discrete convolution theorem - for circular convolution of
    two sequences - allows us to efficiently calculate the output of
    convolution by first multiplying FFTs of the input sequences and
    then applying an inverse FFT.

    Parameters:
        N (int): The value of N. Defaults to 4.
        L (int): The value of L. Defaults to 16.
        step (float): The step value. Defaults to 1.0 / 16.

    Returns:
        None
    """
    ssm = random_SSM(rng, N)
    u = jax.random.uniform(rng, (L,))
    jax.random.split(rng, 3)

    # RNN
    rec = run_SSM(*ssm, u)

    # CNN
    ssmb = discretize(*ssm, step=step)
    conv = causal_convolution(u, K_conv(*ssmb, L))

    assert np.allclose(rec.ravel(), conv.ravel())


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """Initializes an random 1D array with values ranging from `dt_min` to `dt_max`.

    Used to define the step size in log space.

    Args:
        dt_min: Minimal time delta. Defaults to 0.001.
        dt_max: Maximal time delta. Defaults to 0.1.

    Returns:
        Array: the initialized array
    """
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init

def cloneLayer(layer):
    """Clones a layer and returns a copy.

    Args:
        layer: a SSM layer

    Returns:
        SSMLayer: a cloned SSM layer.
    """
    return nn.vmap(
        layer,
        in_axes=1,
        out_axes=1,
        variable_axes={"params": 1, "cache": 1, "prime": 1},
        split_rngs={"params":True}
    )


if __name__ == "__main__":
    rng = jax.random.PRNGKey(1)

    SSMLayer = cloneLayer(SSMLayer)

    BatchStackModel = nn.vmap(
        StackedModel,
        in_axes=0,
        out_axes=0,
        variable_axes={"params": None, "dropout": None, "cache": 0, "prime": None},
        split_rngs={"params": False, "dropout": True},
    )