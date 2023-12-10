import numpy as np
from functools import partial
from ssm import run_SSM
import plot_utils

def example_mass(k, b , m):
    A = np.array([[0.0, 1.0], [-k/m, -b/m]])
    B = np.array([[0.0], [1.0/m]])
    C = np.array([[1.0, 0.0]])

    return A, B, C

@partial(np.vectorize, signature="()->()")
def example_force(t):
    x = np.sin(10 * t)
    return x * (x > 0.5)

def example_ssm():
    # SSM
    ssm = example_mass(k=40, b=5, m=1)

    # L samples of u(t)
    L = 100
    step = 1.0 / L
    ks = np.arange(L)
    u = example_force(ks * step)

    # Approximation of y(t)
    y = run_SSM(*ssm, u)
    plot_utils.plot_force_example_SSM(L, ks, u, y)

