"""
Standard results from SIR model to use as targets to test solvers
"""

from scipy.special import lambertw
from scipy.special import factorial
import numpy as np

def final_size_success_mean(R0, s0=1.0):
    """
    Successful epidemic final size (fraction)

    R0: basic reproductive number
    s0: initial fraction susceptible (optional, default = 1)
    """
    if (s0>1) or (s0<0):
        raise ValueError("s0 must be between 0 and 1")

    Reff = s0*R0

    if Reff <= 1:
        return 0

    s_final = -(1/R0) * lambertw(-Reff * np.exp(-Reff) ).real

    return s0 - s_final

def final_size_success_sd(N, R0, s0=1):
    """
    Standard deviation of successful epidemic final size (absolute)
    eq (1) of https://www.stat.berkeley.edu/~aldous/260-FMIE/Papers/britton.pdf

    R0: basic reproductive number
    s0: initial fraction susceptible (optional, default = 1)
    """
    if (s0>1) or (s0<0):
        raise ValueError("s0 must be between 0 and 1")

    # fractional final size
    z = final_size_success_mean(R0, s0=s0)

    # # r^2 = V (I)/(E(I))^2 is the squared coefficient of variation of the
    # # infectious period, so for an Exponential distribution r=1
    # r = 1
    # final_sd = np.sqrt(N * z * (1-z) * (1 + r**2 *(1-z) * R0)) / (1 - R0 * (1-z))

    final_sd = np.sqrt(N * z * (1 - z)) / (1 - R0 * (1 - z))

    return final_sd

def probability_failure(R0):
    """
    Probability of continuous time branching process going extinct in subcritical regime.
    """
    return 1/R0

def failed_dist(R0, n):
    """
    Final size distribution, conditioned upon failure
    https://www.mdpi.com/1660-4601/7/3/1186
    """
    r = 1 / R0
    n = np.asarray(n)
    return ((r * n) ** (n - 1) / factorial(n)) * np.exp(-r * n)

def final_size_failed_mn(R0):
    """
    Expected final size of epidemic conditioned upon failure
    """
    return R0/(R0-1)

def final_size_failed_sd(R0):
    """
    Expected standard deviation of epidemic conditioned upon failure
    """
    return np.sqrt( (R0**2)/((R0-1)**3) )

# TODO: are these including initial case? yes i think so