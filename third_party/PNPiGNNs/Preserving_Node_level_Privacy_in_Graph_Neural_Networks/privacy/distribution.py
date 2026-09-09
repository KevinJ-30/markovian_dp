import torch
import numpy as np
import scipy.stats as stats
import torch.functional as F
import sys
import os
import scipy    

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import f_DP_transform.tail_boud as tail_boud
 
# from scipy.stats import norm as scipy_norm
# import privacy.log_space as log_space




G = torch.distributions.normal.Normal(0, 1)
L = torch.distributions.laplace.Laplace(0, 1)

def log_a_minus_b(a, b):
    assert (a >= b).all(), "a must be greater than or equal to b for log_a_minus_b"
    return a + torch.log1p(-torch.exp(b - a))

def logerfc(x):
    """
    Numerically stable approximation of log(erfc(x)) for torch tensors.

    Parameters:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: log(erfc(x)) computed in a numerically stable way.
    """
    ##Safe threshold: beyond this, erfc is extremely small, and log(erfc(x)) ≈ -x^2 - log(pi)/2 - log(x)
    # threshold = 20.0
    # safe_region = x < threshold

    # # Direct computation where erfc is not too small
    # direct = torch.log(torch.erfc(x))

    # # Asymptotic expansion for large x: log(erfc(x)) ≈ -x^2 - log(pi)/2 - log(x)
    # # This is derived from Abramowitz & Stegun 7.1.23

    # approx = -x**2 - 0.5 * np.log(np.pi) - torch.log(x)
    # return torch.where(safe_region, direct, approx)

    '''
    more accurate approximation for large x:
    '''
    threshold = 20.0

    safe_region = x < threshold

    # Direct computation where it's safe
    direct = torch.log(torch.erfc(x))

    # Improved asymptotic approximation (based on A&S 7.1.23 with more terms)
    x_safe = x.clone()
    x_safe[x_safe < threshold] = threshold  # Avoid division by zero in log for bad x

    leading = -x_safe**2 - 0.5 * np.log(np.pi) - torch.log(x_safe)
    correction = torch.log1p(
        -1.0 / (2.0 * x_safe**2) + 3.0 / (4.0 * x_safe**4)
    )  # log(1 + ...) correction
    approx = leading + correction
    return torch.where(safe_region, direct, approx)

class Gaussian:
    def __init__(self, mu=0, sigma=1):
        self.mu = mu
        self.sigma = sigma
        self.dist = torch.distributions.normal.Normal(self.mu, self.sigma)

    def pdf(self, x):
        return self.dist.log_prob(x).exp()

    def cdf(self, x):
        return self.dist.cdf(x)
    
    def log_pdf(self, x):
        return self.dist.log_prob(x)
    
    def log_cdf(self, x):
        return np.log(0.5) + logerfc(-x / np.sqrt(2))
    
    def log_sf(self, x):
        return np.log(0.5) + logerfc(x / np.sqrt(2))
    
class Laplace:
    def __init__(self, mu=0, b=1):
        self.mu = mu
        self.b = b
        self.dist = torch.distributions.laplace.Laplace(self.mu, self.b)

    def pdf(self, x):
        return self.dist.log_prob(x).exp()

    def cdf(self, x):
        return self.dist.cdf(x)

    def log_pdf(self, x):
        return self.dist.log_prob(x)
    
    def log_cdf(self, x):
        # log(CDF) for Laplace: log(0.5) + (x - mu)/b for x < mu, log(1 - 0.5 * exp(-(x-mu)/b)) for x >= mu
        result  = torch.zeros_like(x)
        loc = x < self.mu

        result[loc] = np.log(0.5) + (x[loc] - self.mu) / self.b
        result[~loc] = log_a_minus_b(
            torch.zeros_like(x[~loc]),
            np.log(0.5) - (x[~loc] - self.mu) / self.b 
        )
        return result

    def log_sf(self, x):
        return log_a_minus_b(
            torch.zeros_like(x),
            self.log_cdf(x)
        )
    
class eps_delta:
    def __init__(self, eps=0, delta=1):
        self.eps = eps
        self.delta = delta
        # self.dist = torch.distributions.laplace.Laplace(self.mu, self.b)

    def pdf(self, x):
        return self.dist.log_prob(x).exp()

    def cdf(self, x):
        return self.dist.cdf(x)

    def log_pdf(self, x):
        return self.dist.log_prob(x)
    
    def log_cdf(self, x):
        pass

    def log_sf(self, x):
        pass