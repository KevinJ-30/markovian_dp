# from scipy.special import binom
from scipy.stats import binom
import torch
import numpy as np
import math
import time
import os
from tqdm import tqdm
from pathlib import Path
import sys
import os
# import scipy    

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype_float = torch.float64
dtype_int = torch.int32

import privacy.distribution as distribution

def log_factorial(n):
    return torch.lgamma(n + 1)

def eps_from_delta_rdp(orders, rdp, delta):
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values at
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)

    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - (np.log(delta) + np.log(orders_vec)) / (orders_vec - 1)
        + np.log((orders_vec - 1) / orders_vec)
    )

    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    # print(f'best alpha: {orders_vec[idx_opt]}')

    return eps[idx_opt], orders_vec[idx_opt]


class divergence_computer:
    def __init__(self, q_b, steps, D_out, M_train=1, saving_path=None, device="cpu"):
        assert q_b > 0 and q_b <= 1, f'q_b should be in (0, 1], but got {q_b}'
        assert steps >= 0, f'steps must be non-negative, but got {steps}'
        assert D_out > 0, f'D_out should be greater than 0, but got {D_out}'
        assert q_b * M_train / D_out <= 1, 'q_b * M_train / D_out must not exceed one'

        self.q_b = q_b
        self.steps = int(steps)
        self.D_out = D_out
        self.M_train = M_train
        self.saving_path = saving_path
        self.device = torch.device(device)
        # self.probs = self.generate_probs()

        ''''''
        self.G = distribution.Gaussian(mu = 0, sigma = 1)
        self.alphas = torch.tensor([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 30)))
        # self.alpha = 3

        ''''''
        TN_left = 1e-8
        TN_right = 1 - TN_left
        TN_points = int(1e6)
        self.TN = torch.linspace(TN_left, TN_right, TN_points, device=self.device, dtype=dtype_float)
        self.log_inter_length = torch.log(self.TN[1] - self.TN[0])  # log of the interval length for TN

        max_rdp = 0
        d_out_at_max_rdp = None

        '''the result can be stored in a file, but here we compute it directly'''
        print(f'Computing RDP for D_out from 1 to {D_out} to find the worst case...')
        s_time = time.time()
        for d_out in range(1, D_out + 1):
            
            ''''''
            self.prob_trucate_th = -30
            self.sens = self.generate_sens(d_out)
            self.log_prob = self.compute_log_probs(d_out)

            '''normalize log_prob'''
            self.log_prob = self.normalize_log_prob(self.log_prob, 0)
            loc = self.log_prob >= self.prob_trucate_th
            # print(f'loc: {loc.shape}, log_prob: {self.log_prob.shape}, sens: {self.sens.shape}')
            self.sens = self.sens[loc]
            self.log_prob = self.log_prob[loc]

            rdp = self.rdp_from_noise_simple(sigma=1, delta=1e-5, show_flag=False)
            if rdp > max_rdp:
                max_rdp = rdp
                d_out_at_max_rdp = d_out
            # print(f'D_out = {d_out}, RDP = {rdp:.4f}, max RDP = {max_rdp:.4f}')
        print(f'Max RDP {max_rdp:.4f} at D_out = {d_out_at_max_rdp}, computation time: {time.time() - s_time:.2f} seconds')

        self.prob_trucate_th = -30
        self.sens = self.generate_sens(d_out_at_max_rdp)
        self.log_prob = self.compute_log_probs(d_out_at_max_rdp)

        '''normalize log_prob'''
        self.log_prob = self.normalize_log_prob(self.log_prob, 0)
        loc = self.log_prob >= self.prob_trucate_th
        self.sens = self.sens[loc]
        self.log_prob = self.log_prob[loc]
        


    def normalize_log_prob(self, log_density, log_interval_length):
        """
        Normalize the log probability density function.
        """
        log_density = log_density - torch.logsumexp(log_density + log_interval_length, dim=0)
        return log_density

    def density(self, TN, sigma = None):
        x = distribution.G.icdf(TN)
        x = x.reshape(1, -1)

        mu = self.sens / sigma  # Ensure x is a column vector
        mu = mu.reshape(-1, 1)

        log_prob = self.log_prob.reshape(-1, 1)  # Ensure log_prob is a column vector
        log_prob = log_prob.repeat(1, x.shape[1])  # Repeat log_prob to match x's shape

        log_den = 0.5 * (2 * x - mu) * mu + log_prob
        log_den = torch.logsumexp(log_den, dim=0)

        log_den = self.normalize_log_prob(log_den, self.log_inter_length)
        return log_den


    def eps_from_noise(self, sigma, delta = 1e-5, show_flag = True):


        def rdp_per_alpha(alpha):
            log_each_term = self.density(self.TN, sigma) * (1 - alpha) + self.log_inter_length
            # print(f' shape of log_each_term: {log_each_term.shape}, TN shape: {self.TN.shape}, sigma: {sigma:.8f}')
            log_each_term = log_each_term.reshape(-1, 1)  # Ensure log_each_term is a column vector
            log_integral = torch.logsumexp(log_each_term, dim = 0)  # log of the integral
            rdp = log_integral / (alpha - 1)
            rdp = rdp.item()  # Convert to scalar
            # print(f'==> rdp: {rdp:.8f}, alpha: {alpha:.8f}, sigma: {sigma:.8f}')

            log_each_term = self.density(self.TN, sigma) * (alpha) + self.log_inter_length
            # print(f' shape of log_each_term: {log_each_term.shape}, TN shape: {self.TN.shape}, sigma: {sigma:.8f}')
            log_each_term = log_each_term.reshape(-1, 1)  # Ensure log_each_term is a column vector
            log_integral = torch.logsumexp(log_each_term, dim = 0)  # log of the integral
            rdp_2 = log_integral / (alpha - 1)
            rdp_2 = rdp_2.item()  # Convert to scalar

            rdp = max(rdp, rdp_2)
            return rdp * self.steps
        
        orders = self.alphas.cpu().numpy()
        rdp = np.array([rdp_per_alpha(alpha) for alpha in orders])

        eps, alpha = eps_from_delta_rdp(orders, rdp, delta)
        return  eps, alpha
    
    def rdp_from_noise_simple(self, sigma, delta = 1e-5, show_flag = True):

        alpha = 1.3
        log_each_term = self.density(self.TN, sigma) * (1 - alpha) + self.log_inter_length
        # print(f' shape of log_each_term: {log_each_term.shape}, TN shape: {self.TN.shape}, sigma: {sigma:.8f}')
        log_each_term = log_each_term.reshape(-1, 1)  # Ensure log_each_term is a column vector
        log_integral = torch.logsumexp(log_each_term, dim = 0)  # log of the integral
        rdp = log_integral / (alpha - 1)
        rdp = rdp.item()  # Convert to scalar
        # print(f'==> rdp: {rdp:.8f}, alpha: {alpha:.8f}, sigma: {sigma:.8f}')

        log_each_term = self.density(self.TN, sigma) * (alpha) + self.log_inter_length
        # print(f' shape of log_each_term: {log_each_term.shape}, TN shape: {self.TN.shape}, sigma: {sigma:.8f}')
        log_each_term = log_each_term.reshape(-1, 1)  # Ensure log_each_term is a column vector
        log_integral = torch.logsumexp(log_each_term, dim = 0)  # log of the integral
        rdp_2 = log_integral / (alpha - 1)
        rdp_2 = rdp_2.item()  # Convert to scalar

        rdp = max(rdp, rdp_2)
        return rdp * self.steps
    
    def noise_from_eps(self, eps, delta = 1e-5, verbose = True):
        '''return sigma'''
        '''using binary search'''
        def calculate():
            sigma_small = 0.001
            sigma_large = 100
            sigma = (sigma_small + sigma_large) / 2
            while sigma_large - sigma_small > 1e-2:
                tmp_eps, _ = self.eps_from_noise(sigma = sigma, delta = delta, show_flag = False)
                if tmp_eps > eps:
                    sigma_small = sigma
                else:
                    sigma_large = sigma
                sigma = (sigma_small + sigma_large) / 2
            return sigma

        print(f'privacy accounting...')
        return calculate()


    def generate_sens(self, d_out):
        sens =  torch.tensor(range(d_out+1))
        sens = sens.tolist()
        sens = [sens[0], 0.5] + sens[1:]
        sens = torch.tensor(sens, device=self.device, dtype=dtype_float)
        return sens
    
    def compute_log_probs(self, d_out):
        if d_out == 0:
             raise ValueError(f'D_out should be greater than 0, but got {d_out}')
        
        choosing_prob = self.q_b * self.M_train / d_out #/ np.log(self.D_out + 1)

        ns =  torch.arange(0, d_out + 1, dtype=torch.int32)
        log_of_factorials = log_factorial(ns)
        log_of_factorials = log_of_factorials.reshape(-1)

        log_binom = log_of_factorials[-1] - log_of_factorials - log_of_factorials.flip( dims = (0,) ) 
        log_prob_1 = ns.view(-1) * np.log(choosing_prob)
        log_prob_2 = (d_out - ns).view(-1) * np.log(1 - choosing_prob)
        log_prob = np.log(1 - self.q_b) + log_binom + log_prob_1 + log_prob_2

        log_prob = log_prob.reshape(-1).tolist()
        log_prob = log_prob[:1] + [np.log(self.q_b)] + log_prob[1:]
        log_prob = torch.tensor(log_prob, device=self.device, dtype=dtype_float)
        assert torch.all(log_prob <= 0), "All entries in log_prob should be less or equal to 0"
        return log_prob

def check_largest_eps():
    delta = 1e-5
    sigma = 2
    comp = divergence_computer(
        q_b=0.1, steps=10, D_out=1, M_train=1, saving_path=Path(__file__).parent,
    )
    eps_1, _ = comp.eps_from_noise(sigma=sigma, delta=delta)
    print(f'eps for D_out=1: {eps_1:.4f}')
    max_eps = eps_1
    for D_out in range(1, 100000):
        comp = divergence_computer(
            q_b=0.1, steps=10, D_out=int(D_out), M_train=1, saving_path=Path(__file__).parent,
        )
        eps, _ = comp.eps_from_noise(sigma=sigma, delta=delta)
        print(f'eps for D_out={D_out}: {eps:.8f}, eps_max: {max_eps:.8f}')
        assert eps >= max_eps or abs(eps - max_eps) < 1e-2, (
            f'eps for D_out={D_out} is smaller than eps for D_out=1'
        )
        max_eps = max(max_eps, eps)


if __name__ == "__main__":
    ''''''
    computer = divergence_computer(
        q_b=0.2,
        steps=45,
        D_out=20000,
        M_train=1,
        saving_path=Path(__file__).parent,
    )

    delta = 1e-5
    eps, alpha = computer.eps_from_noise(sigma = 1.65, delta = delta)
    print(f'eps: {eps}, alpha: {alpha}')

    eps = 2
    sigma = computer.noise_from_eps(eps, delta)
    print(f'sigma: {sigma}')    

    # Check eps for D_out from 1 to 1,000,000 and compare to D_out=1
    ''''''
    # check_largest_eps()