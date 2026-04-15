"""Privacy accounting using Opacus RDPAccountant."""

from opacus.accountants import RDPAccountant


def compute_epsilon(noise_multiplier, sample_rate, num_steps, delta):
    """
    Compute epsilon for a given noise multiplier and training configuration.

    Args:
        noise_multiplier: sigma / C ratio used during training
        sample_rate: fraction of dataset in each DP step (1/num_bins or
                     (1 - subsample_prob) / num_bins for Algo 3)
        num_steps: total number of training steps
        delta: target delta for (epsilon, delta)-DP

    Returns:
        epsilon: computed privacy budget
    """
    accountant = RDPAccountant()
    for _ in range(num_steps):
        accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return accountant.get_epsilon(delta=delta)
