"""Privacy accounting using Opacus accountants (RDP or PRV / PLD-based)."""


def compute_epsilon(noise_multiplier, sample_rate, num_steps, delta,
                    accountant='rdp'):
    """
    Compute epsilon for a given noise multiplier and training configuration.

    Args:
        noise_multiplier: sigma / C ratio used during training.
        sample_rate: fraction of dataset in each DP step (1/num_bins or
                     (1 - subsample_prob) / num_bins for Algo 3).
        num_steps: total number of training steps.
        delta: target delta for (epsilon, delta)-DP.
        accountant: 'rdp' (default) or 'prv'. PRV is the Privacy Random
                    Variable accountant (PLD-based), which gives tighter
                    epsilon bounds for the subsampled Gaussian than RDP.

    Returns:
        epsilon: computed privacy budget.
    """
    if accountant == 'rdp':
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
    elif accountant == 'prv':
        from opacus.accountants import PRVAccountant
        acc = PRVAccountant()
    else:
        raise ValueError(
            f"Unknown accountant '{accountant}'. Supported: 'rdp', 'prv'."
        )
    for _ in range(num_steps):
        acc.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
    return acc.get_epsilon(delta=delta)
