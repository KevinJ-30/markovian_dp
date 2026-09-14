"""The (batch_size, epochs) -> (p1, T) conversion.

run.py accepts the schedule in data units and converts it to the step units the
accountant prices.  These tests pin the conversion itself; the CLI wiring is
exercised by the smoke runs.
"""

import pytest

from src.sparse.run import p1_for_batch, steps_for_epochs


# ---------------------------------------------------------------- p1 = B / N

def test_p1_is_batch_over_pool():
    assert p1_for_batch(512, 44906) == pytest.approx(512 / 44906)


def test_p1_reproduces_the_hardcoded_dataset_constants():
    """The values _dataset_settings.sh carries were B=512 over each train pool.

    Guards against the two drifting apart: if someone edits a p1 constant
    without editing the pool it came from, this is where it shows up.
    """
    for pool, want in [(44906, 0.011402), (153932, 0.003326),
                       (537635, 0.000952), (1254902, 0.000408)]:
        assert p1_for_batch(512, pool) == pytest.approx(want, abs=5e-7)


def test_full_batch_is_p1_one():
    assert p1_for_batch(1000, 1000) == 1.0


def test_batch_larger_than_pool_is_rejected():
    # Silently clamping would hand the accountant a p1 > 1 and price a
    # sampling rate the sampler cannot realize.
    with pytest.raises(ValueError, match="exceeds the eligible root pool"):
        p1_for_batch(1001, 1000)


def test_nonpositive_batch_is_rejected():
    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        p1_for_batch(0, 1000)


# ------------------------------------------------------------ T = epochs / p1

def test_steps_is_epochs_over_p1():
    assert steps_for_epochs(1, 0.01) == 100
    assert steps_for_epochs(10, 0.01) == 1000


def test_pool_size_cancels():
    """T depends only on the rate, so B/N is the whole story.

    Two graphs of very different size at the same batch and epoch budget differ
    in T by exactly their pool ratio, which is what makes the arms comparable.
    """
    small = steps_for_epochs(5, p1_for_batch(512, 44906))
    big = steps_for_epochs(5, p1_for_batch(512, 1254902))
    # rel=1e-2 because T is an integer: round(5/0.01140159) = 439 against an
    # exact 438.54, which alone moves the ratio by ~0.1%.
    assert big / small == pytest.approx(1254902 / 44906, rel=1e-2)


def test_roundtrip_reaches_the_requested_epochs():
    for epochs in (0.5, 1, 5, 100):
        for pool in (44906, 537635):
            p1 = p1_for_batch(512, pool)
            T = steps_for_epochs(epochs, p1)
            assert T * p1 == pytest.approx(epochs, rel=0.02)


def test_at_least_one_step():
    # An epoch budget below one step's worth of data still has to train.
    assert steps_for_epochs(0.001, 0.5) == 1


def test_invalid_rates_and_budgets_are_rejected():
    with pytest.raises(ValueError, match="p1 must be in"):
        steps_for_epochs(1, 0.0)
    with pytest.raises(ValueError, match="p1 must be in"):
        steps_for_epochs(1, 1.5)
    with pytest.raises(ValueError, match="epochs must be > 0"):
        steps_for_epochs(0, 0.5)


# ------------------------------------------------- what this is for: equal data

def test_equal_epochs_beats_equal_steps_for_comparability():
    """A fixed T is a different amount of data on every dataset.

    This is the confound the flags exist to remove: at the T=300 the drivers
    used to hardcode, PPI-large saw 3.4 passes over its training nodes and
    Amazon saw 0.12 -- a 28x difference read as a utility difference.
    """
    pools = {'ppi-large': 44906, 'amazon': 1254902}
    at_fixed_T = {k: 300 * p1_for_batch(512, v) for k, v in pools.items()}
    assert at_fixed_T['ppi-large'] == pytest.approx(3.42, abs=0.01)
    assert at_fixed_T['amazon'] == pytest.approx(0.122, abs=0.01)

    # Specified as epochs, both arms see the same data by construction.
    at_fixed_epochs = {
        k: steps_for_epochs(5, p1_for_batch(512, v)) * p1_for_batch(512, v)
        for k, v in pools.items()}
    # Exact up to the integer rounding of T (439 steps is 5.005 epochs).
    assert at_fixed_epochs['ppi-large'] == pytest.approx(5, rel=1e-2)
    assert at_fixed_epochs['amazon'] == pytest.approx(5, rel=1e-2)
