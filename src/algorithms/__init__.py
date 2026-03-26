"""
Algorithm registry.
"""

from src.algorithms.balls_and_bins import BallsAndBins
from src.algorithms.algo2 import RemoveSinks
from src.algorithms.algo3 import RemoveSinksSubsampled

ALGORITHMS = {
    1: BallsAndBins,
    2: RemoveSinks,
    3: RemoveSinksSubsampled,
}


def get_algorithm(algo_id: int, **kwargs):
    """Instantiate an algorithm by ID. Extra kwargs are passed to the constructor."""
    if algo_id not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algo_id}. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[algo_id](**kwargs)
