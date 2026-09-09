from typing_extensions import Self
from autodp.mechanism_zoo import Mechanism
import numpy as np
from core import console


class NoisyMechanism(Mechanism):
    def __init__(self, noise_scale: float):
        # "noise_scale" is the std of the noise divide by the sensitivity
        super().__init__()
        self.name = 'NoisyMechanism'
        self.params = {'noise_scale': noise_scale}

    def update(self, noise_scale: float) -> Self:
        self.params.pop('noise_scale')
        self.__init__(noise_scale, **self.params)
        return self

    def is_zero(self) -> bool:
        for alpha in range(2,100):
            if self.RenyiDP(alpha) > 0:
                return False
        return True

    def is_inf(self) -> bool:
        for alpha in range(2,100):
            if not np.isinf(self.RenyiDP(alpha)):
                return False
        return True

    def calibrate(self, eps: float, delta: float) -> float:
        if self.params['noise_scale'] == 0:
            self.update(noise_scale=1)  # to avoid is_inf being true

        console.debug('checking if the mechanism is inf or zero...')
        if np.isinf(eps) or self.is_inf() or self.is_zero():
            self.update(noise_scale=0)
            return 0.0
        if not np.isfinite(eps) or eps <= 0 or not 0 < delta < 1:
            raise ValueError('eps must be positive and finite; delta must be in (0, 1)')

        console.debug('calibration begins...')
        def privacy_loss(scale: float) -> float:
            return self.update(scale).get_approxDP(delta)

        high = max(float(self.params['noise_scale']), 1.0)
        high_epsilon = privacy_loss(high)
        for _ in range(1024):
            if high_epsilon <= eps:
                break
            high *= 2.0
            high_epsilon = privacy_loss(high)
        else:
            raise RuntimeError('calibrator could not bracket a safe noise scale')

        low = high / 2.0
        low_epsilon = privacy_loss(low)
        while high_epsilon < eps - 1e-3 and low_epsilon > eps + 1e-3:
            midpoint = (low + high) / 2.0
            midpoint_epsilon = privacy_loss(midpoint)
            if midpoint_epsilon <= eps:
                high, high_epsilon = midpoint, midpoint_epsilon
            else:
                low, low_epsilon = midpoint, midpoint_epsilon

        self.update(high)
        return high