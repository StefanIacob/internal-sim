import numpy as np
import scipy.special

import nengo

def generate_scaling_functions(means, scales, filename=None):
    """ Generates functions to scale the input to neural ensembles.
    You can either pass in the means and scales or have them calculated
    from a npz file storing the input to the ensemble.
    """

    if filename is not None:
        data = np.load(filename)['arr_0']
        means = np.mean(data, axis=0)
        scales = np.max(data, axis=0) - np.min(data, axis=0)

    def scale_down(x):
        x = (x - means) / scales
        return x
    scale_up = lambda x: x * scales + means

    return scale_down, scale_up


class AreaIntercepts(nengo.dists.Distribution):
    """ Generate an optimally distributed set of intercepts in
    high-dimensional space.
    """
    dimensions = nengo.params.NumberParam('dimensions')
    base = nengo.dists.DistributionParam('base')

    def __init__(self, dimensions, base=nengo.dists.Uniform(-1, 1)):
        super(AreaIntercepts, self).__init__()
        self.dimensions = dimensions
        self.base = base

    def __repr(self):
        return ("AreaIntercepts(dimensions=%r, base=%r)" %
                (self.dimensions, self.base))

    def transform(self, x):
        sign = 1
        if x > 0:
            x = -x
            sign = -1
        return sign * np.sqrt(1 - scipy.special.betaincinv(
            (self.dimensions + 1) / 2.0, 0.5, x + 1))

    def sample(self, n, d=None, rng=np.random):
        s = self.base.sample(n=n, d=d, rng=rng)
        for i in range(len(s)):
            s[i] = self.transform(s[i])
        return s
