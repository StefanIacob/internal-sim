import nengo
import numpy as np

class WeightSaver(object):
    def __init__(self, connection, filename, sample_every=10, weights=False):
        # assert isinstance(connection.pre, nengo.Ensemble) or isinstance(connection.pre, nengo.Ensemble.neurons)
        if not filename.endswith('.npy'):
            filename = filename + '.npy'
        self.filename = filename
        # connection.solver = LoadFrom(self.filename, weights=weights)
        self.probe = nengo.Probe(connection, 'weights', sample_every=sample_every)
        self.connection = connection

    def save(self, sim):
        weights = sim.data[self.probe][-1].T
        np.save(self.filename, weights)
