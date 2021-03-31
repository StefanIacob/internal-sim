import numpy as np

import nengo
from utils import generate_scaling_functions

def generate():
    jac_means = [0, 0, 0, 0]
    jac_scale = [2.5, 2.5, 2.5, 2.5]
    
    jac_to_ens, ens_to_jac = generate_scaling_functions(jac_means, jac_scale)
    
    dEdq = nengo.Network()
    with dEdq:
        # Input node contains: q (unscaled), x (scaled), xt (scaled)
        dEdq.input = nengo.Node(size_in=6, size_out=6)
        def J_func(x):
            q = x
            J = np.zeros((4,))
            l = [1.5, 1.3]
            # define column entries right to left
            J[1] = l[1] * -np.sin(q[0]+q[1])
            J[3] = l[1] * np.cos(q[0]+q[1])
    
            J[0] = l[0] * -np.sin(q[0]) + J[1]
            J[2] = l[0] * np.cos(q[0]) + J[3]
            return jac_to_ens(J)
            
        dEdq.q = nengo.Ensemble(500, 2)
        dEdq.distance_x = nengo.Ensemble(100, 2)
        dEdq.dedq1 = nengo.Ensemble(500,6)
        
        # Connect Input
        nengo.Connection(dEdq.input[:2], dEdq.q) # Current q
        nengo.Connection(dEdq.input[2:4], dEdq.distance_x) # Current x
        nengo.Connection(dEdq.input[4:6], dEdq.distance_x, transform=-1) # Target x
        
        # Connections
        nengo.Connection(dEdq.distance_x, dEdq.dedq1[:2]) # Distance from x to target x 
        nengo.Connection(dEdq.q, dEdq.dedq1[2:], function=J_func) # Compute jacobian from q
    
        dEdq.dedq2 = nengo.Ensemble(500,2)
        
        def dist_times_jac(x):
            dist = np.array(x[:2])
            jac = np.array(x[2:])
            jac = jac.reshape((2,2))
            E = np.matmul(dist,jac)
            return E.reshape(2,)
        
        # Multiply distance in x with Jacobian to obtain error gradient
        nengo.Connection(dEdq.dedq1, dEdq.dedq2, function=dist_times_jac)
        
        # Output node contains: qt, distance_x
        dEdq.output = nengo.Node(size_in=4, size_out=4)
        
        # Connect Output
        nengo.Connection(dEdq.dedq2, dEdq.output[:2]) # Error gradient dE/dq
        nengo.Connection(dEdq.distance_x, dEdq.output[2:])  # Distance in x
    model = dEdq
    return model
