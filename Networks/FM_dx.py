import numpy as np

import nengo
from utils import generate_scaling_functions

def generate(direct_mode=False, means=None, scales=None):
    Forward_Model = nengo.Network()
    with Forward_Model:
        
        def switch_learn_func(t, x):
            s, e1, e2 = x
            if s > 0.5:
                # train stage
                return e1, e2
            else:
                # test stage
                return 0, 0
            
        Forward_Model.input = nengo.Node(size_in=9, size_out=9)  # q, dq, x, u, s
        Forward_Model.f1_u_q_dq = nengo.Ensemble(3000, 6)
        Forward_Model.f2_dx_next = nengo.Ensemble(3000, 2)
        Forward_Model.dx = nengo.Ensemble(400, 2)
        Forward_Model.error_udx = nengo.Ensemble(400, 2)
        Forward_Model.switch_learn = nengo.Node(output=switch_learn_func, size_in=3, size_out=2)
        nengo.Connection(Forward_Model.input[6:8], Forward_Model.f1_u_q_dq[:2])
        nengo.Connection(Forward_Model.input[:2], Forward_Model.f1_u_q_dq[2:4])
        nengo.Connection(Forward_Model.input[2:4], Forward_Model.f1_u_q_dq[4:6])
        nengo.Connection(Forward_Model.input[4:6], Forward_Model.dx)
        nengo.Connection(Forward_Model.input[4:6], Forward_Model.dx, synapse=0.1, transform=-1)
        nengo.Connection(Forward_Model.input[-1], Forward_Model.switch_learn[0])
        Forward_Model.forward = nengo.Connection(Forward_Model.f1_u_q_dq, Forward_Model.f2_dx_next, transform=np.random.uniform(size=(2,6)), learning_rule_type=nengo.PES())
        
        nengo.Connection(Forward_Model.dx, Forward_Model.error_udx, transform=-1)
        nengo.Connection(Forward_Model.f2_dx_next, Forward_Model.error_udx)
        nengo.Connection(Forward_Model.error_udx, Forward_Model.switch_learn[1:])
        nengo.Connection(Forward_Model.switch_learn, Forward_Model.forward.learning_rule)
        
        Forward_Model.output = nengo.Node(size_out=2,size_in=2)
        nengo.Connection(Forward_Model.f2_dx_next, Forward_Model.output)
    return Forward_Model
