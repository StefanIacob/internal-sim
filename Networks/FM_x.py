import numpy as np

import nengo

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
        Forward_Model.f1_u_q_dq_x = nengo.Ensemble(4000, 8)
        Forward_Model.f2_x_next = nengo.Ensemble(4000, 2)
        Forward_Model.error_ux = nengo.Ensemble(400, 2)
        Forward_Model.switch_learn = nengo.Node(output=switch_learn_func, size_in=3, size_out=2)
        nengo.Connection(Forward_Model.input[6:8], Forward_Model.f1_u_q_dq_x[:2])
        nengo.Connection(Forward_Model.input[:2], Forward_Model.f1_u_q_dq_x[2:4])
        nengo.Connection(Forward_Model.input[2:4], Forward_Model.f1_u_q_dq_x[4:6])
        nengo.Connection(Forward_Model.input[4:6], Forward_Model.f1_u_q_dq_x[6:8])
        nengo.Connection(Forward_Model.input[-1], Forward_Model.switch_learn[0])
        Forward_Model.forward = nengo.Connection(Forward_Model.f1_u_q_dq_x, Forward_Model.f2_x_next, transform=np.random.uniform(size=(2,8)), learning_rule_type=nengo.PES())
        
        nengo.Connection(Forward_Model.input[4:6], Forward_Model.error_ux, transform=-1)
        nengo.Connection(Forward_Model.f2_x_next, Forward_Model.error_ux, synapse=0.1)
        nengo.Connection(Forward_Model.error_ux, Forward_Model.switch_learn[1:])
        nengo.Connection(Forward_Model.switch_learn, Forward_Model.forward.learning_rule)
        
        Forward_Model.output = nengo.Node(size_out=2,size_in=2)
        nengo.Connection(Forward_Model.f2_x_next, Forward_Model.output)
    return Forward_Model
