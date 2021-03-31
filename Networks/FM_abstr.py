import numpy as np
import nengo


model = nengo.Network()

with model:
    a = nengo.Ensemble(100,1)

# def generate(syn, probes):
#     Forward_Model = nengo.Network()
#     with Forward_Model:
        
#         def switch_learn_func(t, x):
#             s, e1, e2 = x
#             if s > 0:
#                 # train stage
#                 return e1, e2
#             else:
#                 # test stage
#                 return 0, 0
            
#         Forward_Model.input = nengo.Node(size_in=9, size_out=9)  # q, dq, x, u, s
#         Forward_Model.f1_u_q_dq = nengo.Ensemble(2000, 6)
#         Forward_Model.f2_q_next = nengo.Ensemble(2000, 2)
#         Forward_Model.error_uq = nengo.Ensemble(200, 2)
#         if probes:
#             Forward_Model.probe_fm_error = nengo.Probe(Forward_Model.error_uq)
#         Forward_Model.switch_learn = nengo.Node(output=switch_learn_func, size_in=3, size_out=2)
#         nengo.Connection(Forward_Model.input[6:8], Forward_Model.f1_u_q_dq[:2])
#         nengo.Connection(Forward_Model.input[:2], Forward_Model.f1_u_q_dq[2:4])
#         nengo.Connection(Forward_Model.input[2:4], Forward_Model.f1_u_q_dq[4:6])
#         nengo.Connection(Forward_Model.input[-1], Forward_Model.switch_learn[0])
#         Forward_Model.forward = nengo.Connection(Forward_Model.f1_u_q_dq, Forward_Model.f2_q_next, transform=np.random.uniform(size=(2,6)), learning_rule_type=nengo.PES())
        
#         nengo.Connection(Forward_Model.input[:2], Forward_Model.error_uq, transform=-1)
#         nengo.Connection(Forward_Model.f2_q_next, Forward_Model.error_uq, synapse=0.15 + syn - 0.001)
#         nengo.Connection(Forward_Model.error_uq, Forward_Model.switch_learn[1:])
#         nengo.Connection(Forward_Model.switch_learn, Forward_Model.forward.learning_rule)
        
#         Forward_Model.output = nengo.Node(size_out=2,size_in=2)
#         nengo.Connection(Forward_Model.f2_q_next, Forward_Model.output)
#     return Forward_Model
