import nengo
import numpy as np

def generate(delay_syn=0.1):
    q_vel = nengo.Network()
    with q_vel:
        
        q_vel.input = nengo.Node(size_in=2, size_out=2) # q in ens space
        q_vel.dq = nengo.Ensemble(500, 2, radius=1)
        q_vel.output = nengo.Node(size_in=2, size_out=2) # dq in ens space
        
        nengo.Connection(q_vel.input, q_vel.dq, synapse=0)
        nengo.Connection(q_vel.input, q_vel.dq, synapse=delay_syn, transform=-1)
        nengo.Connection(q_vel.dq, q_vel.output, synapse=0)
        
    return q_vel
