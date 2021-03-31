from utils import generate_scaling_functions
import nengo
import numpy as np

q_means = [np.pi/2, np.pi/2]
q_scale = np.pi/2
q_to_ens, ens_to_q = generate_scaling_functions(q_means, q_scale)


def generate(l):
    inv_kin = nengo.Network()
    with inv_kin:
        def inverse_kinematics(x):
            x_sign = np.sign(x[0])
            y_sign = np.sign(x[1])
            
            cos_q2 = (x[0]**2 + x[1]**2 - l[0]**2 - l[1]**2)/(2*l[0]*l[1])
            cos_q2 = np.maximum(cos_q2, -1)
            cos_q2 = np.minimum(cos_q2, 1)
            q2 = np.arccos(cos_q2)
            q1 = np.arctan(x[1]/(x[0] + 0.00001)) - np.arctan((l[1] * np.sin(q2))/(l[0] + l[1] * np.cos(q2)))
            
            q = np.zeros((2,))
            if x_sign > 0 and y_sign > 0:
                q = np.array([q1, q2])
            
            if x_sign < 0 and y_sign > 0:
                q = np.array([q1 + np.pi, q2])
                
            if x_sign > 0 and y_sign < 0:
                q = np.array([q1, q2])
            
            if x_sign < 0 and y_sign < 0:
                q = np.array([q1 + np.pi, q2])
                                
            return q 
        
        inv_kin.input = nengo.Node(size_in=2, size_out=2)
        inv_kin.x_to_q_1 = nengo.Ensemble(500, 2, radius=4)
        inv_kin.x_to_q_2 = nengo.Ensemble(500, 2, radius = 4)
        inv_kin.output = nengo.Node(size_in=2, size_out=2)
        nengo.Connection(inv_kin.input, inv_kin.x_to_q_1)
        nengo.Connection(inv_kin.x_to_q_1, inv_kin.x_to_q_2, function = inverse_kinematics)
        nengo.Connection(inv_kin.x_to_q_2, inv_kin.output, function=q_to_ens, synapse=0.1)
    return inv_kin
