import numpy as np
import nengo
import weightsaver


def generate(out_neurons, syn, probes, load_weights=False, seed=0):
    Forward_Model = nengo.Network(seed=seed)
    with Forward_Model:

        def switch_learn_func(t, x):
            s, eq1, eq2, edq1, edq2 = x
            if s > 0:
                # train stage
                return eq1, eq2, edq1, edq2
            else:
                # test stage
                return 0, 0, 0, 0

        Forward_Model.input = nengo.Node(size_in=9, size_out=9)  # q, dq, x, u, s

        Forward_Model.f1_u_dq = nengo.Ensemble(3000, 4)
        Forward_Model.f1_dq_q = nengo.Ensemble(3000, 4)
        # Forward_Model.f1_u_dq_q = nengo.Ensemble(4000, 6)


        # Forward_Model.f2_q_next = nengo.Ensemble(out_neurons, 4)
        Forward_Model.f2_q_next = nengo.Ensemble(out_neurons, 2)
        Forward_Model.f2_dq_next = nengo.Ensemble(out_neurons, 2)
        Forward_Model.error_uq_dq = nengo.Ensemble(400, 4)
        if probes:
            Forward_Model.probe_fm_error = nengo.Probe(Forward_Model.error_uq_dq)
        Forward_Model.switch_learn = nengo.Node(output=switch_learn_func, size_in=5, size_out=4)
        nengo.Connection(Forward_Model.input[6:8], Forward_Model.f1_u_dq[:2])
        nengo.Connection(Forward_Model.input[:2], Forward_Model.f1_dq_q[2:4])
        nengo.Connection(Forward_Model.input[2:4], Forward_Model.f1_u_dq[2:4])
        nengo.Connection(Forward_Model.input[2:4], Forward_Model.f1_dq_q[:2])
        nengo.Connection(Forward_Model.input[-1], Forward_Model.switch_learn[0])
        if load_weights:
            weights_q = np.load("Networks/weights/forward_q.npy").T
            weights_dq = np.load("Networks/weights/forward_q.npy").T
            Forward_Model.forward_q = nengo.Connection(Forward_Model.f1_dq_q.neurons, Forward_Model.f2_q_next,
                                                       transform=weights_q,
                                                       learning_rule_type=nengo.PES())
            Forward_Model.forward_dq = nengo.Connection(Forward_Model.f1_u_dq.neurons, Forward_Model.f2_dq_next,
                                                        transform=weights_dq,
                                                        learning_rule_type=nengo.PES())
        else:
            weights_q = np.random.uniform(size=(2, 4))
            weights_dq = np.random.uniform(size=(2, 4))

            Forward_Model.forward_q = nengo.Connection(Forward_Model.f1_dq_q, Forward_Model.f2_q_next,
                                                       transform=weights_q,
                                                       learning_rule_type=nengo.PES())
            Forward_Model.forward_dq = nengo.Connection(Forward_Model.f1_u_dq, Forward_Model.f2_dq_next,
                                                        transform=weights_dq,
                                                        learning_rule_type=nengo.PES())

        Forward_Model.ws_FM_q = weightsaver.WeightSaver(Forward_Model.forward_q, 'Networks/weights/forward_q.npy')
        Forward_Model.ws_FM_dq = weightsaver.WeightSaver(Forward_Model.forward_dq, 'Networks/weights/forward_dq.npy')

        nengo.Connection(Forward_Model.input[:4], Forward_Model.error_uq_dq, transform=-1)
        nengo.Connection(Forward_Model.f2_q_next, Forward_Model.error_uq_dq[:2], synapse=0.15 + syn - 0.001)
        nengo.Connection(Forward_Model.f2_dq_next, Forward_Model.error_uq_dq[2:], synapse=0.15 + syn - 0.001)
        # def error_scaling(x):
        #     q0, q1, dq0, dq1 = x
        #     dq0 = dq0 * 4**(1+dq0)
        #     dq1 = dq1 * 4**(1+dq1)
        #     q0 = 0.5*q0
        #     q1 = 0.5*q1
        #     return [q0, q1, dq0, dq1]
        nengo.Connection(Forward_Model.error_uq_dq, Forward_Model.switch_learn[1:])  # , function=error_scaling)
        nengo.Connection(Forward_Model.switch_learn[:2], Forward_Model.forward_q.learning_rule)
        nengo.Connection(Forward_Model.switch_learn[2:], Forward_Model.forward_dq.learning_rule)

        Forward_Model.output = nengo.Node(size_out=4, size_in=4)
        nengo.Connection(Forward_Model.f2_q_next, Forward_Model.output[:2])
        nengo.Connection(Forward_Model.f2_dq_next, Forward_Model.output[2:])
    return Forward_Model
