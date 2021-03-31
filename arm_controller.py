import numpy as np
import nengo
import arm
import weightsaver
from utils import generate_scaling_functions
from Networks import FM_q_dq, x_2_q, weights


def generate_target_function(targ_set_train, targ_set_test, presentation_time_train, presentation_time_test,
                             train_time_1, train_time_2, hold_begin, hold_end):
    """
    Generates a Node output function that presents targets serially for the reaching model
    :param targ_set_train: list of 2d targets
    :param targ_set_test: list of 2d targets
    :param presentation_time_train: time each train target is shown
    :param presentation_time_test: time each test target is shown
    :param train_time: time spent in training_phase
    :return: function usable in nengo Node that outputs targets serially
    """

    def targ_func(t):
        if train_time_1 + train_time_2 + hold_end > t > train_time_1 + train_time_2 - hold_begin:
            # fixed target at the end
            return [0, 2]
        if t < train_time_1:
            return targ_set_train[int(t / presentation_time_train) % len(targ_set_train)]
        else:
            return targ_set_test[int(t / presentation_time_test) % len(targ_set_test)]

    return targ_func


def generic_switch_node_func(t, x):
    s = x[0]
    return s * x[1:]


def generate(train_set, test_set, experiment_group=0, time_train_1=50, time_train_2=150,
             time_train_sim=0, begin_hold=5, end_hold=25, present_time_train=10,
             present_time_test=5, fm_output_neurons=3000, sim_switch_time=5, probes=True,
             load_weights=False, seed=0, tau=0.05, delta=0.0002, c=0):
    """
    Generates nengo network for learning reaching behaviour
    :param train_set: Target set used during training period
    :param test_set:  Target set used during testing period
    :param experiment_group: 0 for control, 1 for internal sim, 2-5 for internal sim + varying lesion sizes, 6 for train
    :param time_train_1: time spent for isolated training of FM
    :param time_train_1: time spent for training FM and IM together
    :param present_time_train: time per target during train phase
    :param present_time_test: time per target during test phase
    :param fm_output_neurons: Number of neurons in FM output layer
    :param probes: If true, measures relevant network data using probes
    :return: nengo network
    """

    targ_func = generate_target_function(train_set, test_set, present_time_train, present_time_test, time_train_1,
                                         time_train_2, begin_hold, end_hold)

    start_pos = np.array([0, 1])
    arm_sim = arm.Arm2Link(dt=1e-3)  # Make use of arm from REACH (src: TO DO)

    # set the initial position of the arm
    arm_sim.init_q = arm_sim.inv_kinematics(start_pos)
    arm_sim.reset()

    # Scaling functions
    q_scale = 2.9 / 2
    # q_means = [np.pi / 2, np.pi / 2]
    q_means = 3.1/2
    x_means = start_pos
    x_scale = 2

    dq_means = np.array([0, 0])
    dq_scale = [np.pi * 2, np.pi * 2]  # TODO: change scale

    u_means = np.array([0, 0])
    u_scale = np.array([18, 14])  # need to tweak

    q_to_ens, ens_to_q = generate_scaling_functions(q_means, q_scale)
    x_to_ens, ens_to_x = generate_scaling_functions(x_means, x_scale)
    dq_to_ens, ens_to_dq = generate_scaling_functions(dq_means, dq_scale)
    u_to_ens, ens_to_u = generate_scaling_functions(u_means, u_scale)

    net = nengo.Network(seed=seed)
    with net:
        def global_sim_flag_function(t):
            if experiment_group > 0 and not (experiment_group == 6):
                start = time_train_1 + time_train_2
                weight = np.maximum(0, t - start)
                weight = np.minimum(weight, sim_switch_time)
                weight = 1 - ((weight) / sim_switch_time)
                return weight
            else:
                return 1

        # global internal simulation flag switch
        net.global_sim_flag = nengo.Node(output=global_sim_flag_function)

        # ======= Inverse Model ======= #
        # arm output: [q1, q2, dq1, dq2, x, y, u1, u2]
        net.dim = arm_sim.DOF
        net.arm_node = arm_sim.create_nengo_node()
        net.target_x = nengo.Node(targ_func, size_out=2)
        net.i1_q_dq_qt = nengo.Ensemble(300, 6)
        net.i2_u = nengo.Ensemble(300, 2)
        net.error_qu = nengo.Ensemble(200, 2)

        # Target transformation xt to qt
        net.xt_2_qt = x_2_q.generate(arm_sim.l)
        nengo.Connection(net.target_x, net.xt_2_qt.input)

        if load_weights:
            weights = np.load('Networks/weights/inverse.npy').T
        else:
            weights = np.random.uniform(size=(2, 6))

        if load_weights:
            net.inverse_q_u = nengo.Connection(net.i1_q_dq_qt.neurons, net.i2_u, transform=weights,
                                               learning_rule_type=nengo.PES(),
                                               synapse=tau)
        else:
            net.inverse_q_u = nengo.Connection(net.i1_q_dq_qt, net.i2_u, transform=weights,
                                               learning_rule_type=nengo.PES(),
                                               synapse=tau)
        net.ws_IM = weightsaver.WeightSaver(net.inverse_q_u, 'Networks/weights/inverse.npy')
        nengo.Connection(net.xt_2_qt.output, net.i1_q_dq_qt[4:])
        nengo.Connection(net.i2_u, net.arm_node[:2], function=ens_to_u)
        nengo.Connection(net.xt_2_qt.output, net.error_qu, transform=-1)
        nengo.Connection(net.arm_node[:2], net.error_qu, function=q_to_ens)
        nengo.Connection(net.i2_u, net.inverse_q_u.learning_rule, transform=1)

        # Switch 
        net.arm_to_IM_switch = nengo.Node(size_in=5, size_out=4, output=generic_switch_node_func)
        nengo.Connection(net.global_sim_flag, net.arm_to_IM_switch[0])
        nengo.Connection(net.arm_node[:2], net.arm_to_IM_switch[1:3], function=q_to_ens, synapse=0)
        nengo.Connection(net.arm_node[2:4], net.arm_to_IM_switch[3:5], function=dq_to_ens, synapse=0)
        nengo.Connection(net.arm_to_IM_switch[:2], net.i1_q_dq_qt[:2])
        nengo.Connection(net.arm_to_IM_switch[2:4], net.i1_q_dq_qt[2:4])

        # ======= Forward Model connections ======= #
        def FM_learn_func(t):
            if t < time_train_1 + time_train_2:
                return 1
            else:
                return 0

        net.FM = FM_q_dq.generate(fm_output_neurons, tau, probes, load_weights,
                                  seed=seed)  # Input: q1, q2, dq1, dq2, x, y, u1, u2, s
        net.FM_learn = nengo.Node(output=FM_learn_func)
        net.error_qu_next = nengo.Ensemble(100, 2)

        if probes:
            net.probe_error = nengo.Probe(net.error_qu)
            net.probe_arm = nengo.Probe(net.arm_node)
            net.probe_next_error = nengo.Probe(net.error_qu_next)
            net.probe_fm = nengo.Probe(net.FM.output)

        net.arm_to_FM_switch = nengo.Node(size_in=5, size_out=4, output=generic_switch_node_func)
        nengo.Connection(net.global_sim_flag, net.arm_to_FM_switch[0])
        nengo.Connection(net.arm_node[:2], net.arm_to_FM_switch[1:3], function=q_to_ens, synapse=c)
        nengo.Connection(net.arm_node[2:4], net.arm_to_FM_switch[3:5], function=dq_to_ens, synapse=c)
        nengo.Connection(net.arm_to_FM_switch, net.FM.input[:4], synapse=0)
        nengo.Connection(net.i2_u, net.FM.input[6:8])
        nengo.Connection(net.FM_learn, net.FM.input[8])
        nengo.Connection(net.FM.output[:2], net.error_qu_next)
        nengo.Connection(net.xt_2_qt.output, net.error_qu_next, transform=-1)

        # ====== FM Loop ====== #
        net.FM_loop_state = nengo.Node(size_in=5, size_out=4, output=generic_switch_node_func)

        nengo.Connection(net.global_sim_flag, net.FM_loop_state[0], function=lambda x: 1 - x)
        nengo.Connection(net.FM.output, net.FM_loop_state[1:5])
        nengo.Connection(net.FM_loop_state, net.FM.input[:4], synapse=tau - delta)
        nengo.Connection(net.FM_loop_state, net.i1_q_dq_qt[:4], synapse=0)

        def next_func(t, x):
            """
            Selects between two incoming error signals (one based on FM and one based on current values from the arm).
            Depends on time (training/test)
            :param t:
            :param x:
            :return:
            """
            e1_c, e2_c, e1_n, e2_n = x
            if t < time_train_1:
                return e1_c, e2_c
            else:
                return e1_n, e2_n

        net.next_switch = nengo.Node(next_func, size_in=4, size_out=2)
        nengo.Connection(net.error_qu, net.next_switch[:2])
        nengo.Connection(net.error_qu_next, net.next_switch[2:4])
        nengo.Connection(net.next_switch, net.inverse_q_u.learning_rule)

        # ====== Lesion module ====== #
        def lesion_func(t):
            if t < time_train_1 + time_train_2 + time_train_sim + sim_switch_time:
                return 0
            else:
                if experiment_group > 1 and not (experiment_group == 6):
                    return -1
                else:
                    return 0

        net.lesion_current = nengo.Node(output=lesion_func)
        lesioned_neurons = np.zeros((fm_output_neurons, 1))
        # Choose which neurons are going to be lesioned
        # Lesion percentage depends on condition/group
        # condition 2: 20%
        # condition 3: 40%
        # condition 4: 60%
        # condition 5: 80%
        lesion_percentage = max(0, 2 * (experiment_group - 1) / 10)
        nr_of_lesions = int(lesion_percentage * fm_output_neurons)
        lesioned_ind = np.random.choice(range(fm_output_neurons), nr_of_lesions)

        for i in lesioned_ind:
            lesioned_neurons[i, 0] = 1

        nengo.Connection(net.lesion_current, net.FM.f2_q_next.neurons, transform=lesioned_neurons)
        nengo.Connection(net.lesion_current, net.FM.f2_dq_next.neurons, transform=lesioned_neurons)

    model = net

    return model
