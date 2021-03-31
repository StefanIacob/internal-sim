import numpy as np
import nengo
from arm_controller import generate

dist1 = 2.5
dist2 = 1.5

train_targ = [np.array([np.sqrt((dist1**2)/2), np.sqrt((dist1**2)/2)]),
              np.array([-np.sqrt((dist1**2)/2), np.sqrt((dist1**2)/2)]),
              np.array([0.9, np.sqrt(dist1**2 - 0.9**2)]),
              np.array([-0.9, np.sqrt(dist1**2 - 0.9**2)]),
              np.array([0, 2.5]),
              np.array([0, 1.5]),
              np.array([np.sqrt((dist2**2)/2), np.sqrt((dist2**2)/2)]),
              np.array([-np.sqrt((dist2**2)/2), np.sqrt((dist2**2)/2)]),
              np.array([0.5, np.sqrt(dist2**2 - 0.5**2)]),
              np.array([-0.5, np.sqrt(dist2**2 - 0.5**2)]),
              ]

# GUI CODE
initial_train_time = 20
secondary_train_time = 200
time_until_lesion = 0
measure_time = 80
pres_train = 10
pres_test = 10
sim_switch_time = 5
folder = '27-10-2020'
np.random.shuffle(train_targ)
test_targ = list(np.random.uniform([-2.5, 0.5], [2.5, 2.5], size=(10, 2)))

condition=0
# model = generate(train_targ, test_targ, condition, initial_train_time, secondary_train_time, time_until_lesion,
#                          pres_train, pres_test, sim_switch_time=sim_switch_time, probes=True, load_weights=False)

model = generate(train_targ, test_targ, condition, initial_train_time, secondary_train_time, time_until_lesion,
                         5, 25, pres_train, pres_test, sim_switch_time=sim_switch_time, probes=True,
                         load_weights=False, seed=0, tau=0.065, delta=-0.0004, c=0)
# # WEIGHT TRAIN + SAVE CODE
# np.random.shuffle(train_targ)
# test_targ = list(np.random.uniform([-2.5, 0.8], [2.5, 2.5], size=(10, 2)))
# model = generate(train_targ, test_targ, 6, 20, 70, 0, 10, 10, sim_switch_time=5, probes=False, load_weights=False)
# sim = nengo.Simulator(model)
# sim.run(20 + 70)
# model.ws_IM.save(sim)
# model.FM.ws_FM_q.save(sim)
# model.FM.ws_FM_dq.save(sim)


# EXPERIMENT CODE
# initial_train_time = 20
# secondary_train_time = 80
# time_until_lesion = 0
# measure_time = 50
# pres_train = 10
# pres_test = 10
# sim_switch_time = 2
# folder = '1-12-2020'
# np.random.shuffle(train_targ)
# test_targ = list(np.random.uniform([-2.5, 0.5], [2.5, 2.5], size=(10, 2)))
# tau = 0.0900
# delta = 0.0002
# for run in range(1, 3):
#     seed = np.random.randint(1000)
#     for condition in range(6):

#         print('run ', run , ', condition ', condition)
#         np.random.shuffle(train_targ)
#         test_targ = list(np.random.uniform([-2.5, 0.5], [2.5, 2.5], size=(10, 2)))


#         model = generate(train_targ, test_targ, condition, initial_train_time, secondary_train_time, time_until_lesion,
#                          pres_train, pres_test, sim_switch_time=sim_switch_time, probes=True, load_weights=False,
#                          seed=seed, tau=tau, delta=delta)
#         sim = nengo.Simulator(model)
#         sim.run(initial_train_time + secondary_train_time + measure_time)
#         arm_data = sim.data[model.probe_arm][:, :]
#         error_data = sim.data[model.probe_error][:, :]
#         next_error_data = sim.data[model.probe_next_error][:, :]
#         fm_data = sim.data[model.probe_fm][:, :]
#         fm_error_data = sim.data[model.FM.probe_fm_error][:, :]
#         print('Saving data...')
#         np.save("Exp_results/" + str(folder) + "/condition_" + str(condition) + "/" + str(run) + "/arm_data.npy", arm_data)
#         np.save("Exp_results/" + str(folder) + "/condition_" + str(condition) + "/" + str(run) + "/error_data.npy", error_data)
#         np.save("Exp_results/" + str(folder) + "/condition_" + str(condition) + "/" + str(run) + "/next_error_data.npy", next_error_data)
#         np.save("Exp_results/" + str(folder) + "/condition_" + str(condition) + "/" + str(run) + "/fm_data.npy", fm_data)
#         np.save("Exp_results/" + str(folder) + "/condition_" + str(condition) + "/" + str(run) + "/fm_error_data.npy", fm_error_data)
#         print('Done!')
