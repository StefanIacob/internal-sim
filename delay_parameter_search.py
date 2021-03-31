import numpy as np
import nengo
from arm_controller import generate
from os import path, mkdir

# Parameter list
taus = np.round(np.arange(0.08, 0.1, 0.0025), 3)
print(taus)
deltas = np.round(np.arange(0.0002, 0.0008, 0.0001), 4)
print(deltas)
c = 0
trial = 1
folder = "1-12-2020"
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

initial_train_time = 20
secondary_train_time = 80
time_until_lesion = 0
measure_time = 50
begin_hold = 5
end_hold = 25
pres_train = 10
pres_test = 10
sim_switch_time = 2
seed = 0
np.random.shuffle(train_targ)
test_x_range = [-2.5, 2.5]
test_y_range = [0.5, 2.5]
test_targ = list(np.random.uniform([test_x_range[0], test_y_range[0]], [test_x_range[1], test_y_range[1]], size=(10, 2)))

info_path = path.join('param_search', folder, 'info.txt')
info_file = open(info_path, "w")
info_file.write("Tau search space: " + str(taus))
info_file.write("\nDelta search space: " + str(deltas))
info_file.write("\nc: " + str(c))
info_file.write("\nFirst train time (IM & FM separate): " + str(initial_train_time))
info_file.write("\nSecond train time: " + str(secondary_train_time))
info_file.write("\nMeasure time: " + str(measure_time))
info_file.write("\nFixed target starts " + str(begin_hold) + " seconds before end of second train time")
info_file.write("\nFixed target ends " + str(end_hold) + " seconds after end of second train time")
info_file.write("\nDuration of flipping internal sim switch: " + str(sim_switch_time))
info_file.write("\nFirst train time targets are each presented (s): " + str(pres_train))
info_file.write("\nSecond train time targets are each presented (s): " + str(pres_test))
info_file.write("\nSecond train time targets x range: " + str(test_x_range))
info_file.write("\nSecond train time targets y range: " + str(test_y_range))
info_file.write("\nNengo seed: " + str(seed))
info_file.close()


for tau in taus:
    tau_folder = 'tau_' + str(tau)
    tau_folder = path.join('param_search', folder, tau_folder)
    if not path.isdir(tau_folder):
        mkdir(tau_folder)

    for delta in deltas:
        print("Tau: " + str(tau))
        print("Delta: " + str(delta))
        delta_folder = 'delta_' + str(delta)
        delta_folder = path.join(tau_folder, delta_folder) #, str(trial))
        if not path.isdir(delta_folder):
            mkdir(delta_folder)

        delta_folder = path.join(delta_folder, str(trial))
        if not path.isdir(delta_folder):
            mkdir(delta_folder)

        save_path = delta_folder
        np.random.shuffle(train_targ)
        test_targ = list(np.random.uniform([-2.5, 0.5], [2.5, 2.5], size=(10, 2)))
        model = generate(train_targ, test_targ, 1, initial_train_time, secondary_train_time, time_until_lesion,
                         begin_hold, end_hold, pres_train, pres_test, sim_switch_time=sim_switch_time, probes=True,
                         load_weights=False, seed=seed, tau=tau, delta=delta, c=c)
        sim = nengo.Simulator(model)
        sim.run(initial_train_time + secondary_train_time + measure_time)
        arm_data = sim.data[model.probe_arm][:, :]
        error_data = sim.data[model.probe_error][:, :]
        next_error_data = sim.data[model.probe_next_error][:, :]
        fm_data = sim.data[model.probe_fm][:, :]
        fm_error_data = sim.data[model.FM.probe_fm_error][:, :]
        print('Saving data...')
        np.save(path.join(save_path, "arm_data.npy"), arm_data)
        np.save(path.join(save_path, "error_data.npy"), error_data)
        np.save(path.join(save_path, "next_error_data.npy"), next_error_data)
        np.save(path.join(save_path, "fm_data.npy"), fm_data)
        np.save(path.join(save_path, "fm_error_data.npy"), fm_error_data)
        print('Done!')