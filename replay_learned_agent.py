import sys

import torch.random

sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.utils.callbacks import PlotDataset

import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    agent = Serializable.load('/home/tim/Documents/quadruped_gail_unitreeA1_only_states_2022-12-21_19-24-48'
                              '/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99'
                              '/2/agent_epoch_457_J_994.937137.msh')
# first and best agent '/home/tim/Documents/quadruped_vail_unitreeA1_only_states_2022-12-20_22-27-17'
    #                               '/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.9/0/'
    #                               'agent_epoch_54_J_986.807868.msh'
# with constraint range:
# 0.01 0/agent_epoch_112_J_960.332362 slowest
# 0.01 0/agent_epoch_198_J_988.503833 norm
# 0.1 0/agent_epoch_121_J_987.961037
#0.5 1/agent_epoch_130_J_994.569152 Best


    gamma = 0.99
    horizon = 1000

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 500  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq  # env_freq / desired_contr_freq

    # set a reward for logging
    reward_callback = lambda state, action, next_state: np.exp(- np.square(state[16] - 0.6))  # x-velocity as reward

    # prepare trajectory params
    traj_params = dict(traj_path="./Vail/dataset_temp_concatenated_optimal_states.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))

    # create the environment
    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=True,
                    traj_params=traj_params, random_start=True,#  init_step_no=0,
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])

    #plot_data_callbacks = PlotDataset(env.info)
    #core = Core(mdp=env, agent=agent, callback_step=plot_data_callbacks)
    core = Core(mdp=env, agent=agent)
    #core.agent.policy.deterministic = False
    dataset, env_info = core.evaluate(n_episodes=100, render=True, get_env_info=True)
    for sample in dataset:
        print("Has fallen: ", sample[4]) #sample = (state, action, reward, next_state, absorbing, last)
    R_mean = np.mean(compute_J(dataset))
    J_mean = np.mean(compute_J(dataset, gamma=gamma))
    L = np.mean(compute_episodes_length(dataset))

    print("J_mean: ", J_mean)
    print("R_mean: ", R_mean)
    print("L_mean:", L)


"""
copy agent: scp -r ta11pajo@lcluster16.hrz.tu-darmstadt.de:/work/scratch/ta11pajo/quadruped_vail_2022-12-15_02-27-23/ /home/tim/Documents/IRL_unitreeA1/
log into ssh: ssh -L 16006:127.0.0.1:6006 ta11pajo@lcluster16.hrz.tu-darmstadt.de
execute in /work/scratch/ta11pajo/<experiment>: tensorboard --logdir quadruped_gail_unitreeA1_2022-12-15_02-27-07 --port 6006
-> open http://localhost:16006
"""

"""
files in scratch:
quadruped_gail_unitreeA1_2022-12-15_02-27-07: first run - position ctrl; couldsn't find stable walking version (shaking and friction too low?)
quadruped_gail_unitreeA1_only_states_2022-12-15_11-30-08 - position; forgot to cut out init stepping
quadruped_gail_unitreeA1_only_states_2022-12-21_19-24-48 - torque ctrl; with 250k dataset
    stable; has peak at 250k whats after that; no as stable as vail for gait init
quadruped_vail_2022-12-15_02-27-23 - position ctrl; first try; couldsn't find stable walking version (shaking and friction too low?); quick okay solution gets worse then
quadruped_vail_unitreeA1_only_states_2022-12-15_11-29-47 - position; forgot to cut out init stepping
quadruped_vail_unitreeA1_only_states_2022-12-20_22-27-17 - torque control; good results but diverges/gets worse -> makes mini steps; with 250k dataset
quadruped_vail_unitreeA1_only_states_2022-12-21_19-42-53 - torque ctrl with range of info_constraints and changed gamme =.99
    view picture/discord for more details
quadruped_vail_unitreeA1_only_states_position_2022-12-21_19-45-30 - position ctrl with default info_constraint 
    shatering like before
quadruped_vail_unitreeA1_only_states_2022-12-22_22-12-17 - MIST - vergessen conda activate !!!!!!!!!!
quadruped_vail_unitreeA1_only_states_2022-12-22_23-43-19 - BUT WITH 15 SAMPLES: torque ctrl with range of info_constraints and changed gamme =.99
    info_constraint 0.01: tend to make mini steps at the end; some agents are good at the very first beginning, but all getting worse/more unsecure
    info_constraint 0.1: same like 0.01; ministeps/vibrating feet even worse
    info_constraint 0.5: same but every agent is good at the beginnin; but is getting wrse a the end/not so bad mini steps
    
quadruped_gail_unitreeA1_only_states_2022-12-23_16-11-37 - gail only states with stricter has_fallen
much better than pictures seem. One seed is stable and gets more and more unstable but all the other mostly stable/a little dopple step with the fornt left feet at the end

quadruped_vail_unitreeA1_only_states_2022-12-27_16-47-07 - vail only states with torque controlwith 15 samples each with info_constraint 0.001 and 1

quadruped_gail_unitreeA1_only_states_2022-12-27_18-58-37 - gail only states with position  with default xml (loer gain)


"""
