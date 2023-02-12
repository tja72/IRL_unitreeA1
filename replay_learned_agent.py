import os.path
import sys

import torch.random

sys.path.append('/home/tim/Documents/mushroom_rl_imitation_shared/')


from mushroom_rl.core.serialization import Serializable
from mushroom_rl.environments.mujoco_envs.quadrupeds import UnitreeA1
from mushroom_rl.core import Core
from mushroom_rl.utils.dataset import compute_J, compute_episodes_length
from mushroom_rl.utils.callbacks import PlotDataset

from mushroom_rl.environments.mujoco_envs.quadrupeds.unitreeA1 import interpolate_remap, interpolate_map, reward_callback
from mushroom_rl.environments.mujoco_envs.humanoids.trajectory import Trajectory


import os

import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':


# first and best agent '/home/tim/Documents/quadruped_vail_unitreeA1_only_states_2022-12-20_22-27-17'
    #                               '/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.9/0/'
    #                               'agent_epoch_54_J_986.807868.msh'
# with constraint range:
# 0.01 0/agent_epoch_112_J_960.332362 slowest
# 0.01 0/agent_epoch_198_J_988.503833 norm
# 0.1 0/agent_epoch_121_J_987.961037
#0.5 1/agent_epoch_130_J_994.569152 Best


    gamma = 0.99
    horizon = 300

    # define env and data frequencies
    env_freq = 1000  # hz, added here as a reminder
    traj_data_freq = 100  # hz, added here as a reminder
    desired_contr_freq = 100  # hz
    n_substeps = env_freq // desired_contr_freq  # env_freq / desired_contr_freq

    use_torque_ctrl = True
    use_2d_ctrl = True





    # prepare trajectory params
    traj_params = dict(traj_path="./data/dataset_only_states_unitreeA1_IRL_new2_0_optimal.npz",
                       traj_dt=(1 / traj_data_freq),
                       control_dt=(1 / desired_contr_freq))








    if use_2d_ctrl:
        traj_params["interpolate_map"] = interpolate_map  # transforms 9dim rot matrix into one rot angle
        traj_params["interpolate_remap"] = interpolate_remap  # and back
        traj_params["traj_path"] = './data/states_2023_02_10_00_08_17.npz'






    # create the environment
    env = UnitreeA1(timestep=1 / env_freq, gamma=gamma, horizon=horizon, n_substeps=n_substeps, use_torque_ctrl=use_torque_ctrl,
                    traj_params=traj_params, random_start=True,# init_step_no=1,
                    use_2d_ctrl=use_2d_ctrl, tmp_dir_name='.',
                    goal_reward="custom", goal_reward_params=dict(reward_callback=reward_callback))

    action_dim = env.info.action_space.shape[0]
    print("Dimensionality of Obs-space:", env.info.observation_space.shape[0])
    print("Dimensionality of Act-space:", env.info.action_space.shape[0])
    #TODO core zeile 183 nicht auskommentieren !!!!!!!!!!!!!!!!!!!!!!!!!
    agents = [
        Serializable.load(
            '/media/tim/929F-6E96/thesis/quadruped_gail_unitreeA1_only_states_2023-02-10_00-26-25/train_D_n_th_epoch___3/lrD___5e-05/use_noisy_targets___0/horizon___1000/gamma___0.99/0/agent_epoch_146_J_755.960853.msh'),
       ]
    for i in range(len(agents)):

        print(agents[i])
        #plot_data_callbacks = PlotDataset(env.info)
        #core = Core(mdp=env, agent=agent, callback_step=plot_data_callbacks)
        core = Core(mdp=env, agent=agents[i])
        #core.agent.policy.deterministic = False
        dataset, env_info = core.evaluate(n_episodes=1, render=True, get_env_info=True)
        A = compute_J(dataset)
        R_mean = np.mean(compute_J(dataset))
        J_mean = np.mean(compute_J(dataset, gamma=gamma))
        L = np.mean(compute_episodes_length(dataset))

        print("J_mean: ", J_mean)
        print("R_mean: ", R_mean)
        print("L_mean:", L)
    """
       Traj minimal height: -0.1845338476
   Traj max x-rotation: -0.17452686699999997
   Traj max y-rotation: -0.05434850219999987
   
   Traj minimal height: -0.1898665265
   Traj max x-rotation: -0.19923093520000013
   Traj max y-rotation: -0.1320023989000001
   """


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
much better than pictures seem. One seed is stable and gets more and more unstable but all the other mostly stable/a little double step with the fornt left feet at the end

quadruped_vail_unitreeA1_only_states_2022-12-27_16-47-07 - vail only states with torque controlwith 15 samples each with info_constraint 0.001 and 1 
Forgot to remove stricter has fallen!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    info_constraint 0.001: all relative good 2. agent, but all are getting worse -> in the end vibrating (in the middle stumble/mini steps)
    info_constraint 1.0: in the beginning (2.) really good -> getting worse but less worse than 0.001; only stumbles a litlle bit and sometimes small steps
    
quadruped_gail_unitreeA1_only_states_2022-12-27_18-58-37 - gail only states with position  with default xml (loer gain)
    Forgot to remove stricter has fallen !!!!!!!!!!!!!!!!!!!!!!
    still very shaky legs/not a nice gait. Gets a little better to the end but definitive not good

quadruped_gail_unitreeA1_2023-01-01_18-03-41 - gail with optimal torques from data model with normal has_fallen
    makes lots of double steps/unstable walk

quadruped_gail_unitreeA1_only_states_2023-01-02_02-28-16 - gail only states with position  with default xml (loer gain)
    no good gait; very unstable

quadruped_vail_unitreeA1_only_states_2023-01-02_02-32-15 - vail only states with torque controlwith 15 samples each with info_constraint 0.001 and 1 
    info_constraint 0.001 most ok in the beginning (only a little double steps/limbing) but getting worse quick; but nearly all limbing in the beginning
    info_constraint 1.0 all in the beginning okay/good; but not as good as I remember with stricter has_fallen; getting quick worse -> starting to limbing; a few in the beginning perfect
        most also limb in the beginning a little bit
    but cannot compare earlier agents cause I store only 4 per seed; and these are the ones that are good with stricter has_fallen
    




quadruped_gail_unitreeA1_2023-01-03_02-43-04 - ICH GLAUBE/NICHT SICHER: gail with kp torques with reset at every step with normal has_fallen
    okay in the beginning but starts making small steps/double steps really fast


quadruped_gail_unitreeA1_only_states_2023-01-10_16-51-33 - gail with 2D walk, normal has_fallen. 8 dir a 50k dataset; multidm obs spec
quadruped_vail_unitreeA1_only_states_2023-01-10_16-44-04 - vail info_constraint=1 with 2D walk, normal has_fallen. 8 dir a 50k dataset; multidm obs spec

quadruped_vail_unitreeA1_only_states_2023-01-11_00-12-28 - vail only states torque with info_constraint 1.0 and less strict has_fallen; one sample to epoch 150 to test differences between has_fallen; tores as many agents as first run with less strict has fallen -> test second stred agent
    seems less stable than with stricter has fallen; slight limbing
    
    
    
quadruped_gail_unitreeA1_only_states_2023-01-19_01-11-20 - new stricter has fallen, updated dataset/correct direction arrow and adjusted traj with finetuned angles, troque only states gail
quadruped_vail_unitreeA1_only_states_2023-01-19_01-22-09 - new stricter has fallen, updated dataset/correct direction arrow and adjusted traj with finetuned angles, troque only states vail 
    problem: interpolation and draw only examples/traj from the first half -> backward walking


still npc not perfect-> slightly goes to the side
quadruped_gail_unitreeA1_only_states_2023-02-06_01-43-39 - gail, torque; new one long traj file; velo as goal, fixed npc model/dataset, interpolation .......
    gail is really really good in forward and backward and doesn't get worse over time but has the same problems with sideways and diagonal
quadruped_vail_unitreeA1_only_states_2023-02-06_01-45-29 - vail info_ =1 torque; new one long traj file; velo as goal, fixed npc model/dataset, interpolation .......
    I found epochs in vail where the forward and backward walking is okay but the sideways/diagonal walking isn't really a gait. Either it just jumps on the same place, it just walks for/backwards or it's just lifting the legs but setting it down again. From my point of view it seems like the has_fallen is to strict and when lifting the legs it tilts a little bit so it wants to fix that immediatly. But that would explain only one of the three effects


quadruped_gail_unitreeA1_only_states_2023-02-10_00-26-25 - gail same as aboe but with a dataset only sidewalking
    problem: wrong reward for logging/storing agent

"""
